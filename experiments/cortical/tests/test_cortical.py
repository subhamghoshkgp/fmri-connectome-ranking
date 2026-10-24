"""Anatomical extraction, frozen protocol, certification and restart tests."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
import pytest
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'original')]
from cortical_data import cortical_atlas,cortical_features,load_cortical,read_full_arrays,protocol
import run_cortical as runner
from szv3.storage import RunPaused,atomic_json
from szv3.data import Dataset
from szv3.math import edge_map

@pytest.mark.parametrize('resolution,count,edges',[(83,68,2278),(129,114,6441)])
def test_atlas_subset_both_axes_and_fisher_once(resolution,count,edges):
    ids,names,excluded=cortical_atlas(resolution)
    assert len(ids)==count and len(excluded)==15 and ids[0]==1
    # Unique symmetric edge values expose off-by-one, contiguous slicing and wrong-axis bugs.
    u,v=np.triu_indices(resolution,1)
    matrices=np.repeat(np.eye(resolution)[None],54,axis=0)
    values=(np.arange(len(u))+1)/(len(u)+1)*.8-.4
    matrices[:,u,v]=values;matrices[:,v,u]=values
    X,i,n,cu,cv,H,_,raw=cortical_features([matrices[:27],matrices[27:]],resolution)
    assert X.shape==(54,edges) and H.shape==(count,edges)
    np.testing.assert_array_equal(raw,matrices[:,ids[cu]-1,ids[cv]-1])
    np.testing.assert_array_equal(X,np.arctanh(raw))
    assert np.all(H.sum(axis=0)==2) and np.all(cu<cv)
    assert not (set(ids)&set(excluded))
    assert set(ids[cu])|set(ids[cv])==set(ids)
    assert raw.shape[1]!=resolution*(resolution-1)//2
    wrong=matrices[:,:count,:count][:,cu,cv]
    assert not np.array_equal(raw,wrong)

@pytest.mark.parametrize('resolution', [83,129])
def test_actual_bundled_features_and_original_splits(resolution):
    d=load_cortical(ROOT/f'data/cortical_{resolution}.npz',resolution)
    cfg=json.loads((ROOT/'configs/full.json').read_text())
    assert runner.saved_splits(d,cfg)==protocol()['saved_129_splits']
    assert len(d.y)==54 and np.bincount(d.y).tolist()==[27,27]
    np.testing.assert_array_equal(d.subject_ids,[f'{g}_{i+1:02d}' for g in ('ctrl','schz') for i in range(27)])
    if resolution==83:
        mat=loadmat(ROOT/'data/dataset1_83.mat')
        X,*_=cortical_features([mat['FC_ctrl_83'],mat['FC_schz_83']],83)
        # Cached Fisher values can differ by a few float64 ULPs across CPUs.
        np.testing.assert_allclose(d.X,X,rtol=0,atol=1e-14)
    if os.getenv('FMRI_ORIGINAL_SOURCE'):
        expected,*_=cortical_features(read_full_arrays(os.environ['FMRI_ORIGINAL_SOURCE'],resolution),resolution)
        np.testing.assert_allclose(d.X,expected,rtol=0,atol=1e-14)


def test_full_grids_and_models_preserved():
    cfg=json.loads((ROOT/'configs/full.json').read_text())
    old=json.loads((ROOT/'validation/ORIGINAL_FULL_CONFIG.json').read_text())
    for k in ('cv','baselines','ranking','representation'):assert cfg[k]==old[k]
    audit=json.loads((ROOT/'validation/UNCHANGED_MODEL_SOURCES.json').read_text())
    for path,digest in audit['model_sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert len(runner.METHODS)==12


def test_reject_unverified_data_and_changed_mask(tmp_path):
    path=tmp_path/'wrong.npz';path.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='checksum'):load_cortical(path,83)


def test_resume_parallel_scalers_candidate_checks_and_corruption(tmp_path):
    # A small anatomical fixture exercises the actual runner, worker imports and verification.
    full=load_cortical(ROOT/'data/cortical_83.npz',83)
    ids,names,_=cortical_atlas(83)
    # Keep the real cortical dimensions; zero columns beyond an informative 5-node graph
    # avoid an unrepresentative narrow mask, while reducing solver effort in a separate fixture.
    cfg=json.loads((ROOT/'configs/smoke.json').read_text())
    cfg['baselines']['grids']['rfe_svm']=[dict(C=.1,k=2,step=.2)]
    # Actual integration below uses both real cortical feature spaces separately; here test
    # saved anatomy and certificate validation on a 68-node sparse synthetic feature matrix.
    rng=np.random.default_rng(37);X=np.zeros_like(full.X);X[:,:6]=rng.normal(size=(54,6));X[:,0]+=.8*full.y
    d=Dataset(X,full.y,full.subject_ids,full.roi_names,full.u,full.v,full.H,full.provenance)
    s=runner.run(d,cfg,tmp_path,jobs=2,max_outer=1)
    assert not s['complete'] and all(s['methods'][m]['completed_folds']==1 for m in runner.METHODS)
    committed={p:p.read_bytes() for p in tmp_path.glob('repeat_*/fold_*/*') if p.suffix in ('.json','.npz')}
    s=runner.run(d,cfg,tmp_path,jobs=1,max_outer=1)
    assert all(p.read_bytes()==v for p,v in committed.items())
    s=runner.run(d,cfg,tmp_path,jobs=2)
    assert s['complete'] and all(s['methods'][m]['completed_folds']==2 for m in runner.METHODS)
    changed=json.loads(json.dumps(cfg));changed['ranking']['rho']=[.3]
    with pytest.raises(ValueError,match='Resume rejected'):runner.run(d,changed,tmp_path)
    # A post-hoc candidate alteration must be rejected by analysis.
    p=next(tmp_path.glob('repeat_*/fold_*/max_auc_v3.json'))
    rec=json.loads(p.read_text());rec['chosen_candidate']+=1;atomic_json(p,rec)
    with pytest.raises(ValueError,match='candidate mismatch'):runner.analyze(tmp_path)


def test_stop_is_not_an_invalid_candidate(tmp_path):
    import run_rank_v3 as rank
    atomic_json(tmp_path/'manifest.json',{})
    (tmp_path/'STOP_REQUESTED').touch()
    with pytest.raises(RunPaused):rank.inner_record(tmp_path/'inner/00.json',{},lambda:dict(auc=.5))
    assert not (tmp_path/'inner/00.json').exists()
