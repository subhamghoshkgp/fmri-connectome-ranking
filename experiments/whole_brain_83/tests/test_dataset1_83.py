from pathlib import Path
import hashlib,json,sys
import numpy as np
import pytest
from scipy.io import loadmat,savemat

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'original')]
from dataset1_83 import load_dataset83
from szv3.data import synthetic_dataset
from szv3.experiment import make_splits
from run_dataset1_83 import choose,run,METHODS


def test_actual_83_loader_fisher_labels_and_saved_splits():
    d=load_dataset83(ROOT/'data/dataset1_83.mat')
    raw=loadmat(ROOT/'data/dataset1_83.mat')
    expected=np.concatenate((raw['FC_ctrl_83'],raw['FC_schz_83']))[:,d.u,d.v]
    np.testing.assert_array_equal(d.X,np.arctanh(expected.astype(float)))
    assert d.X.shape==(54,3403) and len(d.roi_names)==83
    assert d.roi_names[-1]=='brainstem' and np.bincount(d.y).tolist()==[27,27]
    cfg=json.loads((ROOT/'configs/full.json').read_text())
    saved=json.loads((ROOT/'reference/dataset1_129_protocol.json').read_text())
    splits=make_splits(d.y,cfg['cv'])
    assert splits==saved['saved_129_splits']
    for s in splits:
        tr,te=set(s['train']),set(s['test'])
        assert not tr&te and len(tr|te)==54
        assert sorted(i for f in s['inner'] for i in f['valid'])==sorted(tr)
        for inner in s['inner']:
            a,b=set(inner['train']),set(inner['valid'])
            assert not a&b and a|b==tr and not (a|b)&te


def test_source_and_representation_guards(tmp_path):
    p=tmp_path/'wrong.mat';p.write_bytes(b'wrong source')
    with pytest.raises(ValueError,match='checksum'):load_dataset83(p)
    with pytest.raises(ValueError,match='representation'):load_dataset83(ROOT/'data/dataset1_83.mat','raw')
    mat=loadmat(ROOT/'data/dataset1_83.mat')
    mat['FC_ctrl_83'][0,0,1]+=0.1
    savemat(p,{k:v for k,v in mat.items() if not k.startswith('_')})
    with pytest.raises(ValueError,match='symmetric'):load_dataset83(p,verify_source=False)


def test_exact_auc_ties_support_index_and_invalid_candidates():
    rows=[dict(candidate=0,valid=True,mean_auc=.8,se_auc=.05,mean_support=20),
          dict(candidate=1,valid=True,mean_auc=.8,se_auc=.02,mean_support=10),
          dict(candidate=2,valid=True,mean_auc=.79,se_auc=.02,mean_support=2),
          dict(candidate=3,valid=False,mean_auc=.99,se_auc=0,mean_support=0),
          dict(candidate=4,valid=True,mean_auc=.8,se_auc=.02,mean_support=10)]
    assert choose(rows,'max_auc')['candidate']==1
    assert choose(rows,'one_se')['candidate']==2


def test_parallel_all_variants_resume_train_scaling_and_manifest_guard(tmp_path):
    cfg=json.loads((ROOT/'configs/smoke.json').read_text())
    cfg['baselines']['grids']['rfe_svm']=[dict(C=.1,k=2,step=.2)]
    d=synthetic_dataset(n=60,n_rois=5)
    first=run(d,cfg,tmp_path,jobs=2,max_outer=1)
    assert not first['complete']
    assert all(first['methods'][m]['completed_folds']==1 for m in METHODS)
    paths=list(tmp_path.glob('repeat_*/fold_*/*.npz'))
    before={str(p):p.stat().st_mtime_ns for p in paths}
    run(d,cfg,tmp_path,jobs=1,max_outer=1)
    assert all(p.stat().st_mtime_ns==before[str(p)] for p in paths)
    split=make_splits(d.y,cfg['cv'])[0]
    for p in paths:
        with np.load(p) as a:
            np.testing.assert_allclose(a['scaler_mean'],d.X[split['train']].mean(0),atol=1e-12)
            np.testing.assert_allclose(d.X[split['test']]@a['raw_beta']+a['raw_intercept'],a['score'],atol=1e-9)
    final=run(d,cfg,tmp_path,jobs=1)
    assert final['complete']
    changed=json.loads(json.dumps(cfg));changed['cv']['seed']+=1
    with pytest.raises(ValueError,match='Resume rejected'):run(d,changed,tmp_path)


def test_bundled_model_sources_unchanged():
    audit=json.loads((ROOT/'validation/UNCHANGED_MODEL_SOURCES.json').read_text())
    for name,digest in audit['model_sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
