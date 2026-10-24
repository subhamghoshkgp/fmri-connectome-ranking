import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from run_rank_v3 import choose,grids,run,fit_inner,Reference,validate_reference,RankingV3,LogisticProblem,RidgeRanker,METHODS,fold_dir
from szv3.storage import atomic_json


def tree_digest(root):
    h=hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if p.is_file():h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def test_exact_ties_and_invalid_candidates():
    rr=[dict(candidate=i,mean_auc=a,mean_support=s,valid=v) for i,a,s,v in
        [(0,.7,20,True),(1,.7,10,True),(2,.7,10,True),(3,.9,1,False)]]
    assert choose(rr)['candidate']==1
    rr[0]['mean_auc']=.7+1e-12
    assert choose(rr)['candidate']==0
    with pytest.raises(RuntimeError):choose([dict(valid=False)])


def test_full_grid_counts():
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/ranking_legacy.json').read_text())
    _,gg=grids(c)
    assert {m:len(v) for m,v in gg.items()}==dict(v3_expanded=30,v3_rank=30,v3_rank_refit=120,ridge_rank=4)


def test_saved_splits_rejected_when_inner_coverage_repeated(reference,data):
    path=reference/'manifest.json';m=json.loads(path.read_text())
    m['splits'][0]['inner'][1]=m['splits'][0]['inner'][0]
    atomic_json(path,m)
    ref=Reference(reference)
    with pytest.raises(ValueError,match='coverage'):validate_reference(ref,data)


def test_zip_and_directory_references_agree(reference,data,tmp_path):
    z=tmp_path/'reference.zip'
    with zipfile.ZipFile(z,'w') as f:
        for p in reference.rglob('*'):
            if p.is_file():f.write(p,'reselection_results/'+str(p.relative_to(reference)))
        # Different basename must not be mistaken for the main manifest.
        f.writestr('reselection_results/original_manifest.json','{}')
    a,b=Reference(reference),Reference(z)
    try:assert validate_reference(a,data)[0]==validate_reference(b,data)[0]
    finally:a.close();b.close()


def test_feature_matrix_hash_is_enforced(reference,data):
    data.X[0,0]+=.01
    ref=Reference(reference)
    try:
        with pytest.raises(ValueError,match='feature matrix'):validate_reference(ref,data)
    finally:ref.close()


def test_end_to_end_resume_source_protection(reference,data,cfg,tmp_path,monkeypatch):
    out=tmp_path/'new_run';original=tree_digest(reference)
    result=run(data,reference,cfg,out,max_outer=1)
    assert not result['complete']
    split=json.loads((reference/'manifest.json').read_text())['splits'][0]
    for method in METHODS:
        root=out/fold_dir(split)
        with np.load(root/(method+'.npz')) as a:
            np.testing.assert_allclose(a['scaler_mean'],data.X[split['train']].mean(0))
            np.testing.assert_allclose(data.X[split['test']]@a['raw_beta']+a['raw_intercept'],a['score'],atol=1e-12)
            assert set(a['train_indices']).isdisjoint(a['test_indices'])
        tuning=json.loads((root/(method+'_tuning.json')).read_text())
        assert tuning['summary'][choose(tuning['summary'])['candidate']]['valid']
        for r in tuning['inner']:
            assert set(r['train']).isdisjoint(r['valid_indices'])
            assert set(r['train']+r['valid_indices'])==set(split['train'])
            assert not set(r['train']+r['valid_indices'])&set(split['test'])
    result=run(data,reference,cfg,out)
    assert result['complete'] and all(result['methods'][m]['completed_folds']==2 for m in METHODS)
    before=tree_digest(out)
    def fail(*args,**kwargs):raise AssertionError('A completed fit was rerun')
    monkeypatch.setattr(RankingV3,'solve',fail);monkeypatch.setattr(LogisticProblem,'solve',fail)
    monkeypatch.setattr(RidgeRanker,'fit',fail)
    run(data,reference,cfg,out)
    assert tree_digest(out)==before
    assert tree_digest(reference)==original
    with pytest.raises(ValueError,match='separate'):run(data,reference,cfg,reference)
    changed=dict(cfg,rho=[.3])
    with pytest.raises(ValueError,match='Resume rejected'):run(data,reference,changed,out)


def test_invalid_selector_only_invalidates_its_candidates(reference,data,cfg,tmp_path,monkeypatch):
    split=json.loads((reference/'manifest.json').read_text())['splits'][0]
    original=RankingV3.solve;calls=0
    def intermittent(self,*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==1:raise RuntimeError('injected first-selector failure')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(RankingV3,'solve',intermittent)
    root=tmp_path/'inner_failures';selected=fit_inner(data,split,cfg,root)
    assert selected['v3_rank']['candidate']==1
    assert selected['v3_rank_refit']['candidate']>=2
    summary=json.loads((root/'v3_rank_tuning.json').read_text())['summary']
    assert not summary[0]['valid'] and summary[1]['valid']
    saved=json.loads((root/'inner/00/selectors/rank/000.json').read_text())
    assert saved['candidate']==0 and saved['inner']==0 and not saved['valid']
    def fail(*args,**kwargs):raise AssertionError('Cached inner fit was rerun')
    monkeypatch.setattr(RankingV3,'solve',fail);monkeypatch.setattr(LogisticProblem,'solve',fail)
    monkeypatch.setattr(RidgeRanker,'fit',fail)
    assert fit_inner(data,split,cfg,root)==selected


def test_resume_interrupted_inner_tuning(reference,data,cfg,tmp_path,monkeypatch):
    import run_rank_v3 as runner
    split=json.loads((reference/'manifest.json').read_text())['splits'][0]
    root=tmp_path/'interrupted';original=runner.inner_record;calls=0
    def interrupt(path,identity,fit_function):
        nonlocal calls
        calls+=1
        if calls==6:raise KeyboardInterrupt('simulated runtime disconnect')
        return original(path,identity,fit_function)
    monkeypatch.setattr(runner,'inner_record',interrupt)
    with pytest.raises(KeyboardInterrupt):fit_inner(data,split,cfg,root)
    committed={p:p.read_bytes() for p in root.rglob('*.json')}
    assert len(committed)==5
    def verified_resume(path,identity,fit_function):
        if path in committed:
            def fail():raise AssertionError('Committed fit was rerun')
            return original(path,identity,fail)
        return original(path,identity,fit_function)
    monkeypatch.setattr(runner,'inner_record',verified_resume)
    selected=fit_inner(data,split,cfg,root)
    assert set(selected)==set(METHODS)
    assert all(p.read_bytes()==value for p,value in committed.items())
