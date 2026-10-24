import json
from pathlib import Path
import numpy as np
from szv3.data import synthetic_dataset
from szv3.experiment import run,make_splits,choose_candidate


def test_nested_folds_resume_and_scaling(tmp_path):
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/smoke.json').read_text())
    cfg['grids']={m:rows[:1] for m,rows in cfg['grids'].items()}
    d=synthetic_dataset()
    for split in make_splits(d.y,cfg):
        tr,te=set(split['train']),set(split['test'])
        assert not tr&te
        for inner in split['inner']:
            a,b=set(inner['train']),set(inner['valid'])
            assert not a&b and a|b==tr and not (a|b)&te
    first=run(d,cfg,tmp_path,jobs=1,max_outer=1)
    assert first['complete'] is False
    checkpoints=list(tmp_path.glob('repeat_*/fold_*/*.npz'))
    before={str(p):p.stat().st_mtime_ns for p in checkpoints}
    second=run(d,cfg,tmp_path,jobs=1)
    assert second['complete'] is True
    assert all(Path(p).stat().st_mtime_ns==v for p,v in before.items())
    for p in tmp_path.glob('repeat_*/fold_*/*.npz'):
        a=np.load(p,allow_pickle=False)
        # Extreme test values would not enter either mean or scale.
        np.testing.assert_allclose(a['scaler_mean'],d.X[a['train_indices']].mean(0))
        manual=d.X[a['test_indices']]@a['raw_beta']+a['raw_intercept']
        np.testing.assert_allclose(manual,a['score'],atol=1e-9)
    old={str(p):p.stat().st_mtime_ns for p in tmp_path.glob('repeat_*/fold_*/*.npz')}
    run(d,cfg,tmp_path,jobs=1)
    assert all(Path(p).stat().st_mtime_ns==v for p,v in old.items())


def test_selection_excludes_failed_candidate():
    rows=[dict(candidate=0,valid=False,mean_auc=.99,mean_support=1,se_auc=0),
          dict(candidate=1,valid=True,mean_auc=.8,mean_support=10,se_auc=.05),
          dict(candidate=2,valid=True,mean_auc=.78,mean_support=5,se_auc=.05)]
    assert choose_candidate(rows,'one_se')['candidate']==2
    assert choose_candidate(rows,'max_auc')['candidate']==1
