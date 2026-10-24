import json
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'original')]

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold
from szv3.data import Dataset
from szv3.math import edge_map
from szv3.metrics import prediction_metrics
from szv3.storage import atomic_json,atomic_npz,environment
from run_rank_v3 import RULE,REFERENCE_METHODS,fold_dir


@pytest.fixture
def data():
    rng=np.random.default_rng(731)
    y=np.tile([0,1],20);u,v,H=edge_map(5)
    X=rng.normal(size=(len(y),len(u)))
    X[:,0]+=1.2*y;X[:,3]+=.7*y
    return Dataset(X,y,np.array([f'subject_{i}' for i in range(len(y))]),
        np.array([f'roi_{i}' for i in range(5)]),u,v,H,dict(synthetic=True,representation='fisher'))


@pytest.fixture
def cfg():
    c=json.loads((ROOT/'configs/ranking_legacy.json').read_text())
    c.update(alpha=[.5],ratio=[.4,.1],ridge=[.01],rho=[.1,1.])
    return c


@pytest.fixture
def reference(tmp_path,data):
    """Saved predictions are fixture records, never presented as study results."""
    root=tmp_path/'saved_reference';splits=[]
    for fold,(tr,te) in enumerate(StratifiedKFold(2,shuffle=True,random_state=117).split(data.X,data.y)):
        inner=[dict(train=tr[a].tolist(),valid=tr[b].tolist()) for a,b in
            StratifiedKFold(2,shuffle=True,random_state=119).split(data.X[tr],data.y[tr])]
        split=dict(repeat=0,fold=fold,train=tr.tolist(),test=te.tolist(),inner=inner);splits.append(split)
        for method in REFERENCE_METHODS:
            score=data.X[te,0];pred=(score>=.5).astype(int)
            record=dict(method=method,repeat=0,fold=fold,train=tr.tolist(),test=te.tolist(),
                converged=True,selection_rule=RULE,metrics=prediction_metrics(data.y[te],score,pred),
                params=dict(fixture=True),chosen_candidate=0,diagnostics=dict(fixture=True),runtime_seconds=0.,
                selected_edge_count=1)
            old=dict(record);old['selection_rule']='one_se'
            record['old_result']=old
            arrays=dict(train_indices=tr,test_indices=te,y_true=data.y[te],score=score,predicted=pred,
                selected_edges=np.array([0]),beta=np.eye(1,data.X.shape[1],0).ravel())
            arrays.update({'old_'+k:v for k,v in list(arrays.items())})
            path=root/fold_dir(split)/method
            atomic_json(path.with_suffix('.json'),record);atomic_npz(path.with_suffix('.npz'),**arrays)
    atomic_json(root/'manifest.json',dict(data=data.provenance,environment=environment(),splits=splits,
        config=dict(repeats=1,outer_folds=2,inner_folds=2),selection_rule=RULE,signature='synthetic-fixture',
        preprocessed_X_sha256=hashlib.sha256(np.ascontiguousarray(data.X).tobytes()).hexdigest()))
    atomic_npz(root/'dataset_metadata.npz',y=data.y,u=data.u,v=data.v,roi_names=data.roi_names,subject_ids=data.subject_ids)
    return root
