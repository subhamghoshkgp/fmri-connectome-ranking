"""Synthetic dimension check only; no Dataset-1 performance is estimated."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'original'))
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from szv3.math import edge_map
from szv3.storage import atomic_json,environment
from ranking import RankingV3,RidgeRanker,ranking_lambda_max

rng=np.random.default_rng(311)
y=np.tile([0,1],21);u,v,H=edge_map(129)
X=rng.normal(size=(42,len(u)));X[:,0]+=y;X=StandardScaler().fit_transform(X)
alpha=.5;lm=ranking_lambda_max(X,y,H,alpha);lam=.02*lm
start=time.perf_counter();selector=RankingV3(X,y,H).solve(alpha*lam,(1-alpha)*lam,.1)
elapsed=time.perf_counter()-start
support=np.flatnonzero(abs(selector['beta'])>1e-4)
refit=RidgeRanker(X[:,support],y).fit(.1)
assert selector['converged'] and refit['converged']
atomic_json(Path(__file__).resolve().parent/'validation/synthetic_dimension_check.json',dict(
    note='Synthetic execution check only. These timings do not estimate full real-data runtime or AUC.',
    subjects=len(y),rois=129,features=X.shape[1],pairs=selector['pair_count'],support=len(support),
    selector_wall_seconds=elapsed,selector_solver=selector['solver'],selector_diagnostics=selector['diagnostics'],
    refit_seconds=refit['runtime_seconds'],factor_seconds=refit['factor_seconds'],refit_diagnostics=refit['diagnostics'],
    environment=environment()))
print('Synthetic 129-ROI dimension check passed:',elapsed,'selector seconds; support',len(support))
