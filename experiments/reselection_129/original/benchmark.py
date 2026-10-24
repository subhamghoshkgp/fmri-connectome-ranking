"""Benchmark one certified fit; this is not cross-validated evaluation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import argparse
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from szv3.data import synthetic_dataset,load_dataset
from szv3.math import lambda_max
from szv3.solver import LogisticProblem
from szv3.storage import atomic_json
p=argparse.ArgumentParser(); p.add_argument('--out',required=True)
p.add_argument('--data');p.add_argument('--atlas');p.add_argument('--labels')
p.add_argument('--representation',choices=['raw','fisher']);p.add_argument('--synthetic',action='store_true')
a=p.parse_args()
if a.synthetic:
    d=synthetic_dataset(n=54,n_rois=129)
else:
    if not all([a.data,a.atlas,a.labels,a.representation]):p.error('Missing real-data arguments')
    d=load_dataset(a.data,a.atlas,a.labels,a.representation)
X=StandardScaler().fit_transform(d.X)
lm=lambda_max(X,d.y,d.H,.5); start=time.perf_counter()
s=LogisticProblem(X,d.y,d.H).solve(.1*lm,.1*lm,.001)
s.pop('beta');s.pop('z');s['wall_seconds']=time.perf_counter()-start
s['kind']=d.provenance['kind'];s['shape']=list(X.shape)
s['purpose']='Computational feasibility and KKT check only; no held-out performance claim'
atomic_json(a.out,s);print(s)
