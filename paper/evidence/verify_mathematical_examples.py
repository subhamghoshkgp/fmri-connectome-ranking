from pathlib import Path
import json,numpy as np
from scipy.optimize import linprog,minimize
from scipy.special import expit
result={}
d=12;u,v=np.triu_indices(d,1);H=np.zeros((d,len(u)));H[u,np.arange(len(u))]=1;H[v,np.arange(len(u))]=1
checks=[]
for k in range(2,7):
 for kind in ('star','disjoint'):
  beta=np.zeros(len(u));pairs=[(0,j) for j in range(1,k+1)] if kind=='star' else [(2*j,2*j+1) for j in range(k)]
  for j,pair in enumerate(pairs):beta[np.flatnonzero((u==pair[0])&(v==pair[1]))[0]]=.7*(-1)**j
  opt=linprog(np.ones(d),A_ub=-H.T,b_ub=-np.abs(beta),bounds=(0,None),method='highs')
  expected=.7 if kind=='star' else k*.7
  assert opt.success and abs(opt.fun-expected)<1e-12
  checks.append(dict(k=k,support=kind,value=float(opt.fun),expected=expected))
result['star_disjoint_lp_checks']=checks
rng=np.random.default_rng(921);X=rng.normal(size=(16,45));X-=X.mean(axis=0);y=np.tile([0,1],8)
pos=np.repeat(np.flatnonzero(y),8);neg=np.tile(np.flatnonzero(y==0),8);rho=.4
U,s,Vt=np.linalg.svd(X,full_matrices=False);A=U*s

def f(beta,M):
 score=M@beta;diff=score[pos]-score[neg];w=expit(-diff)/len(pos)
 gradscore=np.bincount(neg,weights=w,minlength=len(y))-np.bincount(pos,weights=w,minlength=len(y))
 return float(np.logaddexp(0,-diff).mean()+rho*(beta@beta)/2),M.T@gradscore+rho*beta
full=minimize(f,np.zeros(X.shape[1]),args=(X,),jac=True,method='L-BFGS-B',options=dict(gtol=1e-11,ftol=1e-15,maxiter=2000))
small=minimize(f,np.zeros(A.shape[1]),args=(A,),jac=True,method='L-BFGS-B',options=dict(gtol=1e-11,ftol=1e-15,maxiter=2000))
assert abs(full.fun-small.fun)<1e-12
assert np.max(np.abs(full.x-Vt.T@small.x))<1e-7
result['ridge_reduction_check']=dict(n=16,p=45,rho=rho,untruncated_coordinates=len(s),full_objective=float(full.fun),reduced_objective=float(small.fun),coefficient_max_abs_difference=float(np.max(np.abs(full.x-Vt.T@small.x))))
Path(__file__).with_name('MATHEMATICAL_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print('Proposition examples and rank-deficient ridge reduction independently checked.')
