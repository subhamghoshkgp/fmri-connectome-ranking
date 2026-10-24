import numpy as np
from scipy.optimize import minimize,LinearConstraint,Bounds,linprog
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
import pytest
from szv3.math import edge_map,lambda_max,psi,dual_norm
from szv3.solver import LogisticProblem,ConvergenceError
from szv3.data import synthetic_dataset


def test_geometry():
    u,v,H=edge_map(7); p=len(u)
    assert H.shape==(7,21) and np.all(H.sum(0)==2)
    b=np.zeros(p); b[u==0]=np.linspace(.2,2,6)
    assert abs(psi(b,H)[0]-2)<1e-9
    b*=0
    for pair,value in [((0,1),2),((2,3),3)]: b[(u==pair[0])&(v==pair[1])]=value
    assert abs(psi(b,H)[0]-5)<1e-9
    ut,vt,Ht=edge_map(3)
    assert abs(psi(np.ones(3),Ht)[0]-1.5)<1e-9
    s=np.random.default_rng(2).normal(size=p)
    # Independent LP maximization over {|beta|<=H'z, 1'z<=1,z>=0}.
    A=np.block([[np.eye(p),-H.T.toarray()],[-np.eye(p),-H.T.toarray()],
                [np.zeros((1,p)),np.ones((1,7))]])
    r=linprog(np.r_[-s,np.zeros(7)],A_ub=A,b_ub=np.r_[np.zeros(2*p),1],
              bounds=[(None,None)]*p+[(0,None)]*7,method='highs')
    assert r.success and abs(-r.fun-dual_norm(s,H))<1e-8


@pytest.mark.parametrize('alpha',[.2,.5,.8])
def test_zero_boundary(alpha):
    d=synthetic_dataset(n=70,n_rois=6)
    X=StandardScaler().fit_transform(d.X); lm=lambda_max(X,d.y,d.H,alpha)
    prob=LogisticProblem(X,d.y,d.H)
    zero=prob.solve(alpha*lm*1.001,(1-alpha)*lm*1.001)
    active=prob.solve(alpha*lm*.98,(1-alpha)*lm*.98)
    assert np.max(np.abs(zero['beta']))<1e-5
    assert np.max(np.abs(active['beta']))>1e-4
    assert lambda_max(X,d.y,d.H,1)==np.max(np.abs(X.T@(d.y.mean()-d.y)/len(d.y)))


@pytest.mark.parametrize('alpha,ridge',[(.2,0),(.5,.001),(.8,.01)])
def test_independent_slsqp(alpha,ridge):
    """Independent smooth epigraph formulation, different optimization algorithm."""
    d=synthetic_dataset(n=75,n_rois=5,seed=23); X=StandardScaler().fit_transform(d.X)
    H=d.H.toarray(); n,p=X.shape; V=H.shape[0]
    lam=lambda_max(X,d.y,d.H,alpha)*.2; le,lv=alpha*lam,(1-alpha)*lam
    reference=LogisticProblem(X,d.y,d.H).solve(le,lv,ridge)
    # Parameters [b,beta,t,z], linear inequalities t>=+-beta and H'z>=+-beta.
    I=np.eye(p); O=np.zeros((p,1)); Z=np.zeros((p,V))
    A=np.block([[O,-I,I,Z],[O,I,I,Z],
                [O,-I,np.zeros((p,p)),H.T],[O,I,np.zeros((p,p)),H.T]])
    def fun(w):
        b,be,t,z=w[0],w[1:1+p],w[1+p:1+2*p],w[1+2*p:]
        eta=X@be+b; res=(expit(eta)-d.y)/n
        val=np.mean(np.logaddexp(0,eta)-d.y*eta)+le*t.sum()+lv*z.sum()+ridge*(be@be)/2
        grad=np.r_[res.sum(),X.T@res+ridge*be,np.full(p,le),np.full(V,lv)]
        return val,grad
    w=np.r_[0.,np.zeros(p),np.ones(p),np.ones(V)]
    bounds=Bounds(np.r_[np.full(1+2*p,-np.inf),np.zeros(V)],np.full(1+2*p+V,np.inf))
    r=minimize(fun,w,jac=True,method='SLSQP',bounds=bounds,
               constraints=[LinearConstraint(A,0,np.inf)],options={'ftol':1e-11,'maxiter':1500})
    assert r.success,r.message
    assert abs(r.fun-reference['objective'])<1e-7
    assert np.max(np.abs(r.x[1:1+p]-reference['beta']))<1e-4


def test_failed_solver_rejected():
    d=synthetic_dataset(n=50,n_rois=8)
    with pytest.raises(ConvergenceError):
        LogisticProblem(d.X,d.y,d.H).solve(.01,.01,kkt_tol=0,max_iter=1)


def test_lambda_max_has_exact_constant_scores_and_certificate():
    from sklearn.metrics import roc_auc_score
    from szv3.math import lambda_max
    from szv3.solver import intercept_only_solution
    d=synthetic_dataset(n=54,n_rois=8)
    for alpha,gated in [(0.2,True),(0.8,True),(1.0,False)]:
        lm=lambda_max(d.X,d.y,d.H,alpha) if gated else lambda_max(d.X,d.y,d.H,1.0)
        fit=intercept_only_solution(d.X,d.y,d.H,alpha*lm,
                                    (1-alpha)*lm if gated else 0.,0.,gated)
        scores=d.X@fit['beta']+fit['intercept']
        assert np.ptp(scores)==0
        assert roc_auc_score(d.y,scores)==0.5
        assert fit['diagnostics']['kkt_max']<1e-8
        assert fit['objective'] <= LogisticProblem(d.X,d.y,d.H,gated).solve(
            alpha*lm,(1-alpha)*lm if gated else 0.,0.)['objective']+1e-7
