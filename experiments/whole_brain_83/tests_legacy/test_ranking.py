import numpy as np
import cvxpy as cp
import pytest
from scipy.optimize import check_grad
from sklearn.preprocessing import StandardScaler
from ranking import pairs,loss_score_gradient,ranking_lambda_max,RankingV3,RidgeRanker,ranking_intercept
from szv3.solver import ConvergenceError


def test_pair_gradient_and_translation(data):
    pos,neg=pairs(data.y);s=data.X[:,0]
    assert len(pos)==400
    f=lambda t:loss_score_gradient(t,pos,neg)[0]
    g=lambda t:loss_score_gradient(t,pos,neg)[1]
    assert check_grad(f,g,s)<2e-6
    assert abs(g(s).sum())<1e-14
    assert abs(f(s+23)-f(s))<1e-13


@pytest.mark.parametrize('y',[[0,0],[1,1],[0,2]])
def test_invalid_pairs(y):
    with pytest.raises(ValueError):pairs(y)


def test_lambda_boundary_and_zero_solution(data):
    X=StandardScaler().fit_transform(data.X);alpha=.5
    lm=ranking_lambda_max(X,data.y,data.H,alpha)
    g=-.5*(X[data.y==1].mean(0)-X[data.y==0].mean(0))
    certificate=lambda l:np.max(data.H@np.maximum(abs(g)-alpha*l,0))-(1-alpha)*l
    assert abs(certificate(lm))<1e-12
    assert certificate(lm*.999)>0
    result=RankingV3(X,data.y,data.H).solve(alpha*lm*1.01,(1-alpha)*lm*1.01,.01)
    assert np.max(abs(result['beta']))<1e-5
    assert result['diagnostics']['kkt_max']<=2e-5


def test_selector_matches_independent_dense_pair_problem(data):
    X=StandardScaler().fit_transform(data.X);y=data.y
    pos,neg=pairs(y);D=X[pos]-X[neg];p=X.shape[1]
    le,lv,l2=.015,.02,.1
    actual=RankingV3(X,y,data.H).solve(le,lv,l2)
    beta=cp.Variable(p);z=cp.Variable(data.H.shape[0],nonneg=True)
    objective=cp.sum(cp.logistic(-D@beta))/len(pos)+le*cp.norm1(beta)+lv*cp.sum(z)+l2*cp.sum_squares(beta)/2
    independent=cp.Problem(cp.Minimize(objective),[beta<=data.H.T@z,-beta<=data.H.T@z])
    independent.solve(solver='CLARABEL',max_iter=400,tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
    assert abs(actual['objective']-independent.value)<2e-6
    np.testing.assert_allclose(actual['beta'],beta.value,atol=5e-5)


@pytest.mark.parametrize('wide',[False,True])
def test_ridge_row_space_matches_independent_full_feature_fit(data,wide):
    X=data.X.copy();y=data.y
    if wide:X=np.c_[X,np.tile(X,(1,5)),np.ones(len(y))] # p>n and rank deficient
    X=StandardScaler().fit_transform(X);pos,neg=pairs(y);rho=.1
    result=RidgeRanker(X,y).fit(rho)
    beta=cp.Variable(X.shape[1])
    objective=cp.sum(cp.logistic(-(X[pos]-X[neg])@beta))/len(pos)+rho*cp.sum_squares(beta)/2
    p=cp.Problem(cp.Minimize(objective));p.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
    assert abs(result['objective']-p.value)<2e-7
    np.testing.assert_allclose(result['beta'],beta.value,atol=4e-5)
    assert result['diagnostics']['gradient_max']<=2e-5


def test_empty_support(data):
    r=RidgeRanker(data.X[:,:0],data.y).fit(.1)
    assert r['converged'] and len(r['beta'])==0
    assert r['objective']==np.log(2)
    assert ranking_intercept(data.X[:,:0],data.y,r['beta'])==0


def test_optimizer_success_is_not_enough(data,monkeypatch):
    import ranking
    from types import SimpleNamespace
    monkeypatch.setattr(ranking,'minimize',lambda f,t,**kw:SimpleNamespace(x=t,success=True,message='fake',nit=0))
    with pytest.raises(ConvergenceError):RidgeRanker(data.X,data.y).fit(.1)
