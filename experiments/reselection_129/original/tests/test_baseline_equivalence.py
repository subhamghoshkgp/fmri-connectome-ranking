import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from szv3.data import synthetic_dataset
from szv3.solver import LogisticProblem


def test_logistic_baselines_match_independent_saga():
    d=synthetic_dataset(n=80,n_rois=5,seed=9); X=StandardScaler().fit_transform(d.X)
    for le,l2 in [(.035,0.),(.035,.02)]:
        conic=LogisticProblem(X,d.y,d.H,gated=False).solve(le,0.,l2)
        # sklearn objective uses C relative to summed loss; ours uses mean loss.
        model=LogisticRegression(solver='saga',penalty='elasticnet',
                l1_ratio=le/(le+l2),C=1/(len(d.y)*(le+l2)),
                tol=1e-10,max_iter=50000,random_state=0).fit(X,d.y)
        assert model.n_iter_[0]<50000
        np.testing.assert_allclose(conic['beta'],model.coef_.ravel(),atol=2e-5)
        assert abs(conic['intercept']-model.intercept_[0])<2e-5
