"""RFE-SVM reconstruction, not a claim of exact historical reproduction."""
import time
import warnings
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def svm_pipeline(C, k, step=.2):
    estimator = SVC(kernel='linear', C=C, tol=1e-6, max_iter=200000)
    return Pipeline([('scale', StandardScaler()),
                     ('rfe', RFE(estimator, n_features_to_select=k, step=step))])


def fit_svm(X, y, Xtest, params, seed, calibrate=False):
    start=time.perf_counter()
    model=svm_pipeline(**params)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        if calibrate:
            cal=CalibratedClassifierCV(model,method='sigmoid',ensemble=False,
                   cv=StratifiedKFold(3,shuffle=True,random_state=seed),n_jobs=1)
            cal.fit(X,y)
            model=cal.calibrated_classifiers_[0].estimator
            probabilities=cal.predict_proba(Xtest)[:,1]
        else:
            model.fit(X,y); probabilities=None
    rfe=model.named_steps['rfe']; svm=rfe.estimator_
    if svm.fit_status_!=0: raise RuntimeError('SVM did not converge')
    beta=np.zeros(X.shape[1]); beta[rfe.support_] = svm.coef_.ravel()
    scaler=model.named_steps['scale']
    Xt=scaler.transform(X)[:,rfe.support_]
    signs=2*y-1
    signed_dual=np.zeros(len(y)); signed_dual[svm.support_]=svm.dual_coef_.ravel()
    alpha=np.abs(signed_dual); margin=signs*svm.decision_function(Xt)
    slack=np.maximum(1-margin,0); C=params['C']
    primal=.5*float(svm.coef_.ravel()@svm.coef_.ravel())+C*slack.sum()
    dual=alpha.sum()-.5*float(svm.coef_.ravel()@svm.coef_.ravel())
    gap=float(primal-dual)
    diagnostics=dict(primal_objective=float(primal),dual_objective=float(dual),
        relative_duality_gap=gap/(1+abs(primal)),
        stationarity=float(np.max(np.abs(svm.coef_.ravel()-signed_dual@Xt))),
        intercept_stationarity=float(abs(signed_dual.sum())),
        dual_violation=float(max(0,np.max(alpha-C))),
        complementarity=float(max(np.max(np.abs(alpha*(margin-1+slack))),
                                  np.max(np.abs((C-alpha)*slack)))))
    if diagnostics['relative_duality_gap']>1e-4 or gap < -1e-5:
        raise RuntimeError(f'SVM primal/dual check failed: {diagnostics}')
    return dict(beta=beta,intercept=float(svm.intercept_[0]),z=np.array([]),
                support=np.flatnonzero(rfe.support_),score=model.decision_function(Xtest),
                predicted=model.predict(Xtest),probability=probabilities,
                scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,
                status='libsvm_converged',converged=True,
                diagnostics=diagnostics,objective=float(primal),
                n_iter=int(np.sum(svm.n_iter_)),runtime_seconds=time.perf_counter()-start,
                calibration='training-only 3-fold entire scaler/RFE/SVM pipeline' if calibrate else None,
                support_definition='RFE selected subset; may include tiny SVM weights')
