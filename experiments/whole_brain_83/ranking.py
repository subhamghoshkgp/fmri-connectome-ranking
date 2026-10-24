"""Convex ranking V3 and a ridge ranker. All pairs are TRAINING-subject pairs."""
import time
import warnings
import numpy as np
import cvxpy as cp
from scipy.special import expit
from scipy.optimize import minimize
from szv3.solver import ConvergenceError


def pairs(y):
    y = np.asarray(y)
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Labels must be 0/1')
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError('Both classes are required')
    return np.repeat(positive, len(negative)), np.tile(negative, len(positive))


def loss_score_gradient(scores, positive, negative):
    """Mean pair logistic loss and gradient w.r.t. individual subject scores."""
    differences = scores[positive] - scores[negative]
    weights = expit(-differences) / len(positive)
    residual = (np.bincount(negative, weights=weights, minlength=len(scores)) -
                np.bincount(positive, weights=weights, minlength=len(scores)))
    return float(np.logaddexp(0, -differences).mean()), residual


def ranking_lambda_max(X, y, H, alpha):
    if not 0 < alpha < 1:
        raise ValueError('This gated grid requires 0 < alpha < 1')
    pairs(y)
    gradient = -.5 * (X[y == 1].mean(0) - X[y == 0].mean(0))
    lo, hi = 0., float(np.max(np.abs(gradient)) / alpha)
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.max(H @ np.maximum(np.abs(gradient) - alpha * mid, 0)) <= (1-alpha) * mid:
            hi = mid
        else:
            lo = mid
    return hi


class RankingV3:
    def __init__(self, X, y, H):
        self.X, self.H = np.asarray(X), H
        self.positive, self.negative = pairs(y)
        n, p = X.shape
        self.beta, self.t = cp.Variable(p), cp.Variable(p)
        self.z, self.eta = cp.Variable(H.shape[0]), cp.Variable(n)
        self.le, self.lv, self.l2 = [cp.Parameter(nonneg=True) for _ in range(3)]
        # Shared subject scores avoid constructing the dense (n1*n0)-by-p matrix.
        differences = self.eta[self.negative] - self.eta[self.positive]
        loss = cp.sum(cp.logistic(differences)) / len(self.positive)
        self.constraints = [self.beta <= self.t, -self.beta <= self.t,
            self.beta <= H.T @ self.z, -self.beta <= H.T @ self.z,
            self.z >= 0, self.eta == X @ self.beta]
        penalty = self.le*cp.sum(self.t) + self.lv*cp.sum(self.z) + self.l2*cp.sum_squares(self.beta)/2
        self.problem = cp.Problem(cp.Minimize(loss + penalty), self.constraints)

    def diagnostics(self, le, lv, l2):
        beta, t, z, eta = self.beta.value, self.t.value, self.z.value, self.eta.value
        if any(x is None for x in (beta, t, z, eta)):
            return dict(kkt_max=float('inf'))
        _, residual = loss_score_gradient(eta, self.positive, self.negative)
        ap, am, qp, qm, rz, qeta = [np.asarray(c.dual_value) for c in self.constraints]
        stationarity = [l2*beta+ap-am+qp-qm-self.X.T@qeta,
                        le-ap-am, lv-self.H@(qp+qm)-rz, residual+qeta]
        slack = [t-beta, t+beta, self.H.T@z-beta, self.H.T@z+beta, z]
        nonnegative_duals = [ap, am, qp, qm, rz]
        result = dict(primal_violation=max(0., *(float(np.max(-a)) for a in slack)),
            equality_violation=float(np.max(np.abs(eta-self.X@beta))),
            dual_violation=max(0., *(float(np.max(-a)) for a in nonnegative_duals)),
            stationarity=max(float(np.max(np.abs(a))) for a in stationarity),
            complementarity=max(float(np.max(np.abs(a*b))) for a,b in zip(slack, nonnegative_duals)))
        result['kkt_max'] = max(result.values())
        return result

    def solve(self, le, lv, l2, kkt_tol=2e-5, max_iter=300):
        if le <= 0 or lv <= 0 or l2 < 0:
            raise ValueError('Require positive edge/node penalties and nonnegative ridge')
        self.le.value, self.lv.value, self.l2.value = le, lv, l2
        start = time.perf_counter()
        attempts = []
        settings = [('SCS', dict(eps=2e-7, max_iters=30000)),
            ('CLARABEL', dict(max_iter=max_iter, tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)),
            ('CLARABEL', dict(max_iter=3*max_iter, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11))]
        for solver, options in settings:
            with warnings.catch_warnings(record=True) as caught:
                try:
                    self.problem.solve(solver=solver, warm_start=True, **options)
                    d = self.diagnostics(le, lv, l2)
                    attempts.append(dict(solver=solver, status=self.problem.status, **d,
                                         warnings=[str(w.message) for w in caught]))
                except cp.error.SolverError as exc:
                    attempts.append(dict(solver=solver, error=str(exc)))
                    continue
            if self.problem.status == cp.OPTIMAL and d['kkt_max'] <= kkt_tol:
                beta, z = np.asarray(self.beta.value).copy(), np.asarray(self.z.value).copy()
                if not np.isfinite(beta).all() or not np.isfinite(z).all():
                    continue
                loss, _ = loss_score_gradient(self.X@beta, self.positive, self.negative)
                return dict(beta=beta, z=z, objective=loss+le*np.abs(beta).sum()+lv*z.sum()+l2*(beta@beta)/2,
                    diagnostics=d, converged=True, status='optimal', solver=solver,
                    attempts=attempts, n_iter=int(self.problem.solver_stats.num_iters),
                    runtime_seconds=time.perf_counter()-start,
                    lambda_E=float(le), lambda_V=float(lv), lambda_2=float(l2),
                    pair_count=len(self.positive))
        raise ConvergenceError(f'Ranking V3 rejected: {attempts}')


class RidgeRanker:
    """Exact row-space parameterisation, with a full-feature gradient certificate."""
    def __init__(self, X, y):
        start = time.perf_counter()
        self.X = np.asarray(X)
        self.positive, self.negative = pairs(y)
        if X.shape[1]:
            U, s, self.Vt = np.linalg.svd(X, full_matrices=False)
            self.A = U * s
        else:
            self.Vt, self.A = np.empty((0,0)), np.empty((len(y),0))
        self.factor_seconds = time.perf_counter()-start

    def objective_gradient(self, theta, rho):
        loss, residual = loss_score_gradient(self.A@theta, self.positive, self.negative)
        return loss+rho*(theta@theta)/2, self.A.T@residual+rho*theta

    def fit(self, rho, gradient_tol=2e-5, max_iter=1000):
        if rho <= 0:
            raise ValueError('Ridge ranking requires rho > 0')
        start = time.perf_counter()
        if self.X.shape[1] == 0:
            return dict(beta=np.array([]), objective=float(np.log(2)), diagnostics=dict(gradient_max=0.),
                converged=True, status='empty_support', n_iter=0, solver='analytic', rho=rho,
                attempts=[], runtime_seconds=time.perf_counter()-start,
                factor_seconds=self.factor_seconds, pair_count=len(self.positive))
        theta = np.zeros(self.A.shape[1])
        attempts = []
        for options in [dict(gtol=1e-9, ftol=1e-14, maxiter=max_iter, maxls=50),
                        dict(gtol=1e-11, ftol=1e-16, maxiter=3*max_iter, maxls=100)]:
            result = minimize(self.objective_gradient, theta, args=(rho,), method='L-BFGS-B', jac=True, options=options)
            theta = result.x
            beta = self.Vt.T@theta
            loss, residual = loss_score_gradient(self.X@beta, self.positive, self.negative)
            gradient = self.X.T@residual+rho*beta
            maximum = float(np.max(np.abs(gradient)))
            attempts.append(dict(success=bool(result.success), message=str(result.message),
                                 gradient_max=maximum, n_iter=int(result.nit)))
            if maximum <= gradient_tol and np.isfinite(beta).all():
                return dict(beta=beta, objective=float(loss+rho*(beta@beta)/2),
                    diagnostics=dict(gradient_max=maximum), converged=True,
                    status='gradient_certified', n_iter=int(result.nit), solver='L-BFGS-B', rho=rho,
                    attempts=attempts, runtime_seconds=time.perf_counter()-start,
                    factor_seconds=self.factor_seconds, pair_count=len(self.positive))
        raise ConvergenceError(f'Ridge ranking rejected: {attempts}')


def ranking_intercept(X, y, beta):
    """Training class-midpoint threshold; a constant shift never changes AUC."""
    scores = X@beta
    return -float((scores[y==1].mean()+scores[y==0].mean())/2)
