"""Conic solution of Eq. (5), with independently recomputed KKT residuals.

Loss is the MEAN logistic negative log likelihood. Intercept is unpenalized.
No fit is accepted solely because it reached an iteration budget.
"""
import time
import warnings
import cvxpy as cp
import numpy as np
from scipy.special import expit


class ConvergenceError(RuntimeError):
    pass


def intercept_only_solution(X, y, H, le, lv=0., l2=0., gated=True, tolerance=1e-8):
    """Exact optimum at lambda_max, certified by the zero-activation condition."""
    start = time.perf_counter()
    if le <= 0 or l2 < 0 or (gated and lv <= 0):
        raise ValueError('Invalid intercept-only penalties')
    q = float(np.mean(y))
    if not 0 < q < 1:
        raise ValueError('Both classes must be represented')
    gradient = np.asarray(X.T @ (q-y) / len(y))
    if gated:
        excess = float(np.max(H @ np.maximum(np.abs(gradient)-le, 0))-lv)
    else:
        excess = float(np.max(np.abs(gradient))-le)
    if excess > tolerance:
        raise ConvergenceError(f'Zero solution lacks dual certificate: {excess}')
    b = float(np.log(q/(1-q)))
    kkt = max(0., excess)
    return dict(beta=np.zeros(X.shape[1]), intercept=b, z=np.zeros(H.shape[0]),
                objective=float(-q*np.log(q)-(1-q)*np.log1p(-q)),
                status='optimal', converged=True, n_iter=0,
                diagnostics=dict(primal_violation=0., dual_violation=0.,
                                 stationarity=kkt, complementarity=0., kkt_max=kkt,
                                 analytic_activation_margin=excess),
                attempts=[dict(solver='analytic',status='optimal',
                               activation_margin=excess)],
                runtime_seconds=time.perf_counter()-start,
                solver='analytic',lambda_E=float(le),lambda_V=float(lv),
                lambda_2=float(l2))


class LogisticProblem:
    def __init__(self, X, y, H, gated=True):
        self.X, self.y, self.H, self.gated = X, y, H, gated
        n, p = X.shape
        self.beta, self.b, self.t = cp.Variable(p), cp.Variable(), cp.Variable(p)
        self.le, self.lv, self.l2 = [cp.Parameter(nonneg=True) for _ in range(3)]
        eta = X@self.beta + self.b
        loss = cp.sum(cp.logistic(eta) - cp.multiply(y, eta))/n
        self.constraints = [self.beta <= self.t, -self.beta <= self.t]
        penalty = self.le*cp.sum(self.t) + self.l2*cp.sum_squares(self.beta)/2
        if gated:
            self.z = cp.Variable(H.shape[0])
            self.constraints += [self.beta <= H.T@self.z,
                                 -self.beta <= H.T@self.z, self.z >= 0]
            penalty += self.lv*cp.sum(self.z)
        else:
            self.z = None
        self.problem = cp.Problem(cp.Minimize(loss + penalty), self.constraints)

    def solve(self, le, lv=0., l2=0., kkt_tol=2e-5, max_iter=300):
        if le <= 0 or l2 < 0 or (self.gated and lv <= 0):
            raise ValueError('Require le>0, l2>=0 and lv>0 for gated model')
        self.le.value, self.lv.value, self.l2.value = le, lv, l2
        start = time.perf_counter()
        attempts = []
        settings = [('SCS', dict(eps=2e-7, max_iters=30000)),
                    ('CLARABEL', dict(max_iter=max_iter, tol_gap_abs=1e-9,
                                      tol_gap_rel=1e-9, tol_feas=1e-9)),
                    ('CLARABEL', dict(max_iter=max_iter*3, tol_gap_abs=1e-11,
                                      tol_gap_rel=1e-11, tol_feas=1e-11))]
        for solver, options in settings:
            with warnings.catch_warnings(record=True) as ws:
                try:
                    self.problem.solve(solver=solver, warm_start=True, **options)
                    diag = self._diagnostics(le, lv, l2)
                    attempts.append(dict(solver=solver, status=self.problem.status, **diag,
                                         warnings=[str(w.message) for w in ws]))
                except cp.error.SolverError as e:
                    attempts.append(dict(solver=solver, error=str(e)))
                    continue
            if self.problem.status == cp.OPTIMAL and diag['kkt_max'] <= kkt_tol:
                beta = np.asarray(self.beta.value).copy()
                z = np.asarray(self.z.value).copy() if self.gated else np.zeros(self.H.shape[0])
                return dict(beta=beta, intercept=float(self.b.value), z=z,
                            objective=float(self.problem.value), status=self.problem.status,
                            converged=True, diagnostics=diag, attempts=attempts,
                            n_iter=int(self.problem.solver_stats.num_iters),
                            runtime_seconds=time.perf_counter()-start,
                            solver=solver,
                            lambda_E=float(le), lambda_V=float(lv), lambda_2=float(l2))
        raise ConvergenceError(f'Conic solve rejected: {attempts}')

    def _diagnostics(self, le, lv, l2):
        if self.beta.value is None:
            return dict(kkt_max=float('inf'))
        b, beta, t = float(self.b.value), self.beta.value, self.t.value
        eta = self.X@beta + b
        residual = (expit(eta)-self.y)/len(self.y)
        gradient = self.X.T@residual + l2*beta
        dual = [np.asarray(c.dual_value) for c in self.constraints]
        ap, am = dual[:2]
        stationarity = [gradient+ap-am, le-ap-am, np.array([residual.sum()])]
        slack = [t-beta, t+beta]
        if self.gated:
            qp, qm, r = dual[2:]
            z = self.z.value
            stationarity[0] += qp-qm
            stationarity += [lv-self.H@(qp+qm)-r]
            slack += [self.H.T@z-beta, self.H.T@z+beta, z]
        primal = max(0., max(float(np.max(-s)) for s in slack))
        dual_feas = max(0., max(float(np.max(-d)) for d in dual))
        stat = max(float(np.max(np.abs(a))) for a in stationarity)
        comp = max(float(np.max(np.abs(s*d))) for s,d in zip(slack, dual))
        result = dict(primal_violation=primal, dual_violation=dual_feas,
                      stationarity=stat, complementarity=comp)
        result['kkt_max'] = max(result.values())
        return result
