"""Numerical audit of outer fits using the exact saved dev_v2 choices and splits.

This does not redo inner model selection. Run from the workspace root. Saved
pickle is untrusted input and is read with the restricted audit unpickler.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

here = Path(__file__).resolve().parent
root = here.parent
sys.path.insert(0, str(root / 'old_run_audit'))
from audit_saved_run import read_arrays
from szv3.data import load_dataset
from szv3.solver import LogisticProblem

parser = argparse.ArgumentParser()
parser.add_argument('--method', choices=['v3', 'enet'], required=True)
parser.add_argument('--data', required=True, help='Path to the original Dataset-1 MAT file')
parser.add_argument('--atlas', default=str(here/'reference/ParcellationLausanne2008.xls'))
parser.add_argument('--labels', default=str(here/'reference/labels_index_CORTICAL_Laus2008_all_scales.mat'))
parser.add_argument('--saved', default=str(root/'old_run_audit/records.pkl'))
parser.add_argument('--limit', type=int, default=25)
parser.add_argument('--out', required=True)
args = parser.parse_args()

records = read_arrays(args.saved)
records = sorted((r for r in records if r['method'] == args.method),
                 key=lambda r: r['global_fold'])[:args.limit]
data = load_dataset(args.data, args.atlas, args.labels, 'fisher')
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('a') as f:
    for r in records:
        if r['global_fold'] in [json.loads(line)['global_fold'] for line in out.read_text().splitlines()]:
            continue
        scaler = StandardScaler().fit(data.X[r['train_idx']])
        X = scaler.transform(data.X[r['train_idx']])
        Xt = scaler.transform(data.X[r['test_idx']])
        original_score_error = float(np.max(np.abs(Xt@r['coef']+r['intercept']-r['scores'])))
        if original_score_error > 1e-5:
            raise ValueError(f'Saved score mismatch at {r["global_fold"]}: {original_score_error}')
        info = r['solver_info']
        if args.method == 'v3':
            le, lv, l2 = info['lam_E'], info['lam_V'], info['lam2']
            old_penalty = le*np.abs(r['coef']).sum()+lv*np.sum(r['z'])+l2*np.dot(r['coef'],r['coef'])/2
        else:
            n, C, mix = len(X), info['C'], info['l1_ratio']
            le, lv, l2 = mix/(n*C), 0., (1-mix)/(n*C)
            old_penalty = le*np.abs(r['coef']).sum()+l2*np.dot(r['coef'],r['coef'])/2
        old_eta = X@r['coef']+r['intercept']
        old_obj = float(np.mean(np.logaddexp(0,old_eta)-data.y[r['train_idx']]*old_eta)+old_penalty)
        fit = LogisticProblem(X,data.y[r['train_idx']],data.H,gated=args.method=='v3').solve(le,lv,l2)
        scores = Xt@fit['beta']+fit['intercept']
        entry = dict(method=args.method,global_fold=int(r['global_fold']),repeat=int(r['repeat_id']),
                     fold=int(r['fold_id']),train_idx=r['train_idx'].tolist(),test_idx=r['test_idx'].tolist(),
                     y_test=r['y_test'].tolist(),scores=scores.tolist(),
                     old_scores=r['scores'].tolist(),
                     old_auc=float(r['metrics']['auc']),new_auc=float(roc_auc_score(r['y_test'],scores)),
                     old_balanced_accuracy=float(r['metrics']['balanced_accuracy']),
                     new_balanced_accuracy=float(balanced_accuracy_score(r['y_test'],scores>=0)),
                     old_selected=int(r['n_selected']),new_selected=int(np.sum(np.abs(fit['beta'])>1e-4)),
                     original_score_error=original_score_error,
                     old_objective=old_obj,new_objective=fit['objective'],
                     objective_gap=old_obj-fit['objective'],
                     score_change_max=float(np.max(np.abs(scores-r['scores']))),
                     score_change_mean=float(np.mean(np.abs(scores-r['scores']))),
                     old_converged=bool(info['converged']),
                     certified_converged=fit['converged'],
                     certified_kkt=fit['diagnostics'],
                     certified_iterations=fit['n_iter'],fit_seconds=fit['runtime_seconds'],
                     parameters=dict(lambda_E=le,lambda_V=lv,lambda_2=l2))
        f.write(json.dumps(entry)+'\n');f.flush()
        print(args.method,r['global_fold'],'old/new AUC',entry['old_auc'],entry['new_auc'],
              'objective gap',entry['objective_gap'],'KKT',fit['diagnostics']['kkt_max'],
              'seconds',fit['runtime_seconds'],flush=True)
