"""Paired descriptive comparison of 83 and 129 regions on the same CV folds."""
import csv
from pathlib import Path
import statistics

METRICS = ('auc','accuracy','balanced_accuracy','f1','sensitivity','specificity','support')


def indexed(path):
    rows = {}
    for row in csv.DictReader(Path(path).open()):
        method = row['method'].replace('saved_max_auc_', 'max_auc_').replace('saved_one_se_', 'one_se_')
        key = (method, int(row['repeat']), int(row['fold']))
        if key in rows:
            raise ValueError('Duplicate method/fold')
        rows[key] = {k: float(row[k]) for k in METRICS}
    return rows


def write(path, rows):
    if not rows:
        raise ValueError('No common fold results')
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def compare(reference129, results83, out):
    a, b = indexed(reference129), indexed(results83)
    if set(a) != set(b) or len(b) != 1200:
        raise ValueError('Require all 12 variants and the same 100 folds at both resolutions')
    paired, summary = [], []
    for method in sorted({key[0] for key in a}):
        keys = sorted(key for key in a if key[0] == method)
        deltas = []
        row = dict(method=method, paired_folds=len(keys))
        for key in keys:
            p = dict(method=method, repeat=key[1], fold=key[2])
            for metric in METRICS:
                p[metric+'_129'] = a[key][metric]
                p[metric+'_83'] = b[key][metric]
                p[metric+'_83_minus_129'] = b[key][metric]-a[key][metric]
            paired.append(p);deltas.append(p['auc_83_minus_129'])
        for metric in METRICS:
            row[metric+'_129_mean'] = statistics.mean(a[k][metric] for k in keys)
            row[metric+'_83_mean'] = statistics.mean(b[k][metric] for k in keys)
            row[metric+'_83_minus_129_mean'] = row[metric+'_83_mean']-row[metric+'_129_mean']
        row.update(edge_fraction_129=row['support_129_mean']/8256,
                   edge_fraction_83=row['support_83_mean']/3403,
                   auc_delta_fold_sd=statistics.stdev(deltas),
                   auc_83_wins=sum(d>0 for d in deltas),auc_ties=sum(d==0 for d in deltas),
                   auc_83_losses=sum(d<0 for d in deltas),
                   interpretation='Same subjects and saved CV splits; repeated folds are dependent; descriptive comparison')
        summary.append(row)
    out = Path(out);out.mkdir(parents=True,exist_ok=True)
    write(out/'resolution_83_vs_129_paired_folds.csv',paired)
    write(out/'resolution_83_vs_129_summary.csv',summary)
    return summary
