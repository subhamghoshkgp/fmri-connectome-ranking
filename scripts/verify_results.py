"""Check aggregate experiment tables without data downloads or model fitting."""
from pathlib import Path
from collections import Counter
import csv, statistics

ROOT=Path(__file__).resolve().parents[1]
tables={}
for name in ('whole_brain_83','whole_brain_129','cortical_68','cortical_114'):
    with (ROOT/'results'/name/'analysis/fold_metrics.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    counts=Counter(r['method'] for r in rows)
    assert len(rows)==1200 and len(counts)==12 and set(counts.values())=={100},name
    assert len({(r['method'],r['repeat'],r['fold']) for r in rows})==1200,name
    tables[name]=rows
mapping={'LASSO':'max_auc_lasso','Elastic net':'max_auc_enet','RFE-SVM':'max_auc_rfe_svm',
         'IGL-ref':'max_auc_v3','IGL':'v3_expanded','IGR':'v3_rank','IGR-R':'v3_rank_refit','DRR (dense)':'ridge_rank'}
with (ROOT/'results/primary_results_overview.csv').open(newline='') as f:
    summary=list(csv.DictReader(f))
assert len(summary)==32
for row in summary:
    scope=f"{row['scope']}_{row['regions']}"
    selected=[r for r in tables[scope] if r['method'].removeprefix('saved_')==mapping[row['method']]]
    assert len(selected)==100
    for source,target in [('auc','auc_mean'),('support','mean_edges'),('balanced_accuracy','balanced_accuracy'),('f1','f1')]:
        assert abs(statistics.mean(float(r[source]) for r in selected)-float(row[target]))<1e-11,(scope,row['method'],target)
    assert abs(statistics.stdev(float(r['auc']) for r in selected)-float(row['auc_fold_sd']))<1e-11
print('Verified: 4 representations, 4,800 fold rows, 32 primary summaries.')
