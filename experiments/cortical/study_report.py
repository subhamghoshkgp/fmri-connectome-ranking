"""Combine cortical resolutions without pooling incompatible edge supports."""
from pathlib import Path
import json
from run_cortical import analyze,METHODS,LABELS,read_metrics,METRICS
from szv3.storage import run_lock,atomic_json
from run_rank_v3 import csv_file
import numpy as np


def build(outputs,destination,require_complete=False):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    summaries={};rows=[];paired=[]
    for resolution,path in outputs.items():
        path=Path(path)
        with run_lock(path):s=analyze(path)
        expected=s['expected_folds_per_method']
        if require_complete and (not s['complete'] or expected!=100):raise ValueError(f'{resolution}: 100 folds per variant required')
        summaries[str(resolution)]=s
        for m in METHODS:
            r=s['methods'][m]
            if not r['completed_folds']:continue
            st=r['stability'] or {}
            rows.append(dict(original_resolution=int(resolution),cortical_rois=s['data']['n_rois'],
                available_edges=s['available_edges'],method=m,label=LABELS[m],folds=r['completed_folds'],
                **{k:r[k] for k in ('auc_mean','auc_fold_sd','balanced_accuracy_mean','f1_mean','sensitivity_mean','specificity_mean','support_mean','support_fraction_mean')},
                jaccard=st.get('jaccard'),nogueira=st.get('nogueira')))
    complete=len(summaries)==2 and all(s['complete'] and s['expected_folds_per_method']==100 for s in summaries.values())
    csv_file(destination/'all_cortical_method_summary.csv',rows)
    if set(map(int,outputs))=={83,129}:
        paths={int(r):Path(p) for r,p in outputs.items()}
        a,b=(read_metrics(paths[r]/'analysis/fold_metrics.csv') if (paths[r]/'analysis/fold_metrics.csv').exists() else {} for r in (83,129))
        for key in sorted(set(a)&set(b)):
            row=dict(method=key[0],repeat=key[1],fold=key[2])
            for metric in METRICS:
                row[metric+'_cortex68']=a[key][metric];row[metric+'_cortex114']=b[key][metric]
                row[metric+'_cortex114_minus_cortex68']=b[key][metric]-a[key][metric]
            paired.append(row)
        csv_file(destination/'cortical_resolution_paired_folds.csv',paired)
    final=dict(complete=complete,resolutions=summaries,
        interpretation='Same 54 participants, dependent repeated CV folds. Descriptive comparisons; no independent-cohort or significance claim. Feature universes differ between resolutions and full/cortical analyses. No cross-atlas Jaccard is computed.')
    atomic_json(destination/'summary.json',final)
    lines=['# Cortical-only Dataset-1: both resolutions','',f'Full study complete: **{complete}**.','',
        'Each final resolution requires 100 outer evaluations for each of 12 variants; 2,400 outer models overall.','',
        '| Original atlas | Cortex | Method | Folds | AUC mean | Fold SD | BA | F1 | Mean edges | Jaccard | Nogueira |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'—' if x is None else f'{x:.4f}'
    for r in rows:
        lines.append('| '+' | '.join([str(r['original_resolution']),str(r['cortical_rois']),r['label'],str(r['folds'])]+[fmt(r[k]) for k in ('auc_mean','auc_fold_sd','balanced_accuracy_mean','f1_mean','support_mean','jaccard','nogueira')])+' |')
    lines+=['',final['interpretation'],'',
        'The anatomical restriction removes cortical–subcortical and subcortical–subcortical edges, not just columns selected by a classifier. All model tuning is performed afresh on cortical features.','',
        'Compare cortical_vs_fullbrain_summary.csv within each resolution for matched-fold changes. Dense ridge ranking uses every available cortical connection; its feature-selection stability is not informative. Raw support counts/Jaccard depend on the available feature universe.']
    (destination/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    return final
