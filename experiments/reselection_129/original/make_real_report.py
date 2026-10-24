"""Aggregate immutable audit outputs; never recompute folds from test labels."""
from pathlib import Path
import csv
import json
from collections import Counter
import numpy as np

root = Path(__file__).resolve().parent
audit = json.loads((root.parent/'old_run_audit/verified_audit.json').read_text())
v3 = [json.loads(s) for s in (root/'validation/real_v3_refits.jsonl').read_text().splitlines()]
enet = [json.loads(s) for s in (root/'validation/real_enet_refits.jsonl').read_text().splitlines()]
pil3 = json.loads((root/'validation/real_nested_pilot/analysis/summary.json').read_text())
pil5_path = root/'validation/real_v3_5fold_pilot'
pil5 = json.loads((pil5_path/'analysis/summary.json').read_text())
assert len(v3)==len(enet)==25 and pil3['complete'] and pil5['complete']
assert all(r['certified_converged'] and r['certified_kkt']['kkt_max'] < 2e-5 for r in v3+enet)
assert all(r['original_score_error'] < 1e-5 for r in v3+enet)

def mean(rows, field): return float(np.mean([r[field] for r in rows]))

meta = np.load(pil5_path/'dataset_metadata.npz')
u,v,names=meta['u'],meta['v'],meta['roi_names']
counts=Counter()
for q in sorted(pil5_path.glob('repeat_*/fold_*/v3.npz')):
    with np.load(q) as arr: counts.update(arr['selected_edges'].tolist())
assert len(list(pil5_path.glob('repeat_*/fold_*/v3.npz')))==5
freq_path=pil5_path/'analysis/selected_edge_frequency.csv'
with freq_path.open('w',newline='') as out:
    writer=csv.writer(out)
    writer.writerow(['edge_index_0_based','roi_u_id_1_based','roi_v_id_1_based','roi_u_name','roi_v_name','selected_folds','selection_fraction'])
    for e,count in sorted(counts.items(),key=lambda a:(-a[1],a[0])):
        writer.writerow([e,int(u[e])+1,int(v[e])+1,names[u[e]],names[v[e]],count,count/5])

summary=dict(dataset=dict(bytes=377836209,sha256='8e6944e1a3d31abce6123eea14c31a6b38cc92f01df90c8d95aefdcdacff2b6f',
                          subjects=54,ctrl=27,schz=27,rois=129,edges=8256,representation='Fisher z'),
             saved_run=audit,
             fixed_saved_choices={m:dict(folds=len(rows),old_auc=mean(rows,'old_auc'),certified_auc=mean(rows,'new_auc'),
                                   old_selected=mean(rows,'old_selected'),certified_selected=mean(rows,'new_selected'),
                                   mean_objective_gap=mean(rows,'objective_gap'),
                                   changed_auc_folds=sum(abs(r['old_auc']-r['new_auc'])>1e-10 for r in rows),
                                   max_certified_kkt=max(r['certified_kkt']['kkt_max'] for r in rows))
                                  for m,rows in [('v3',v3),('enet',enet)]},
             independent_pilot_3fold={m:{key:pil3['methods'][m][key] for key in ['auc_fold_mean','balanced_accuracy_fold_mean','selected_edges_mean','all_converged']} for m in pil3['methods']},
             independent_v3_5fold={key:pil5['methods']['v3'][key] for key in ['auc_fold_mean','auc_fold_sd','balanced_accuracy_fold_mean','selected_edges_mean','all_converged','stability_overall']})
(root/'validation/real_audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

p=pil5['methods']['v3']
lines=['# Dataset-1 Version 3: verified results and limits', '',
       '## Source data and numerical reconstruction', '',
       f"The user-provided public link yielded the complete **{summary['dataset']['bytes']:,}-byte** MATLAB v7.3 file (SHA-256 `{summary['dataset']['sha256']}`). It contains 27 CTRL and 27 SCHZ subjects at the 129-ROI scale. Upper-triangle Pearson correlations produce 8,256 edges; Fisher transformation and scaling fitted on training subjects reconstruct every saved old fold score to within 1e-5 (in the first checked fold, below 5e-8). The old lambda-max value matches independently to about 5e-11. No subject-level MAT data is redistributed in this package.", '',
       'The old ZIP has 100 fold records (25 per method); all scores, labels, selected edge indices and reported prediction metrics pass an independent consistency check. The 25 repeated folds reuse the same 54 subjects and must not be treated as 25 independent test datasets. The original imported `szv3_pipeline.zip` source was not supplied, so the old inner solver code cannot be reviewed line by line.', '',
       '## Saved run and certified outer refits', '',
       '| Method | Saved mean fold AUC | Saved converged outer fits | Saved mean selected edges | Certified refit mean fold AUC | Certified mean selected edges |',
       '|---|---:|---:|---:|---:|---:|']
for m in ['rfe_svm','enet','v3','lasso']:
    old=audit['methods'][m];new=summary['fixed_saved_choices'].get(m)
    lines.append(f"| {m} | {old['auc']:.4f} | {old['converged_count']}/25 | {old['selected_edges_mean']:.1f} | {new['certified_auc']:.4f} | {new['certified_selected']:.1f} |" if new else
                 f"| {m} | {old['auc']:.4f} | {old['converged_count']}/25 | {old['selected_edges_mean']:.1f} | — | — |")
lines += ['',
          f"The V3 refits changed held-out AUC in {summary['fixed_saved_choices']['v3']['changed_auc_folds']} of 25 folds; the mean AUC moved from 0.7011 to 0.7064. The Elastic Net refits changed {summary['fixed_saved_choices']['enet']['changed_auc_folds']} of 25 fold AUCs; its mean moved from 0.7869 to 0.7517. Maximum independently checked KKT residuals were {summary['fixed_saved_choices']['v3']['max_certified_kkt']:.2g} and {summary['fixed_saved_choices']['enet']['max_certified_kkt']:.2g}, below the 2e-5 acceptance tolerance. Elastic Net used the saved C and l1_ratio with the equivalent mean-loss penalty; the small-data scikit-learn equivalence test is included.", '',
          '**These are numerical refits at hyperparameters chosen by the old inner search.** The old inner fits did not have saved convergence certificates. This exercise diagnoses the outer solver effect but does not retroactively validate the earlier hyperparameter search.', '',
          '## Fresh nested cross-validation pilots', '',
          '| Experiment | Models | Outer × inner folds | V3 mean fold AUC | V3 balanced accuracy | V3 mean selected edges |',
          '|---|---|---|---:|---:|---:|',
          f"| Exploratory comparison | V3, LASSO, Elastic Net, RFE-SVM | 3 × 3 | {pil3['methods']['v3']['auc_fold_mean']:.4f} | {pil3['methods']['v3']['balanced_accuracy_fold_mean']:.4f} | {pil3['methods']['v3']['selected_edges_mean']:.1f} |",
          f"| V3 only | V3 | 5 × 5 | {p['auc_fold_mean']:.4f} | {p['balanced_accuracy_fold_mean']:.4f} | {p['selected_edges_mean']:.1f} |", '',
          f"The five V3 held-out fold AUCs are {', '.join(f'{json.loads(q.read_text())['metrics']['auc']:.3f}' for q in sorted(pil5_path.glob('repeat_*/fold_*/v3.json')))}. All 100 inner and five outer fits completed with checked primal feasibility, dual feasibility, stationarity and complementarity. The V3 grid has four (alpha, ratio) choices: alpha 0.2 or 0.8 crossed with ratio 0.2 or 0.05, ridge 0. The one-standard-error selection rule chose models within the inner folds. Mean pairwise edge Jaccard across five outer supports is {p['stability_overall']['jaccard']:.3f}; high edge frequency in the accompanying CSV is exploratory, not a validated biomarker.", '',
          'The three-fold comparison used two candidates per method and only three outer folds; its reported mean fold AUCs are '+', '.join(f"{m} {pil3['methods'][m]['auc_fold_mean']:.3f}" for m in ['v3','lasso','enet','rfe_svm'])+'. The reconstructed RFE-SVM is not an exact reproduction of the professor’s original baseline without its source and settings.', '',
          '## Interpretation and outstanding experiment', '',
          'The old atlas name array mismatches the Lausanne IDs for **72 of 129 ROIs**. The edge index order is correct, so numerical scores were unaffected, but named connections in the old reports must be relabelled by the atlas ID. The corrected mapping and old-run edge tables are in `old_run_audit/`. `selected_edge_frequency.csv` uses corrected names from the new five-fold pilot.', '',
          'No paired-fold p-value is reported because repeated folds overlap substantially. With only 54 subjects, the results are variable and do not identify a clinical biomarker or a winning model. The complete professor grid (72 V3 configurations, 5 outer folds × 5 repeats for development and 20 repeats for a final run) remains configured but has **not** been executed on real data. These pilots cannot be represented as that final protocol. V1 and V2 are also outside this V3-first rebuild.', '',
          'The independent implementation, full configs, immutable split manifests, fold predictions, probabilities, selected edges, numerical diagnostics, and 17 passing tests are included. To run the prespecified full grid on suitable compute, use `run.py` with `configs/development_full.json` and the explicit `--representation fisher` argument; keep all fold checkpoints and report failed candidates rather than accepting them.', '']
(root/'REAL_DATA_AUDIT.md').write_text('\n'.join(lines))
print('Wrote REAL_DATA_AUDIT.md, validation/real_audit_summary.json, selected_edge_frequency.csv')
