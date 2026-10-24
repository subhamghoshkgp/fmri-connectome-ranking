"""Descriptive reports; deliberately no automatic GOOD/BAD verdict or naive p-values."""
import csv
import json
from pathlib import Path
import numpy as np
from .metrics import stability,prediction_metrics
from .storage import atomic_json


def csv_file(path,rows):
    if not rows: return
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def analyze(out):
    out=Path(out); manifest=json.loads((out/'manifest.json').read_text())
    cfg=manifest['config']; meta=np.load(out/'dataset_metadata.npz',allow_pickle=False)
    p=len(meta['u']); n=len(meta['y']); expected=cfg['repeats']*cfg['outer_folds']
    report=out/'analysis'; report.mkdir(exist_ok=True)
    result=dict(kind=manifest['data']['kind'],expected_folds_per_method=expected,methods={},
       interpretation='Descriptive CV summaries only. Fold SD is not a confidence interval; repeated folds share subjects.')
    metric_rows=[]; all_records=[]
    for method in cfg['grids']:
        pairs=[]
        for path in sorted(out.glob(f'repeat_*/fold_*/{method}.json')):
            if not path.with_suffix('.npz').exists(): continue
            record=json.loads(path.read_text()); arrays=dict(np.load(path.with_suffix('.npz'),allow_pickle=False))
            recomputed=prediction_metrics(arrays['y_true'],arrays['score'],arrays['predicted'])
            if any(abs(recomputed[k]-record['metrics'][k])>1e-10 for k in recomputed):
                raise ValueError(f'Metric mismatch in {path}')
            pairs.append((record,arrays)); all_records.append(record)
            metric_rows.append(dict(method=method,repeat=record['repeat'],fold=record['fold'],
                                   **record['metrics'],selected_edges=record['selected_edge_count']))
        if not pairs:
            result['methods'][method]=dict(completed_folds=0); continue
        records=[r for r,a in pairs]; supports=[a['selected_edges'] for r,a in pairs]
        m=dict(completed_folds=len(pairs),all_converged=all(r['converged'] for r in records))
        for key in records[0]['metrics']:
            vals=[r['metrics'][key] for r in records]
            m[key+'_fold_mean']=float(np.mean(vals))
            m[key+'_fold_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        m['selected_edges_mean']=float(np.mean([len(s) for s in supports]))
        m['stability_overall']=stability(supports,p)
        m['per_repeat']=[]
        for repeat in range(cfg['repeats']):
            pp=[(r,a) for r,a in pairs if r['repeat']==repeat]
            row=dict(repeat=repeat,fold_count=len(pp))
            if pp:
                row['stability']=stability([a['selected_edges'] for r,a in pp],p)
            if len(pp)==cfg['outer_folds']:
                indices=np.concatenate([a['test_indices'] for r,a in pp])
                if sorted(indices.tolist())!=list(range(n)):
                    raise ValueError('OOF coverage must be exactly once per subject per repeat')
                row['pooled_oof_descriptive']=prediction_metrics(
                    np.concatenate([a['y_true'] for r,a in pp]),
                    np.concatenate([a['score'] for r,a in pp]),
                    np.concatenate([a['predicted'] for r,a in pp]))
            m['per_repeat'].append(row)
        result['methods'][method]=m
        beta=np.stack([a['beta'] for r,a in pairs]); selected=np.zeros_like(beta,dtype=bool)
        for i,s in enumerate(supports): selected[i,s]=True
        frequency=selected.mean(0); incident=np.zeros((len(pairs),len(meta['roi_names'])))
        for i,(r,a) in enumerate(pairs): incident[i,a['incident_rois']]=1
        edge_rows=[dict(edge_index=e,roi_u_id=int(meta['u'][e])+1,roi_v_id=int(meta['v'][e])+1,
                       roi_u=str(meta['roi_names'][meta['u'][e]]),roi_v=str(meta['roi_names'][meta['v'][e]]),
                       frequency=float(frequency[e]),mean_beta=float(beta[:,e].mean()),
                       mean_abs_beta=float(np.abs(beta[:,e]).mean())) for e in range(p)]
        csv_file(report/f'{method}_edge_frequency.csv',sorted(edge_rows,key=lambda r:-r['frequency']))
        z=np.stack([a['z'] for r,a in pairs]) if method=='v3' else None
        roi_rows=[dict(roi_id=i+1,roi=str(name),incident_support_frequency=float(incident[:,i].mean()),
                      mean_capacity=float(z[:,i].mean()) if z is not None else None,
                      capacity_frequency=float((z[:,i]>cfg['support_threshold']).mean()) if z is not None else None)
                  for i,name in enumerate(meta['roi_names'])]
        csv_file(report/f'{method}_roi_frequency.csv',roi_rows)
        if method=='v3':
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            N=len(meta['roi_names']); matrices=[]
            for values in [frequency,np.abs(beta).mean(0)]:
                matrix=np.zeros((N,N)); matrix[meta['u'],meta['v']]=values; matrix+=matrix.T
                matrices.append(matrix)
            fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
            for ax,matrix,title in zip(axes,matrices,['Edge selection frequency','Mean absolute standardized coefficient']):
                im=ax.imshow(matrix,cmap='viridis'); ax.set_title(title,fontsize=10)
                ax.set_xlabel('ROI index (zero based)'); ax.set_ylabel('ROI index (zero based)')
                fig.colorbar(im,ax=ax,shrink=.75)
            fig.suptitle('SYNTHETIC validation only' if result['kind']=='synthetic_only' else 'Dataset-1 V3 descriptive maps')
            fig.savefig(report/'v3_maps.png',dpi=160); plt.close(fig)
    result['complete']=all(m['completed_folds']==expected for m in result['methods'].values())
    # Paired differences are descriptive; no independent-fold significance test.
    differences=[]
    by={(r['method'],r['repeat'],r['fold']):r for r in all_records}
    for method in cfg['grids']:
        if method=='v3': continue
        for r in all_records:
            if r['method']!='v3': continue
            b=by.get((method,r['repeat'],r['fold']))
            if b: differences.append(dict(comparator=method,repeat=r['repeat'],fold=r['fold'],
                    auc_difference=r['metrics']['auc']-b['metrics']['auc'],
                    support_difference=r['selected_edge_count']-b['selected_edge_count']))
    csv_file(report/'fold_metrics.csv',metric_rows)
    csv_file(report/'paired_differences_descriptive.csv',differences)
    atomic_json(report/'summary.json',result)
    lines=['# Experiment summary','',f"Data: **{result['kind']}**. Complete: **{result['complete']}**.",'',
           'Fold means and standard deviations are descriptive, not confidence intervals. No significance claim is made.','',
           '| Method | Folds | AUC mean | AUC fold SD | Selected edges mean | Jaccard |',
           '|---|---:|---:|---:|---:|---:|']
    for method,m in result['methods'].items():
        if m['completed_folds']:
            sd=m['auc_fold_sd']; jac=m['stability_overall']['jaccard']
            lines.append(f"| {method} | {m['completed_folds']}/{expected} | {m['auc_fold_mean']:.4f} | "
                         f"{sd if sd is not None else 'NA'} | {m['selected_edges_mean']:.1f} | {jac} |")
    lines+=['','The RFE-SVM is an explicitly configured reconstruction; historical code was not supplied.',
            'Standard Kuncheva is reported only for equal nontrivial support sizes. See summary.json for valid-pair counts, unequal-size adjusted overlap, Nogueira stability, and per-repeat outputs.',
            'An incomplete run must not be presented as the final experiment.']
    (report/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    return result
