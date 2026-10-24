"""Dataset-1 83-region: eight unchanged models, 12 selection variants, shared nested CV."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
os.environ['MPLBACKEND']='Agg'
import argparse,csv,hashlib,itertools,json,sys,time
from pathlib import Path
import numpy as np
from joblib import Parallel,delayed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'original'))
from dataset1_83 import load_dataset83
from szv3.solver import LogisticProblem,intercept_only_solution
from szv3.baselines import fit_svm
from szv3.experiment import make_splits,penalties
from szv3.metrics import prediction_metrics,stability
from szv3.storage import atomic_json,atomic_npz,init_manifest,run_lock,check_stop,RunPaused
import run_rank_v3 as ranking_runner

BASE=('v3','enet','lasso','rfe_svm')
NEW=ranking_runner.METHODS
METHODS=tuple(f'{rule}_{m}' for m in BASE for rule in ('max_auc','one_se'))+NEW

def choose(rows,rule):
    valid=[r for r in rows if r['valid']]
    if not valid:raise RuntimeError('No candidate valid in every inner fold')
    if rule=='max_auc':return min(valid,key=lambda r:(-r['mean_auc'],r['mean_support'],r['candidate']))
    if rule!='one_se':raise ValueError('Unknown rule')
    best=max(valid,key=lambda r:r['mean_auc'])
    acceptable=[r for r in valid if r['mean_auc']>=best['mean_auc']-best['se_auc']]
    return min(acceptable,key=lambda r:(r['mean_support'],-r['mean_auc'],r['candidate']))

def csv_file(path,rows):ranking_runner.csv_file(path,rows)

def fit_baseline_inner(data,split,cfg,root,method):
    records=[]
    for j,inner in enumerate(split['inner']):
        a,b=np.array(inner['train']),np.array(inner['valid'])
        scaler=StandardScaler().fit(data.X[a]);X=scaler.transform(data.X[a]);Xv=scaler.transform(data.X[b])
        problem=None if method=='rfe_svm' else LogisticProblem(X,data.y[a],data.H,method=='v3')
        for c,param in enumerate(cfg['grids'][method]):
            path=root/'inner_baselines'/method/f'{j:02d}_{c:03d}.json'
            identity=dict(inner=j,candidate=c,params=param,train=a.tolist(),valid_indices=b.tolist())
            def work():
                if method=='rfe_svm':
                    fit=fit_svm(data.X[a],data.y[a],data.X[b],param,split['inner_seed'])
                    score=fit['score'];support=len(fit['support']);lm=None
                else:
                    le,lv,l2,lm=penalties(X,data.y[a],data.H,method,param)
                    fit=intercept_only_solution(X,data.y[a],data.H,le,lv,l2,method=='v3') if param['ratio']==1 else problem.solve(le,lv,l2,**cfg['solver'])
                    score=Xv@fit['beta']+fit['intercept'];support=int(np.sum(np.abs(fit['beta'])>cfg['support_threshold']))
                return dict(auc=float(roc_auc_score(data.y[b],score)),support=support,lambda_max=lm,
                    score=score.tolist(),converged=fit['converged'],diagnostics=fit['diagnostics'],
                    status=fit['status'],runtime_seconds=fit['runtime_seconds'],attempts=fit.get('attempts'))
            row=ranking_runner.inner_record(path,identity,work);records.append(row)
    summary=[]
    for c,param in enumerate(cfg['grids'][method]):
        rr=[r for r in records if r['candidate']==c]
        valid=sorted(r['inner'] for r in rr)==list(range(len(split['inner']))) and all(r['valid'] for r in rr)
        summary.append(dict(candidate=c,params=param,valid=valid,
            mean_auc=float(np.mean([r['auc'] for r in rr])) if valid else None,
            se_auc=float(np.std([r['auc'] for r in rr],ddof=1)/np.sqrt(len(rr))) if valid else None,
            mean_support=float(np.mean([r['support'] for r in rr])) if valid else None))
    atomic_json(root/(method+'_tuning.json'),dict(summary=summary,inner=records))
    return summary

def baseline_outer(data,split,cfg,root,method):
    if all((root/f'{rule}_{method}.json').exists() and (root/f'{rule}_{method}.npz').exists() for rule in ('max_auc','one_se')):return
    rows=fit_baseline_inner(data,split,cfg,root,method)
    train,test=np.asarray(split['train']),np.asarray(split['test'])
    scaler=StandardScaler().fit(data.X[train]);X=scaler.transform(data.X[train]);Xt=scaler.transform(data.X[test])
    fits={}
    for rule in ('max_auc','one_se'):
        check_stop(root)
        name=f'{rule}_{method}';path=root/name
        if path.with_suffix('.json').exists() and path.with_suffix('.npz').exists():continue
        selected=choose(rows,rule);param=selected['params'];c=selected['candidate'];start=time.perf_counter()
        try:
            if c not in fits:
                if method=='rfe_svm':
                    fit=fit_svm(data.X[train],data.y[train],data.X[test],param,split['inner_seed'],True)
                    beta=fit['beta'];z=np.array([]);score=fit['score'];pred=fit['predicted'];prob=fit['probability']
                    support=fit['support'];mean=fit['scaler_mean'];scale=fit['scaler_scale'];lm=None
                else:
                    le,lv,l2,lm=penalties(X,data.y[train],data.H,method,param)
                    fit=intercept_only_solution(X,data.y[train],data.H,le,lv,l2,method=='v3') if param['ratio']==1 else LogisticProblem(X,data.y[train],data.H,method=='v3').solve(le,lv,l2,**cfg['solver'])
                    beta=fit['beta'];z=fit['z'];score=Xt@beta+fit['intercept'];pred=(score>=0).astype(int)
                    from scipy.special import expit
                    prob=expit(score);support=np.flatnonzero(np.abs(beta)>cfg['support_threshold']);mean=scaler.mean_;scale=scaler.scale_
                arrays=dict(beta=beta,z=z,intercept=np.array(fit['intercept']),raw_beta=beta/scale,
                    raw_intercept=np.array(fit['intercept']-mean@(beta/scale)),scaler_mean=mean,scaler_scale=scale,
                    train_indices=train,test_indices=test,y_true=data.y[test],score=score,predicted=pred,
                    probability=prob,selected_edges=support,selected_roi_pairs=np.c_[data.u[support],data.v[support]])
                info={k:v for k,v in fit.items() if k not in ['beta','z','support','score','predicted','probability','scaler_mean','scaler_scale']}
                fits[c]=(arrays,info,lm)
            arrays,info,lm=fits[c];support=arrays['selected_edges']
            atomic_npz(path.with_suffix('.npz'),**arrays)
            record=dict(info,method=name,base_method=method,repeat=split['repeat'],fold=split['fold'],
                train=train.tolist(),test=test.tolist(),params=param,chosen_candidate=c,selection_rule=rule,
                inner_mean_auc=selected['mean_auc'],lambda_max=lm,selected_edge_count=len(support),
                coefficient_nonzero_count=int(np.sum(np.abs(arrays['beta'])>cfg['support_threshold'])),
                metrics=prediction_metrics(arrays['y_true'],arrays['score'],arrays['predicted']),
                hub_concentration=ranking_runner.hub(data,support,arrays['beta']) if method=='v3' else None,
                total_outer_seconds=time.perf_counter()-start,outer_refit_shared_candidate=c in [choose(rows,'max_auc')['candidate']] and rule=='one_se')
            atomic_json(path.with_suffix('.json'),record)
            print(f"repeat={split['repeat']:02d} fold={split['fold']} {name}: AUC={record['metrics']['auc']:.4f}, features={len(support)}",flush=True)
        except Exception as exc:
            atomic_json(root/(name+'_failed.json'),dict(method=name,selection=selected,error=f'{type(exc).__name__}: {exc}'))
            raise

def outer(data,split,cfg,out):
    root=Path(out)/ranking_runner.fold_dir(split);root.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        for m in BASE:
            check_stop(root)
            baseline_outer(data,split,cfg['baselines'],root,m)
        ranking_runner.outer_limited(data,split,cfg['ranking'],Path(out))

def analyze(out):
    out=Path(out);manifest=json.loads((out/'manifest.json').read_text())
    with np.load(out/'dataset_metadata.npz',allow_pickle=False) as a:meta={k:a[k].copy() for k in a.files}
    expected=len(manifest['splits']);p=len(meta['u']);groups={m:[] for m in METHODS};supports={m:[] for m in METHODS}
    rows=[];predictions=[];hubs=[];by={};strows=[];srrows=[];rruntime=[]
    for split in manifest['splits']:
        for m in METHODS:
            path=out/ranking_runner.fold_dir(split)/m
            if not path.with_suffix('.json').exists() or not path.with_suffix('.npz').exists():continue
            r=json.loads(path.with_suffix('.json').read_text())
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
                for k,v in [('train_indices',split['train']),('test_indices',split['test']),('y_true',meta['y'][split['test']])]:np.testing.assert_array_equal(a[k],v)
                metrics=prediction_metrics(a['y_true'],a['score'],a['predicted'])
                if any(abs(metrics[k]-r['metrics'][k])>1e-10 for k in metrics):raise ValueError('Stored metric mismatch')
                support=a['selected_edges'].copy()
                if not r['converged'] or len(support)!=r['selected_edge_count'] or len(np.unique(support))!=len(support) or np.any(support<0) or np.any(support>=p):raise ValueError('Invalid outer fit')
                if not np.isfinite(a['beta']).all() or not np.isfinite(a['score']).all():raise ValueError('Nonfinite model')
                for k,i in enumerate(a['test_indices']):
                    probability=float(a['probability'][k]) if 'probability' in a and len(a['probability']) else None
                    predictions.append(dict(method=m,repeat=split['repeat'],fold=split['fold'],subject_id=str(meta['subject_ids'][i]),
                        subject_index=int(i),y_true=int(a['y_true'][k]),score=float(a['score'][k]),predicted=int(a['predicted'][k]),probability=probability))
                if r.get('hub_concentration') is not None:
                    d=type('Metadata',(),dict(u=meta['u'],v=meta['v'],roi_names=meta['roi_names']))
                    h=ranking_runner.hub(d,support,a['beta'])
                    for k,v in h.items():
                        if v is not None and abs(v-r['hub_concentration'][k])>1e-10:raise ValueError('Hub statistic mismatch')
                    hubs.append(dict(method=m,repeat=split['repeat'],fold=split['fold'],**h))
            groups[m].append(r);supports[m].append(support);by[(m,split['repeat'],split['fold'])]=r
            rows.append(dict(method=m,repeat=split['repeat'],fold=split['fold'],**metrics,support=len(support)))
            rruntime.append(dict(method=m,repeat=split['repeat'],fold=split['fold'],runtime_seconds=r.get('runtime_seconds'),
                total_outer_seconds=r.get('total_outer_seconds'),solver=r.get('solver'),converged=r['converged']))
    summary=dict(complete=all(len(groups[m])==expected for m in METHODS),expected_folds_per_method=expected,
        protocol=manifest['config']['protocol'],data=manifest['data'],methods={},
        interpretation='Fold variability and paired differences are descriptive; repeated folds reuse subjects. Frozen grids are inherited from Dataset-1, with unequal search budgets.',
        partial_export_allowed=True,diagnostics='Metrics/splits/supports/hubs validated from committed model arrays. Inner fits are stored individually.')
    for m,rr in groups.items():
        s=dict(completed_folds=len(rr),all_converged=all(r['converged'] for r in rr))
        if rr:
            for metric in rr[0]['metrics']:
                values=[r['metrics'][metric] for r in rr];s[metric+'_mean']=float(np.mean(values));s[metric+'_fold_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
            s['support_mean']=float(np.mean([r['selected_edge_count'] for r in rr]))
            s['stability']=None
            if m!='ridge_rank':
                st=stability(supports[m],p);st.pop('pairs');s['stability']=st;strows.append(dict(method=m,**st))
                for repeat in sorted({r['repeat'] for r in rr}):
                    ss=[a for a,r in zip(supports[m],rr) if r['repeat']==repeat]
                    st=stability(ss,p);st.pop('pairs');srrows.append(dict(method=m,repeat=repeat,folds=len(ss),**st))
            hh=[h for h in hubs if h['method']==m]
            s['hub_concentration_mean']={k:float(np.mean([h[k] for h in hh if h[k] is not None])) if any(h[k] is not None for h in hh) else None for k in ['degree_hhi','degree_top5_share','strength_hhi','strength_top5_share']} if hh else None
        summary['methods'][m]=s
    paired=[];ps=[]
    comparison_order=NEW+tuple(m for m in METHODS if m not in NEW)
    comparisons=list(itertools.combinations(comparison_order,2))
    for m,b in comparisons:
        deltas=[]
        for split in manifest['splits']:
            a=by.get((m,split['repeat'],split['fold']));z=by.get((b,split['repeat'],split['fold']))
            if a is None or z is None:continue
            row=dict(method=m,comparator=b,repeat=split['repeat'],fold=split['fold'],support_delta=a['selected_edge_count']-z['selected_edge_count'])
            row.update({k+'_delta':a['metrics'][k]-z['metrics'][k] for k in a['metrics']});paired.append(row);deltas.append(row['auc_delta'])
        if deltas:ps.append(dict(method=m,comparator=b,paired_folds=len(deltas),mean_auc_delta=float(np.mean(deltas)),
            delta_fold_sd=float(np.std(deltas,ddof=1)) if len(deltas)>1 else None,
            wins=sum(d>0 for d in deltas),ties=sum(d==0 for d in deltas),losses=sum(d<0 for d in deltas)))
    changes=[]
    for m in BASE:
        pairs=[(by[('max_auc_'+m,s['repeat'],s['fold'])],by[('one_se_'+m,s['repeat'],s['fold'])]) for s in manifest['splits'] if ('max_auc_'+m,s['repeat'],s['fold']) in by and ('one_se_'+m,s['repeat'],s['fold']) in by]
        changes.append(dict(method=m,paired_folds=len(pairs),changed_candidates=sum(a['chosen_candidate']!=b['chosen_candidate'] for a,b in pairs)))
    summary['paired_auc_descriptive']=ps;summary['selection_changes']=changes
    dest=out/'analysis';dest.mkdir(exist_ok=True)
    atomic_json(dest/'summary.json',summary)
    for name,rr in [('fold_metrics',rows),('subject_predictions',predictions),('paired_fold_comparisons',paired),('paired_auc_summary',ps),('stability',strows),('stability_per_repeat',srrows),('hub_concentration',hubs),('runtime',rruntime),('selection_changes',changes)]:csv_file(dest/(name+'.csv'),rr)
    lines=['# Dataset-1 83-region: all-model comparison','',f"Complete: **{summary['complete']}**. Expected outer folds per variant: **{expected}**.",'',
        '| Method | Folds | Mean AUC | Fold SD | Balanced accuracy | F1 | Sensitivity | Specificity | Mean edges | Jaccard | Nogueira |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'NA' if x is None else f'{x:.4f}'
    for m,s in summary['methods'].items():
        if not s['completed_folds']:continue
        st=s['stability'] or {};lines.append('| '+' | '.join([m,str(s['completed_folds'])]+[fmt(s[k]) for k in ['auc_mean','auc_fold_sd','balanced_accuracy_mean','f1_mean','sensitivity_mean','specificity_mean','support_mean']]+[fmt(st.get(k)) for k in ['jaccard','nogueira']])+' |')
    lines+=['','Primary endpoint: mean outer-fold AUC. Ranking scores are not calibrated probabilities.','',summary['interpretation'],'','All 54 subjects are retained: 27 per class. No augmentation or synthetic subjects are introduced.','',manifest['config']['protocol']]
    (dest/'SUMMARY.md').write_text('\n'.join(lines)+'\n');return summary

def source_hash():
    h=hashlib.sha256()
    for path in sorted([HERE/'run_dataset1_83.py',HERE/'dataset1_83.py',HERE/'compare_resolutions.py',HERE/'reference/dataset1_129_protocol.json',HERE/'run_rank_v3.py',HERE/'ranking.py']+list((HERE/'original/szv3').glob('*.py'))):
        h.update(str(path.relative_to(HERE)).encode());h.update(path.read_bytes())
    return h.hexdigest()

def run(data,cfg,out,jobs=1,max_outer=None):
    if jobs<1 or (max_outer is not None and max_outer<1):raise ValueError('Positive jobs/max-outer required')
    ranking_runner.grids(cfg['ranking'])
    if set(cfg['baselines']['grids'])!=set(BASE):raise ValueError('All four baselines required')
    splits=make_splits(data.y,cfg['cv']);out=Path(out).resolve()
    if cfg['cv']==dict(seed=20260929,repeats=20,outer_folds=5,inner_folds=5):
        saved=json.loads((HERE/'reference/dataset1_129_protocol.json').read_text())['saved_129_splits']
        if splits!=saved:raise ValueError('CV split mismatch against completed 129-region study')
        splits=saved
    with run_lock(out):
        (out/'STOP_REQUESTED').unlink(missing_ok=True)
        init_manifest(out,dict(config=cfg,data=data.provenance,splits=splits,source_sha256=source_hash(),
            environment=ranking_runner.versions(),preprocessed_X_sha256=hashlib.sha256(data.X.tobytes()).hexdigest(),methods=list(METHODS)))
        atomic_npz(out/'dataset_metadata.npz',y=data.y,u=data.u,v=data.v,subject_ids=data.subject_ids,roi_names=data.roi_names)
        atomic_json(out/'DATA_AUDIT.json',data.provenance)
        try:
            # Stop flags are checked between fits. A worker returning normally on
            # RunPaused lets the other active worker finish its current fit too.
            Parallel(n_jobs=jobs)(delayed(outer_or_pause)(data,s,cfg,str(out)) for s in splits[:max_outer])
        finally:result=analyze(out)
        if result['complete'] and len(splits)==100:
            from compare_resolutions import compare
            compare(HERE/'reference/dataset1_129_fold_metrics.csv',out/'analysis/fold_metrics.csv',out/'analysis')
        return result

def outer_or_pause(data,split,cfg,out):
    try: outer(data,split,cfg,out)
    except RunPaused: return 'paused'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default=str(HERE/'data/dataset1_83.mat'));parser.add_argument('--out',required=True)
    parser.add_argument('--config',default=str(HERE/'configs/full.json'));parser.add_argument('--jobs',type=int,default=1)
    parser.add_argument('--max-outer',type=int);parser.add_argument('--analyze-only',action='store_true');parser.add_argument('--inspect-only',action='store_true')
    args=parser.parse_args()
    if args.analyze_only:
        with run_lock(args.out): result=analyze(args.out)
        print(json.dumps(result,indent=2));return
    cfg=json.loads(Path(args.config).read_text())
    data=load_dataset83(args.data,cfg['representation'])
    if args.inspect_only:print(json.dumps(data.provenance,indent=2));return
    s=run(data,cfg,args.out,args.jobs,args.max_outer)
    print('Complete:',s['complete'])
if __name__=='__main__':main()
