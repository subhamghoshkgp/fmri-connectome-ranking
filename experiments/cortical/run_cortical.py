"""Checkpointed cortical-only Dataset-1 study, both atlas resolutions."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
os.environ['MPLBACKEND']='Agg'
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'original')]
import numpy as np
from joblib import Parallel,delayed
from sklearn.preprocessing import StandardScaler
import run_dataset1_83 as legacy
import run_rank_v3 as ranking
from cortical_data import load_cortical,cortical_atlas,protocol
from szv3.experiment import make_splits
from szv3.storage import atomic_json,atomic_npz,init_manifest,run_lock,RunPaused,check_stop

METHODS=legacy.METHODS
LABELS={'max_auc_v3':'Incidence-gated logistic (reference grid)',
        'v3_expanded':'Incidence-gated logistic (expanded grid)',
        'v3_rank':'Incidence-gated ranking','v3_rank_refit':'Incidence-gated ranking + ridge refit',
        'ridge_rank':'Dense ridge ranking', 'max_auc_enet':'Elastic net',
        'max_auc_lasso':'LASSO','max_auc_rfe_svm':'RFE-SVM',
        'one_se_v3':'Reference-grid gated logistic (one-SE)',
        'one_se_enet':'Elastic net (one-SE)','one_se_lasso':'LASSO (one-SE)',
        'one_se_rfe_svm':'RFE-SVM (one-SE)'}
METRICS=('auc','accuracy','balanced_accuracy','f1','sensitivity','specificity','support')


def source_hash():
    files=sorted(list(HERE.glob('*.py'))+list((HERE/'original/szv3').glob('*.py'))+
        [HERE/'reference/dataset1_129_protocol.json',HERE/'data/CORTICAL_DATA_MANIFEST.json'])
    h=hashlib.sha256()
    for p in files:h.update(p.relative_to(HERE).as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()


def saved_splits(data,cfg):
    splits=make_splits(data.y,cfg['cv'])
    if cfg['cv']==dict(seed=20260929,repeats=20,outer_folds=5,inner_folds=5):
        exact=protocol()['saved_129_splits']
        if splits!=exact:raise ValueError('Generated splits differ from saved original splits')
        splits=exact
    for split in splits:
        tr,te=split['train'],split['test']
        if set(tr)&set(te) or sorted(tr+te)!=list(range(len(data.y))):raise ValueError('Invalid outer split')
        if sorted(i for f in split['inner'] for i in f['valid'])!=sorted(tr):raise ValueError('Invalid inner coverage')
        for f in split['inner']:
            if set(f['train'])&set(f['valid']) or sorted(f['train']+f['valid'])!=sorted(tr):raise ValueError('Invalid inner split')
    return splits


def outer_or_pause(data,split,cfg,out):
    try:
        check_stop(out)
        return legacy.outer(data,split,cfg,out)
    except RunPaused:return 'paused'


def diagnostic_check(record,method,cfg):
    if not record.get('converged'):raise ValueError('Unconverged outer model')
    diag=record['diagnostics']
    if method.endswith('rfe_svm'):
        if diag['relative_duality_gap']>1e-4:raise ValueError('SVM duality certificate exceeds threshold')
    elif method in ('ridge_rank','v3_rank_refit'):
        if diag['gradient_max']>cfg['ranking']['ranking_refit']['gradient_tol']:raise ValueError('Rank-refit gradient certificate exceeds tolerance')
    elif diag['kkt_max']>cfg['ranking' if method in ('v3_rank','v3_expanded') else 'baselines']['solver']['kkt_tol']:
        raise ValueError('KKT certificate exceeds tolerance')
    if method=='v3_rank_refit':
        info=record['selector_info']
        if not info['converged'] or info['diagnostics']['kkt_max']>cfg['ranking']['solver']['kkt_tol']:
            raise ValueError('Refit selector is not KKT certified')


def analyze(out):
    out=Path(out);manifest=json.loads((out/'manifest.json').read_text());cfg=manifest['config']
    with np.load(out/'input_features.npz',allow_pickle=False) as a:X=a['X'].copy()
    if hashlib.sha256(X.tobytes()).hexdigest()!=manifest['preprocessed_X_sha256']:raise ValueError('Feature fingerprint mismatch')
    ids=np.asarray(manifest['data']['original_roi_ids_matlab_1based']);resolution=manifest['data']['original_resolution']
    with np.load(out/'dataset_metadata.npz',allow_pickle=False) as a:meta={k:a[k].copy() for k in a.files}
    atlas_ids,atlas_labels,_=cortical_atlas(resolution)
    np.testing.assert_array_equal(ids,atlas_ids);np.testing.assert_array_equal(meta['roi_names'],atlas_labels)
    np.testing.assert_array_equal(meta['original_roi_ids'],ids)
    choices=[];certs=[];freq={m:np.zeros(X.shape[1],int) for m in METHODS};counts={m:0 for m in METHODS}
    for split in manifest['splits']:
        tr,te=np.asarray(split['train']),np.asarray(split['test'])
        scaler=StandardScaler().fit(X[tr]);Xt=scaler.transform(X[te])
        for m in METHODS:
            stem=out/ranking.fold_dir(split)/m
            if not stem.with_suffix('.json').exists() or not stem.with_suffix('.npz').exists():continue
            r=json.loads(stem.with_suffix('.json').read_text());diagnostic_check(r,m,cfg)
            with np.load(stem.with_suffix('.npz'),allow_pickle=False) as a:
                if a['beta'].shape!=(X.shape[1],):raise ValueError('Wrong coefficient dimension')
                np.testing.assert_allclose(a['scaler_mean'],scaler.mean_,atol=1e-12,rtol=0)
                np.testing.assert_allclose(a['scaler_scale'],scaler.scale_,atol=1e-12,rtol=0)
                np.testing.assert_allclose(Xt@a['beta']+a['intercept'],a['score'],atol=2e-9,rtol=0)
                np.testing.assert_allclose(a['raw_beta'],a['beta']/a['scaler_scale'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(a['raw_intercept'],a['intercept']-a['scaler_mean']@a['raw_beta'],atol=1e-10,rtol=0)
                np.testing.assert_allclose(X[te]@a['raw_beta']+a['raw_intercept'],a['score'],atol=2e-9,rtol=0)
                np.testing.assert_array_equal(a['predicted'],(a['score']>=0).astype(int))
                support=a['selected_edges'];np.testing.assert_array_equal(a['selected_roi_pairs'],np.c_[meta['u'][support],meta['v'][support]])
                if m=='ridge_rank':np.testing.assert_array_equal(support,np.arange(X.shape[1]))
                elif m in ('v3_rank','v3_expanded') or m.endswith(('_v3','_enet','_lasso')):
                    threshold=cfg['ranking' if m in ('v3_rank','v3_expanded') else 'baselines']['support_threshold']
                    np.testing.assert_array_equal(support,np.flatnonzero(np.abs(a['beta'])>threshold))
                elif m=='v3_rank_refit':
                    expected=np.flatnonzero(np.abs(a['selector_beta'])>cfg['ranking']['support_threshold'])
                    np.testing.assert_array_equal(support,expected)
                if np.any(np.abs(a['beta'][np.setdiff1d(np.arange(X.shape[1]),support)])>cfg['baselines']['support_threshold']):
                    raise ValueError('Unexpected coefficient outside selection support')
                freq[m][support]+=1;counts[m]+=1
            # Validate chosen candidate against the committed inner summaries, without fitting.
            base=r.get('base_method',m);tune=stem.parent/(base+'_tuning.json')
            if not tune.exists():raise ValueError('Missing tuning summary for committed outer fit')
            rows=json.loads(tune.read_text())['summary']
            chosen=legacy.choose(rows,r['selection_rule']) if m not in ranking.METHODS else ranking.choose(rows)
            if r['chosen_candidate']!=chosen['candidate'] or r['params']!=chosen['params']:raise ValueError('Chosen inner candidate mismatch')
            choices.append(dict(method=m,repeat=split['repeat'],fold=split['fold'],chosen_candidate=r['chosen_candidate'],
                inner_mean_auc=r['inner_mean_auc'],params_json=json.dumps(r['params'],sort_keys=True),
                selector_params_json=json.dumps(r.get('selector_params'),sort_keys=True)))
            certs.append(dict(method=m,repeat=split['repeat'],fold=split['fold'],converged=r['converged'],
                status=r['status'],solver=r.get('solver','libsvm'),diagnostics_json=json.dumps(r['diagnostics'],sort_keys=True),
                selector_diagnostics_json=json.dumps((r.get('selector_info') or {}).get('diagnostics'),sort_keys=True)))
    # Original numerical aggregation validates split coverage, stored metrics, supports and hubs.
    result=legacy.analyze(out)
    result['verification']='Scalers, scores, raw coefficients, support masks, atlas indices, inner candidate choices, convergence/KKT, saved splits, metrics and hubs checked.'
    result['anatomical_scope']='Cortical-to-cortical only; subcortical and brainstem ROIs excluded from both axes.'
    result['available_edges']=X.shape[1]
    dest=out/'analysis'
    for m,s in result['methods'].items():
        if s['completed_folds']:s['support_fraction_mean']=s['support_mean']/X.shape[1]
    atomic_json(dest/'summary.json',result)
    md=(dest/'SUMMARY.md').read_text().replace('# Dataset-1 83-region: all-model comparison',
        f'# Cortical-only Dataset-1: {resolution} original regions → {len(ids)} cortical regions')
    md+='\n\n'+result['verification']+'\nCortical-only performance is a hypothesis to test, not a guaranteed improvement.\n'
    (dest/'SUMMARY.md').write_text(md)
    ranking.csv_file(dest/'chosen_candidates.csv',choices);ranking.csv_file(dest/'convergence_certificates.csv',certs)
    region_rows=[dict(cortical_index_0based=i,original_matlab_id=int(idx),label=str(meta['roi_names'][i])) for i,idx in enumerate(ids)]
    edge_rows=[dict(edge_index=e,cortical_u_0based=int(u),cortical_v_0based=int(v),
        original_u_matlab_id=int(ids[u]),original_v_matlab_id=int(ids[v]),
        label_u=str(meta['roi_names'][u]),label_v=str(meta['roi_names'][v])) for e,(u,v) in enumerate(zip(meta['u'],meta['v']))]
    ranking.csv_file(dest/'cortical_regions.csv',region_rows);ranking.csv_file(dest/'cortical_edges.csv',edge_rows)
    ranking.csv_file(dest/'method_summary.csv',[dict(method=m,label=LABELS[m],completed_folds=s['completed_folds'],**{k:s[k] for k in ('auc_mean','auc_fold_sd','balanced_accuracy_mean','f1_mean','sensitivity_mean','specificity_mean','support_mean','support_fraction_mean')},jaccard=(s['stability'] or {}).get('jaccard'),nogueira=(s['stability'] or {}).get('nogueira')) for m,s in result['methods'].items() if s['completed_folds']])
    for m in METHODS:
        if counts[m] and m!='ridge_rank':ranking.csv_file(dest/(m+'_edge_frequency.csv'),
            [dict(edge_rows[e],selected_fits=int(freq[m][e]),completed_fits=counts[m],selection_frequency=float(freq[m][e]/counts[m])) for e in range(X.shape[1])])
    if manifest['splits']==protocol()['saved_129_splits']:
        compare_fullbrain(out,resolution,result['complete'] and len(manifest['splits'])==100)
    return result


def read_metrics(path):
    result={}
    with Path(path).open(newline='') as f:
        for r in csv.DictReader(f):
            m=r['method'].replace('saved_max_auc_','max_auc_').replace('saved_one_se_','one_se_')
            key=(m,int(r['repeat']),int(r['fold']))
            if key in result:raise ValueError('Duplicate reference metric row')
            result[key]={k:float(r[k]) for k in METRICS}
    return result


def compare_fullbrain(out,resolution,complete):
    a=read_metrics(out/'analysis/fold_metrics.csv') if (out/'analysis/fold_metrics.csv').exists() else {}
    b=read_metrics(HERE/f'reference/full_brain_{resolution}_fold_metrics.csv')
    if complete and (set(a)!=set(b) or len(a)!=1200):raise ValueError('Full-brain comparison fold coverage mismatch')
    rows=[];summary=[]
    for key in sorted(set(a)&set(b)):
        m,r,f=key;row=dict(method=m,repeat=r,fold=f)
        for metric in METRICS:
            row[metric+'_cortical']=a[key][metric];row[metric+'_fullbrain']=b[key][metric]
            row[metric+'_cortical_minus_fullbrain']=a[key][metric]-b[key][metric]
        rows.append(row)
    for m in METHODS:
        rr=[r for r in rows if r['method']==m]
        if not rr:continue
        s=dict(method=m,paired_folds=len(rr),complete_study=complete,
            note='Same participants and saved folds; descriptive paired differences, not independent-fold inference')
        for metric in METRICS:
            for suffix in ('cortical','fullbrain','cortical_minus_fullbrain'):
                s[metric+'_'+suffix+'_mean']=float(np.mean([r[metric+'_'+suffix] for r in rr]))
        delta=[r['auc_cortical_minus_fullbrain'] for r in rr]
        s.update(auc_wins=sum(d>1e-12 for d in delta),auc_ties=sum(abs(d)<=1e-12 for d in delta),auc_losses=sum(d< -1e-12 for d in delta))
        summary.append(s)
    ranking.csv_file(out/'analysis/cortical_vs_fullbrain_paired_folds.csv',rows)
    ranking.csv_file(out/'analysis/cortical_vs_fullbrain_summary.csv',summary)


def run(data,cfg,out,jobs=1,max_outer=None):
    if not isinstance(jobs,int) or jobs<1 or (max_outer is not None and max_outer<1):raise ValueError('Positive jobs/max-outer required')
    ranking.grids(cfg['ranking']);splits=saved_splits(data,cfg);out=Path(out).resolve()
    if set(cfg['baselines']['grids'])!=set(legacy.BASE):raise ValueError('All baseline grids are required')
    with run_lock(out):
        init_manifest(out,dict(config=cfg,data=data.provenance,splits=splits,methods=list(METHODS),
            source_sha256=source_hash(),environment=ranking.versions(),
            preprocessed_X_sha256=hashlib.sha256(data.X.tobytes()).hexdigest()))
        (out/'STOP_REQUESTED').unlink(missing_ok=True)
        if not (out/'input_features.npz').exists():atomic_npz(out/'input_features.npz',X=data.X)
        if not (out/'dataset_metadata.npz').exists():atomic_npz(out/'dataset_metadata.npz',
            y=data.y,u=data.u,v=data.v,subject_ids=data.subject_ids,roi_names=data.roi_names,
            original_roi_ids=np.asarray(data.provenance['original_roi_ids_matlab_1based']))
        if not (out/'DATA_AUDIT.json').exists():atomic_json(out/'DATA_AUDIT.json',data.provenance)
        start=time.monotonic();status='finished'
        try:
            Parallel(n_jobs=jobs)(delayed(outer_or_pause)(data,s,cfg,str(out)) for s in splits[:max_outer])
            if (out/'STOP_REQUESTED').exists():status='paused'
        except BaseException:
            status='interrupted_or_failed';raise
        finally:
            # A failed/interrupting parallel run may still be shutting down; let the explicit
            # analysis command verify committed files after it has exited.
            log=out/'INVOCATIONS.json';history=json.loads(log.read_text()) if log.exists() else []
            history.append(dict(start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(time.time()-(time.monotonic()-start))),
                elapsed_seconds=time.monotonic()-start,jobs=jobs,max_outer=max_outer,status=status))
            atomic_json(log,history)
        return analyze(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resolution',type=int,choices=(83,129),required=True)
    p.add_argument('--data');p.add_argument('--out',required=True)
    p.add_argument('--config',default=str(HERE/'configs/full.json'));p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--max-outer',type=int);p.add_argument('--analyze-only',action='store_true');p.add_argument('--inspect-only',action='store_true')
    args=p.parse_args()
    if args.analyze_only:
        with run_lock(args.out):s=analyze(args.out)
        print('Verified complete:',s['complete']);return
    path=args.data or str(HERE/f'data/cortical_{args.resolution}.npz');d=load_cortical(path,args.resolution)
    if args.inspect_only:print(json.dumps(d.provenance,indent=2));return
    s=run(d,json.loads(Path(args.config).read_text()),args.out,args.jobs,args.max_outer)
    print('Verified complete:',s['complete'])
if __name__=='__main__':main()
