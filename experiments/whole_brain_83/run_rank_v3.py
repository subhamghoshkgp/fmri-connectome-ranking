"""New subject-level nested-CV experiment. Existing reference results are read only."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import csv
import hashlib
import importlib.metadata
import io
import itertools
import json
from pathlib import Path
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'original'))
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from szv3.data import load_dataset
from szv3.math import lambda_max
from szv3.solver import LogisticProblem
from szv3.metrics import prediction_metrics, stability
from szv3.storage import atomic_json, atomic_npz, init_manifest, run_lock, environment, check_stop
from ranking import RankingV3, RidgeRanker, ranking_lambda_max, ranking_intercept

METHODS = ('v3_expanded','v3_rank','v3_rank_refit','ridge_rank')
REFERENCE_METHODS = ('v3','enet','lasso','rfe_svm')
RULE = 'max_inner_auc_exact_ties_support_then_index'


class Reference:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.z = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.z:
            names = self.z.namelist()
            manifests = [n for n in names if Path(n).name == 'manifest.json']
            if len(manifests) != 1 or len(names) != len(set(names)):
                raise ValueError('Expected a single results manifest and unique ZIP paths')
            self.prefix = manifests[0][:-len('manifest.json')]
        else:
            self.prefix = ''
        self.manifest = self.json('manifest.json')

    def read(self, name):
        return self.z.read(self.prefix+name) if self.z else (self.path/name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))

    def arrays(self, name):
        with np.load(io.BytesIO(self.read(name)), allow_pickle=False) as a:
            return {k:a[k].copy() for k in a.files}

    def close(self):
        if self.z: self.z.close()


def grids(cfg):
    if cfg['selection_rule'] != RULE or set(cfg['methods']) != set(METHODS):
        raise ValueError('The four prespecified experiment arms must be present')
    if any(not 0<a<1 for a in cfg['alpha']) or any(not 0<r<1 for r in cfg['ratio']):
        raise ValueError('Require 0<alpha<1 and 0<ratio<1')
    if any(r<=0 for r in cfg['ridge']+cfg['rho']):
        raise ValueError('All ridge strengths must be positive')
    for axis in ('alpha','ratio','ridge','rho'):
        if not cfg[axis] or len(cfg[axis])!=len(set(cfg[axis])):
            raise ValueError('Grid axes must be nonempty and contain no duplicates')
    if cfg['rank_prediction_threshold']!='training_class_score_midpoint':
        raise ValueError('Unsupported ranking classification threshold')
    selectors = [dict(alpha=a,ratio=r,ridge=l2) for a,r,l2 in itertools.product(cfg['alpha'],cfg['ratio'],cfg['ridge'])]
    refits = [dict(selector=s,rho=rho) for s,rho in itertools.product(range(len(selectors)),cfg['rho'])]
    return selectors, dict(v3_expanded=selectors, v3_rank=selectors, v3_rank_refit=refits,
                          ridge_rank=[dict(rho=r) for r in cfg['rho']])


def choose(rows):
    valid = [r for r in rows if r['valid']]
    if not valid: raise RuntimeError('No candidate valid in every inner fold')
    if any(not np.isfinite(r['mean_auc']) or not np.isfinite(r['mean_support']) for r in valid):
        raise ValueError('Nonfinite valid tuning summary')
    return min(valid,key=lambda r:(-r['mean_auc'],r['mean_support'],r['candidate']))


def source_hash():
    h = hashlib.sha256()
    paths = [HERE/'ranking.py', HERE/'run_rank_v3.py'] + sorted((HERE/'original/szv3').glob('*.py'))
    for path in sorted(paths):
        h.update(str(path.relative_to(HERE)).encode());h.update(path.read_bytes())
    return h.hexdigest()


def versions():
    return dict(environment(), extra={p:importlib.metadata.version(p) for p in ('scs','joblib','threadpoolctl')})


def fold_dir(split):
    return f"repeat_{split['repeat']:02d}/fold_{split['fold']:02d}"


def csv_file(path, rows):
    if not rows: return
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending = path.with_suffix('.pending')
    with pending.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(pending,path)


def validate_reference(ref, data):
    m = ref.manifest
    if m.get('selection_rule') != RULE:
        raise ValueError('Reference must be the completed max-inner-AUC re-selection run')
    if data.provenance != m['data']:
        raise ValueError('Dataset/preprocessing differs from reference')
    if hashlib.sha256(np.ascontiguousarray(data.X).tobytes()).hexdigest()!=m['preprocessed_X_sha256']:
        raise ValueError('Preprocessed feature matrix differs from reference')
    if environment() != m['environment']:
        raise ValueError('Match the original logged Python/package versions before running')
    meta = ref.arrays('dataset_metadata.npz')
    for key in ('y','u','v','roi_names','subject_ids'):
        np.testing.assert_array_equal(getattr(data,key),meta[key])
    splits=m['splits'];expected=m['config']['repeats']*m['config']['outer_folds']
    if len(splits)!=expected:raise ValueError('Wrong number of saved splits')
    seen=set();n=len(data.y);p=len(data.u)
    digest=hashlib.sha256(ref.read('manifest.json')+ref.read('dataset_metadata.npz'))
    records=[];array_rows=[]
    for split in splits:
        key=(split['repeat'],split['fold'])
        if key in seen:raise ValueError('Duplicate outer fold')
        seen.add(key)
        tr,te=split['train'],split['test']
        if set(tr)&set(te) or sorted(tr+te)!=list(range(n)):raise ValueError('Bad saved outer split')
        if len(split['inner'])!=m['config']['inner_folds']:raise ValueError('Bad inner fold count')
        for inner in split['inner']:
            if set(inner['train'])&set(inner['valid']) or sorted(inner['train']+inner['valid'])!=sorted(tr):
                raise ValueError('Bad saved inner split')
        if sorted(i for inner in split['inner'] for i in inner['valid'])!=sorted(tr):
            raise ValueError('Saved inner validation coverage must be exactly once')
        for method in REFERENCE_METHODS:
            name=fold_dir(split)+'/'+method
            for suffix in ('.json','.npz'):
                digest.update((name+suffix).encode());digest.update(ref.read(name+suffix))
            r=ref.json(name+'.json');a=ref.arrays(name+'.npz')
            if not r['converged'] or r['selection_rule']!=RULE or r['train']!=tr or r['test']!=te:
                raise ValueError('Reference fold is incomplete or mismatched')
            for selection,pre,record in [('max_auc','',r),('one_se','old_',r['old_result'])]:
                if not record['converged']:raise ValueError('Reference contains an unconverged model')
                np.testing.assert_array_equal(a[pre+'train_indices'],tr)
                np.testing.assert_array_equal(a[pre+'test_indices'],te)
                np.testing.assert_array_equal(a[pre+'y_true'],data.y[te])
                mm=prediction_metrics(a[pre+'y_true'],a[pre+'score'],a[pre+'predicted'])
                if any(abs(mm[k]-record['metrics'][k])>1e-10 for k in mm):raise ValueError('Reference metric mismatch')
                support=a[pre+'selected_edges']
                if (len(support)!=record['selected_edge_count'] or len(np.unique(support))!=len(support)
                        or np.any(support<0) or np.any(support>=p)):
                    raise ValueError('Reference support mismatch')
                label='saved_'+selection+'_'+method
                records.append(dict(method=label,repeat=split['repeat'],fold=split['fold'],metrics=mm,
                    selected_edge_count=len(support),params=record['params'],chosen_candidate=record['chosen_candidate'],
                    converged=record['converged'],diagnostics=record['diagnostics'],runtime_seconds=record['runtime_seconds'],
                    hub_concentration=hub(data,support,a[pre+'beta']) if method=='v3' else None))
                array_rows.append((label,split,a[pre+'score'],a[pre+'predicted'],support))
    for repeat in range(m['config']['repeats']):
        ss=[s for s in splits if s['repeat']==repeat]
        if len(ss)!=m['config']['outer_folds'] or sorted(i for s in ss for i in s['test'])!=list(range(n)):
            raise ValueError('Saved OOF coverage is not exactly once per subject')
    width=max(len(s['test']) for s in splits);N=len(array_rows)
    score=np.zeros((N,width));pred=np.full((N,width),-1,dtype=int);test=np.full((N,width),-1,dtype=int)
    support=np.zeros((N,p),dtype=bool)
    for i,(label,split,scores,labels,edges) in enumerate(array_rows):
        k=len(split['test']);score[i,:k]=scores;pred[i,:k]=labels;test[i,:k]=split['test'];support[i,edges]=True
    arrays=dict(method=np.array([r[0] for r in array_rows]),repeat=np.array([r[1]['repeat'] for r in array_rows]),
        fold=np.array([r[1]['fold'] for r in array_rows]),score=score,predicted=pred,test_indices=test,support=support)
    return digest.hexdigest(),records,arrays


def selector_fit(X,y,H,param,kind,problem,cfg):
    lm=(ranking_lambda_max(X,y,H,param['alpha']) if kind=='rank' else lambda_max(X,y,H,param['alpha']))
    lam=lm*param['ratio']
    fit=problem.solve(param['alpha']*lam,(1-param['alpha'])*lam,param['ridge'],**cfg['solver'])
    fit['lambda_max']=lm
    return fit


def inner_record(path, identity, fit_function):
    check_stop(path)  # Outside the candidate-error handler: stopping is not failure.
    if path.exists():
        r=json.loads(path.read_text())
        if any(r.get(k)!=v for k,v in identity.items()):
            raise ValueError('Cached inner-fit identity mismatch')
        return r
    start=time.perf_counter()
    try:
        r=fit_function();r.update(identity);r.update(valid=True)
    except Exception as exc:
        r=dict(identity,valid=False,error=f'{type(exc).__name__}: {exc}')
    r['checkpoint_seconds']=time.perf_counter()-start
    atomic_json(path,r)
    return r


def fit_inner(data,split,cfg,root):
    selectors,candidate_grids=grids(cfg)
    rows={m:[] for m in METHODS}
    from sklearn.metrics import roc_auc_score
    for inner_id,inner in enumerate(split['inner']):
        a,b=np.asarray(inner['train']),np.asarray(inner['valid'])
        scaler=StandardScaler().fit(data.X[a]);X,Xv=scaler.transform(data.X[a]),scaler.transform(data.X[b])
        cache=root/'inner'/f'{inner_id:02d}'
        logistic_problem=rank_problem=None
        for kind,method in [('logistic','v3_expanded'),('rank','v3_rank')]:
            for c,param in enumerate(selectors):
                path=cache/'selectors'/kind/f'{c:03d}.json'
                mask_path=path.with_suffix('.npz')
                def selector_work():
                    nonlocal logistic_problem,rank_problem
                    if kind=='logistic':
                        if logistic_problem is None:logistic_problem=LogisticProblem(X,data.y[a],data.H,True)
                        problem=logistic_problem
                    else:
                        if rank_problem is None:rank_problem=RankingV3(X,data.y[a],data.H)
                        problem=rank_problem
                    fit=selector_fit(X,data.y[a],data.H,param,kind,problem,cfg)
                    beta,z=fit.pop('beta'),fit.pop('z');support=np.flatnonzero(np.abs(beta)>cfg['support_threshold'])
                    if kind=='logistic':score=Xv@beta+fit['intercept']
                    else:score=Xv@beta+ranking_intercept(X,data.y[a],beta)
                    # Only the mask is needed for ranking refits. This keeps 15,000
                    # selector checkpoints small, rather than saving dense beta arrays.
                    if kind=='rank':atomic_npz(mask_path,selected_edges=support)
                    return dict(candidate=c,params=param,inner=inner_id,train=a.tolist(),valid_indices=b.tolist(),
                        auc=float(roc_auc_score(data.y[b],score)),support=len(support),score=score.tolist(),**fit)
                identity=dict(candidate=c,params=param,inner=inner_id,train=a.tolist(),valid_indices=b.tolist())
                row=inner_record(path,identity,selector_work)
                rows[method].append(row)
                if kind!='rank':continue
                # This selector is shared by all four rho candidates. A fresh
                # validation-fold selector is never fitted outside its train split.
                ranker=None
                if row['valid']:
                    if not mask_path.exists():raise ValueError('Missing mask for committed selector')
                    with np.load(mask_path,allow_pickle=False) as arr:support=arr['selected_edges'].copy()
                for j,rho in enumerate(cfg['rho']):
                    candidate=c*len(cfg['rho'])+j;path=cache/'rank_refit'/f'{candidate:03d}.json'
                    def refit_work():
                        nonlocal ranker
                        if not row['valid']:raise RuntimeError('Selector was invalid')
                        if ranker is None:ranker=RidgeRanker(X[:,support],data.y[a])
                        fit=ranker.fit(rho,**cfg['ranking_refit']);beta=fit.pop('beta')
                        score=Xv[:,support]@beta+ranking_intercept(X[:,support],data.y[a],beta)
                        return dict(candidate=candidate,params=dict(selector=c,rho=rho),inner=inner_id,
                            train=a.tolist(),valid_indices=b.tolist(),auc=float(roc_auc_score(data.y[b],score)),
                            support=len(support),score=score.tolist(),**fit)
                    identity=dict(candidate=candidate,params=dict(selector=c,rho=rho),inner=inner_id,
                        train=a.tolist(),valid_indices=b.tolist())
                    rr=inner_record(path,identity,refit_work)
                    rows['v3_rank_refit'].append(rr)
        ranker=None
        for c,param in enumerate(candidate_grids['ridge_rank']):
            path=cache/'ridge_rank'/f'{c:03d}.json'
            def ridge_work():
                nonlocal ranker
                if ranker is None:ranker=RidgeRanker(X,data.y[a])
                fit=ranker.fit(param['rho'],**cfg['ranking_refit']);beta=fit.pop('beta')
                score=Xv@beta+ranking_intercept(X,data.y[a],beta)
                return dict(candidate=c,params=param,inner=inner_id,train=a.tolist(),valid_indices=b.tolist(),
                    auc=float(roc_auc_score(data.y[b],score)),support=X.shape[1],score=score.tolist(),**fit)
            identity=dict(candidate=c,params=param,inner=inner_id,train=a.tolist(),valid_indices=b.tolist())
            rows['ridge_rank'].append(inner_record(path,identity,ridge_work))
    chosen={}
    for method,grid in candidate_grids.items():
        summary=[]
        for c,param in enumerate(grid):
            rr=[r for r in rows[method] if r['candidate']==c]
            valid=(sorted(r['inner'] for r in rr)==list(range(len(split['inner']))) and all(r['valid'] for r in rr))
            summary.append(dict(candidate=c,params=param,valid=valid,
                mean_auc=float(np.mean([r['auc'] for r in rr])) if valid else None,
                se_auc=float(np.std([r['auc'] for r in rr],ddof=1)/np.sqrt(len(rr))) if valid else None,
                mean_support=float(np.mean([r['support'] for r in rr])) if valid else None))
        atomic_json(root/(method+'_tuning.json'),dict(summary=summary,inner=rows[method],selection_rule=RULE))
        chosen[method]=choose(summary)
    return chosen


def hub(data,support,beta):
    d=np.bincount(np.r_[data.u[support],data.v[support]],minlength=len(data.roi_names)).astype(float)
    w=np.abs(beta[support]);strength=np.bincount(data.u[support],weights=w,minlength=len(d))+np.bincount(data.v[support],weights=w,minlength=len(d))
    result={}
    for key,values in [('degree',d),('strength',strength)]:
        share=values/values.sum() if values.sum() else None
        result[key+'_hhi']=float(share@share) if share is not None else None
        result[key+'_top5_share']=float(np.sort(share)[-5:].sum()) if share is not None else None
    return result


def outer(data,split,cfg,out):
    with threadpool_limits(limits=1):return outer_limited(data,split,cfg,Path(out))


def outer_limited(data,split,cfg,out):
    root=out/fold_dir(split);root.mkdir(parents=True,exist_ok=True)
    if all((root/(m+'.json')).exists() and (root/(m+'.npz')).exists() for m in METHODS):return
    selected=fit_inner(data,split,cfg,root)
    selectors,_=grids(cfg)
    tr,te=np.asarray(split['train']),np.asarray(split['test'])
    scaler=StandardScaler().fit(data.X[tr]);X,Xt=scaler.transform(data.X[tr]),scaler.transform(data.X[te])
    selector_cache={}
    for method in METHODS:
        check_stop(root)
        final=root/(method+'.json');arr=final.with_suffix('.npz')
        if final.exists() and arr.exists():continue
        start=time.perf_counter();choice=selected[method];param=choice['params']
        try:
            selector_beta=np.zeros(data.X.shape[1]);selector_z=np.zeros(len(data.roi_names));selector_support=np.array([],int)
            selector_info=None;selector_params=None
            if method!='ridge_rank':
                kind='logistic' if method=='v3_expanded' else 'rank'
                sid=param['selector'] if method=='v3_rank_refit' else choice['candidate']
                selector_params=selectors[sid]
                if (kind,sid) not in selector_cache:
                    problem=LogisticProblem(X,data.y[tr],data.H,True) if kind=='logistic' else RankingV3(X,data.y[tr],data.H)
                    selector_cache[(kind,sid)]=selector_fit(X,data.y[tr],data.H,selectors[sid],kind,problem,cfg)
                fit=selector_cache[(kind,sid)]
                selector_beta,selector_z=fit['beta'].copy(),fit['z'].copy()
                selector_info={k:v for k,v in fit.items() if k not in ('beta','z')}
                selector_support=np.flatnonzero(np.abs(selector_beta)>cfg['support_threshold'])
            if method in ('ridge_rank','v3_rank_refit'):
                subset=np.arange(data.X.shape[1]) if method=='ridge_rank' else selector_support
                fit=RidgeRanker(X[:,subset],data.y[tr]).fit(param['rho'],**cfg['ranking_refit'])
                beta=np.zeros(data.X.shape[1]);beta[subset]=fit.pop('beta')
                intercept=ranking_intercept(X,data.y[tr],beta)
                info=fit
                support=subset  # Ridge uses the whole input subset, like RFE.
            else:
                beta=selector_beta.copy();info=selector_info.copy()
                intercept=info['intercept'] if method=='v3_expanded' else ranking_intercept(X,data.y[tr],beta)
                support=selector_support
            info.pop('intercept',None)
            score=Xt@beta+intercept;pred=(score>=0).astype(int)
            probability=expit(score) if method=='v3_expanded' else np.array([])
            raw_beta=beta/scaler.scale_;raw_intercept=intercept-scaler.mean_@raw_beta
            incident=np.unique(np.r_[data.u[support],data.v[support]])
            atomic_npz(arr,beta=beta,intercept=np.array(intercept),raw_beta=raw_beta,raw_intercept=np.array(raw_intercept),
                scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,train_indices=tr,test_indices=te,
                y_true=data.y[te],score=score,predicted=pred,probability=probability,selected_edges=support,
                selected_roi_pairs=np.c_[data.u[support],data.v[support]],incident_rois=incident,
                selector_beta=selector_beta,selector_z=selector_z,selector_edges=selector_support)
            record=dict(method=method,repeat=split['repeat'],fold=split['fold'],train=tr.tolist(),test=te.tolist(),
                params=param,chosen_candidate=choice['candidate'],selection_rule=RULE,inner_mean_auc=choice['mean_auc'],
                metrics=prediction_metrics(data.y[te],score,pred),selected_edge_count=len(support),
                coefficient_nonzero_count=int(np.sum(np.abs(beta)>cfg['support_threshold'])),
                selector_edge_count=len(selector_support),selector_info=selector_info,selector_params=selector_params,
                intercept=intercept,hub_concentration=hub(data,support,beta) if method!='ridge_rank' else None,
                total_outer_seconds=time.perf_counter()-start,
                prediction_rule='logistic zero logit' if method=='v3_expanded' else 'training class-score midpoint; AUC unaffected by centering',
                probability_note='logistic probabilities' if method=='v3_expanded' else 'Ranking scores are not calibrated probabilities',**info)
            atomic_json(final,record)
            fail=root/(method+'_failed.json')
            if fail.exists():fail.unlink()
            print(f"repeat={split['repeat']:02d} fold={split['fold']} {method}: AUC={record['metrics']['auc']:.4f}, features={len(support)}",flush=True)
        except Exception as exc:
            atomic_json(root/(method+'_failed.json'),dict(method=method,choice=choice,error=f'{type(exc).__name__}: {exc}',
                runtime_seconds=time.perf_counter()-start,policy='No alternative candidate selected after outer failure'))
            raise


def analyze(out):
    out=Path(out);manifest=json.loads((out/'manifest.json').read_text())
    meta=dict(np.load(out/'dataset_metadata.npz',allow_pickle=False));expected=len(manifest['splits'])
    report=out/'analysis';report.mkdir(exist_ok=True)
    records=json.loads((out/'reference_records.json').read_text())
    ra=dict(np.load(out/'reference_predictions.npz',allow_pickle=False))
    grouped={};supports={};all_rows=[];hub_rows=[]
    for i,r in enumerate(records):
        grouped.setdefault(r['method'],[]).append(r);supports.setdefault(r['method'],[]).append(np.flatnonzero(ra['support'][i]))
        if r['hub_concentration'] is not None:
            hub_rows.append(dict(method=r['method'],repeat=r['repeat'],fold=r['fold'],**r['hub_concentration']))
    for method in METHODS:
        grouped[method]=[];supports[method]=[]
        for split in manifest['splits']:
            path=out/fold_dir(split)/(method+'.json')
            if not path.exists():continue
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
                r=json.loads(path.read_text());mm=prediction_metrics(a['y_true'],a['score'],a['predicted'])
                if any(abs(mm[k]-r['metrics'][k])>1e-10 for k in mm):raise ValueError('New metric mismatch')
                np.testing.assert_array_equal(a['test_indices'],split['test'])
                np.testing.assert_array_equal(a['train_indices'],split['train'])
                np.testing.assert_array_equal(a['y_true'],meta['y'][split['test']])
                if not r['converged'] or len(a['selected_edges'])!=r['selected_edge_count']:raise ValueError('Invalid outer commit')
                grouped[method].append(r);supports[method].append(a['selected_edges'].copy())
                if r['hub_concentration'] is not None:hub_rows.append(dict(method=method,repeat=r['repeat'],fold=r['fold'],**r['hub_concentration']))
    summary=dict(expected_folds_per_method=expected,methods={},
        complete=all(len(grouped[m])==expected for m in METHODS),
        interpretation='Exploratory development on previously inspected subjects. Fold SD is not a confidence interval. No independent-fold p-values or automatic superiority claim. Saved baselines were not retuned to this larger search budget.',
        probability_note='Ranking methods report raw scores and train-midpoint labels; classification thresholds differ from saved baselines. AUC is the primary endpoint.')
    stability_rows=[];per_repeat_rows=[]
    for method,rr in grouped.items():
        s=dict(completed_folds=len(rr),all_converged=all(r['converged'] for r in rr))
        if rr:
            for metric in rr[0]['metrics']:
                vals=[r['metrics'][metric] for r in rr];s[metric+'_mean']=float(np.mean(vals));s[metric+'_fold_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
            s['support_mean']=float(np.mean([r['selected_edge_count'] for r in rr]))
            hh=[r['hub_concentration'] for r in rr if r['hub_concentration'] is not None]
            s['hub_concentration_mean']={k:float(np.mean([h[k] for h in hh if h[k] is not None]))
                if any(h[k] is not None for h in hh) else None for k in hh[0]} if hh else None
            if method=='ridge_rank':s['stability']=None;s['stability_note']='All edges used; feature-selection stability would be trivially one and is not reported.'
            else:
                st=stability(supports[method],len(meta['u']));st.pop('pairs');s['stability']=st
                stability_rows.append(dict(method=method,**st))
                for repeat in sorted({r['repeat'] for r in rr}):
                    ss=[support for r,support in zip(rr,supports[method]) if r['repeat']==repeat]
                    st_r=stability(ss,len(meta['u']));st_r.pop('pairs')
                    per_repeat_rows.append(dict(method=method,repeat=repeat,folds=len(ss),**st_r))
            for r in rr:all_rows.append(dict(method=method,repeat=r['repeat'],fold=r['fold'],**r['metrics'],support=r['selected_edge_count']))
        summary['methods'][method]=s
    by={(r['method'],r['repeat'],r['fold']):r for rr in grouped.values() for r in rr}
    comparisons=[];paired_summaries=[]
    contrasts=[(m,'saved_max_auc_'+b) for m in METHODS for b in REFERENCE_METHODS]
    contrasts += [('v3_rank','v3_expanded'),('v3_rank_refit','v3_rank'),('v3_rank_refit','ridge_rank')]
    for method,baseline in contrasts:
        diffs=[]
        for r in grouped[method]:
            other=by.get((baseline,r['repeat'],r['fold']))
            if not other:continue
            delta=r['metrics']['auc']-other['metrics']['auc'];diffs.append(delta)
            comparisons.append(dict(method=method,comparator=baseline,repeat=r['repeat'],fold=r['fold'],
                auc_delta=delta,balanced_accuracy_delta=r['metrics']['balanced_accuracy']-other['metrics']['balanced_accuracy'],
                support_delta=r['selected_edge_count']-other['selected_edge_count']))
        if diffs:paired_summaries.append(dict(method=method,comparator=baseline,paired_folds=len(diffs),
            mean_auc_delta=float(np.mean(diffs)),delta_fold_sd=float(np.std(diffs,ddof=1)) if len(diffs)>1 else None,
            positive_delta_folds=int(np.sum(np.asarray(diffs)>0)),exact_ties=int(np.sum(np.asarray(diffs)==0))))
    summary['paired_auc_descriptive']=paired_summaries
    if len(grouped['v3_rank_refit'])==expected:
        new=summary['methods']['v3_rank_refit']['auc_mean']
        summary['hypothesis_check']=dict(greater_mean_auc_than_each_saved_baseline=all(new>summary['methods']['saved_max_auc_'+m]['auc_mean'] for m in REFERENCE_METHODS),
            note='This checks observed means only; it is not statistical significance or external validation.')
    csv_file(report/'fold_metrics.csv',all_rows);csv_file(report/'stability.csv',stability_rows)
    csv_file(report/'stability_per_repeat.csv',per_repeat_rows);csv_file(report/'hub_concentration.csv',hub_rows)
    csv_file(report/'paired_fold_comparisons.csv',comparisons);csv_file(report/'paired_auc_summary.csv',paired_summaries)
    atomic_json(report/'summary.json',summary)
    lines=['# Ranking V3 hypothesis test','',f"Complete: **{summary['complete']}**. Expected outer folds per new method: **{expected}**.",'',
        '| Method | Folds | AUC mean | AUC fold SD | Balanced accuracy | F1 | Mean features | Jaccard |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,s in summary['methods'].items():
        if not s['completed_folds']:continue
        jac=s['stability']['jaccard'] if s['stability'] else None
        lines.append(f"| {method} | {s['completed_folds']} | {s['auc_mean']:.4f} | {s['auc_fold_sd']} | {s['balanced_accuracy_mean']:.4f} | {s['f1_mean']:.4f} | {s['support_mean']:.2f} | {jac} |")
    lines+=['','New arms: expanded ordinary logistic V3; ranking V3; ranking V3 with ridge refit; ungated ridge ranking. Saved one-SE and max-AUC baselines are reference rows, not newly trained models.', '',
        'The ranking-vs-expanded-logistic contrast matches selector grids. The refit-vs-ranking contrast adds the jointly tuned refit strength. The ungated ranker checks whether gating adds value. These comparisons assess pipelines and tuning procedures, not isolated causal effects.', '',
        'Only label a run complete when each new method has all expected folds. New inner fits are essential; old tuning summaries cannot validate a changed loss or a new refit.', '',
        'AUC is primary. Ranking scores are not calibrated probabilities; labels use the training class-score midpoint. Larger input supports can inflate overlap. V3 selector capacities describe the selector, not the refitted coefficients.', '',
        'Fold SD and paired fold deltas are descriptive. The same subjects and outer results have already informed development. No independent-fold significance test is provided. Retune baselines with comparable search effort and validate on a new cohort before making a superiority claim.']
    (report/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    return summary


def run(data,reference_path,cfg,out,jobs=1,max_outer=None):
    out=Path(out).resolve();ref=Reference(reference_path)
    try:
        if out==ref.path:raise ValueError('New output must be separate from reference results')
        if not ref.z and (out==ref.path or out.is_relative_to(ref.path) or ref.path.is_relative_to(out)):
            raise ValueError('New output must be separate from reference results')
        grids(cfg)
        fingerprint,records,arrays=validate_reference(ref,data)
        with run_lock(out):
            init_manifest(out,dict(config=cfg,data=data.provenance,splits=ref.manifest['splits'],
                source_sha256=source_hash(),environment=versions(),reference_fingerprint=fingerprint,
                reference_signature=ref.manifest['signature'],
                preprocessed_X_sha256=hashlib.sha256(np.ascontiguousarray(data.X).tobytes()).hexdigest(),
                experiment='New ranking hypothesis; exploratory. Reference baselines are read only.'))
            atomic_json(out/'reference_records.json',records);atomic_npz(out/'reference_predictions.npz',**arrays)
            atomic_npz(out/'dataset_metadata.npz',u=data.u,v=data.v,y=data.y,roi_names=data.roi_names,subject_ids=data.subject_ids)
            splits=ref.manifest['splits'] if max_outer is None else ref.manifest['splits'][:max_outer]
            Parallel(n_jobs=jobs)(delayed(outer)(data,s,cfg,str(out)) for s in splits)
            return analyze(out)
    finally:ref.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',help='Completed max-inner-AUC re-selection ZIP or results directory')
    p.add_argument('--data');p.add_argument('--out',required=True)
    p.add_argument('--config',default=str(HERE/'configs/full.json'))
    p.add_argument('--atlas',default=str(HERE/'original/reference/ParcellationLausanne2008.xls'))
    p.add_argument('--labels',default=str(HERE/'original/reference/labels_index_CORTICAL_Laus2008_all_scales.mat'))
    p.add_argument('--jobs',type=int,default=1);p.add_argument('--max-outer',type=int)
    p.add_argument('--analyze-only',action='store_true')
    args=p.parse_args()
    if args.analyze_only:print(json.dumps(analyze(args.out),indent=2));return
    if not args.reference or not args.data:p.error('--reference and --data are required for training')
    if args.jobs<1 or (args.max_outer is not None and args.max_outer<1):p.error('jobs/max-outer must be positive')
    ref=Reference(args.reference);representation=ref.manifest['data']['representation'];ref.close()
    data=load_dataset(args.data,args.atlas,args.labels,representation)
    result=run(data,args.reference,json.loads(Path(args.config).read_text()),args.out,args.jobs,args.max_outer)
    print('Complete:',result['complete'])


if __name__=='__main__':main()
