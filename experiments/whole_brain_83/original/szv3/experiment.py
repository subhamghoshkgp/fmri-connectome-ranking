"""Nested CV with immutable manifests and per-fit checkpoint commits."""
from pathlib import Path
import json
import time
import numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed
from .math import lambda_max
from .solver import LogisticProblem,intercept_only_solution
from .metrics import prediction_metrics
from .baselines import fit_svm
from .storage import (atomic_json,atomic_npz,init_manifest,source_digest,
                      environment,run_lock)


def make_splits(y, config):
    rows=[]
    for repeat in range(config['repeats']):
        outer_seed=config['seed']+repeat
        outer=StratifiedKFold(config['outer_folds'],shuffle=True,random_state=outer_seed)
        for fold,(train,test) in enumerate(outer.split(np.zeros(len(y)),y)):
            inner_seed=config['seed']+10000+repeat*config['outer_folds']+fold
            inner=StratifiedKFold(config['inner_folds'],shuffle=True,random_state=inner_seed)
            rows.append(dict(repeat=repeat,fold=fold,outer_seed=outer_seed,
                inner_seed=inner_seed,train=train.tolist(),test=test.tolist(),
                inner=[dict(train=train[a].tolist(),valid=train[b].tolist())
                       for a,b in inner.split(np.zeros(len(train)),y[train])]))
    return rows


def penalties(X,y,H,method,param):
    if method=='v3':
        lm=lambda_max(X,y,H,param['alpha']); lam=lm*param['ratio']
        return param['alpha']*lam,(1-param['alpha'])*lam,param['ridge'],lm
    a=1. if method=='lasso' else param['alpha']
    lm=float(np.max(np.abs(X.T@(y.mean()-y)/len(y))))/a
    lam=lm*param['ratio']
    return a*lam,0.,(1-a)*lam,lm


def choose_candidate(rows, rule):
    valid=[r for r in rows if r['valid']]
    if not valid: raise RuntimeError('No fully converged candidate across all inner folds')
    best=max(valid,key=lambda r:r['mean_auc'])
    allowance=best['se_auc'] if rule=='one_se' else 1e-12
    # One-SE is a tuning heuristic, not a statistical equivalence claim.
    acceptable=[r for r in valid if r['mean_auc']>=best['mean_auc']-allowance]
    return min(acceptable,key=lambda r:(r['mean_support'],-r['mean_auc'],r['candidate']))


def _outer(data, split, cfg, out):
    with threadpool_limits(limits=1):
        return _outer_limited(data,split,cfg,Path(out))


def _outer_limited(data,split,cfg,out):
    tr=np.array(split['train']); te=np.array(split['test'])
    root=out/f"repeat_{split['repeat']:02d}"/f"fold_{split['fold']:02d}"
    root.mkdir(parents=True,exist_ok=True)
    for method,grid in cfg['grids'].items():
        final=root/(method+'.json'); arr=root/(method+'.npz')
        if final.exists() and arr.exists(): continue
        start=time.perf_counter(); inner_rows=[]
        for inner_id,inner in enumerate(split['inner']):
            a=np.array(inner['train']); b=np.array(inner['valid'])
            scaler=StandardScaler().fit(data.X[a])
            X=scaler.transform(data.X[a]); Xv=scaler.transform(data.X[b])
            problem=None if method=='rfe_svm' else LogisticProblem(X,data.y[a],data.H,method=='v3')
            for candidate,param in enumerate(grid):
                path=root/'inner'/method/f'{inner_id:02d}_{candidate:03d}.json'
                if path.exists():
                    row=json.loads(path.read_text()); inner_rows.append(row); continue
                row=dict(inner=inner_id,candidate=candidate,params=param,
                         train=a.tolist(),valid_indices=b.tolist())
                try:
                    if method=='rfe_svm':
                        fit=fit_svm(data.X[a],data.y[a],data.X[b],param,split['inner_seed'])
                        score=fit['score']; support=len(fit['support'])
                    else:
                        le,lv,l2,lm=penalties(X,data.y[a],data.H,method,param)
                        fit=(intercept_only_solution(X,data.y[a],data.H,le,lv,l2,method=='v3')
                             if param['ratio']==1 else problem.solve(le,lv,l2,**cfg['solver']))
                        score=Xv@fit['beta']+fit['intercept']
                        support=int(np.sum(np.abs(fit['beta'])>cfg['support_threshold']))
                        row['lambda_max']=lm
                        row['lambda_E'],row['lambda_V'],row['lambda_2']=le,lv,l2
                    from sklearn.metrics import roc_auc_score
                    row.update(valid=True,auc=float(roc_auc_score(data.y[b],score)),
                               support=support,diagnostics=fit.get('diagnostics'),
                               status=fit['status'],n_iter=fit['n_iter'],
                               runtime_seconds=fit['runtime_seconds'],attempts=fit.get('attempts'))
                except Exception as e:
                    # Invalid candidates remain visible and cannot win tuning.
                    row.update(valid=False,error=f'{type(e).__name__}: {e}')
                atomic_json(path,row); inner_rows.append(row)
        summary=[]
        for candidate,param in enumerate(grid):
            rr=[r for r in inner_rows if r['candidate']==candidate]
            valid=len(rr)==cfg['inner_folds'] and all(r['valid'] for r in rr)
            summary.append(dict(candidate=candidate,params=param,valid=valid,
                mean_auc=float(np.mean([r['auc'] for r in rr])) if valid else None,
                se_auc=float(np.std([r['auc'] for r in rr],ddof=1)/np.sqrt(len(rr))) if valid else None,
                mean_support=float(np.mean([r['support'] for r in rr])) if valid else None))
        atomic_json(root/(method+'_tuning.json'),dict(summary=summary,inner=inner_rows))
        chosen=choose_candidate(summary,cfg['selection_rule']); param=chosen['params']
        if method=='rfe_svm':
            fit=fit_svm(data.X[tr],data.y[tr],data.X[te],param,split['inner_seed'],True)
            score,pred,prob=fit.pop('score'),fit.pop('predicted'),fit.pop('probability')
            support=fit.pop('support'); mean=fit.pop('scaler_mean'); scale=fit.pop('scaler_scale')
        else:
            scaler=StandardScaler().fit(data.X[tr]); mean,scale=scaler.mean_,scaler.scale_
            X=scaler.transform(data.X[tr]); Xt=scaler.transform(data.X[te])
            le,lv,l2,lm=penalties(X,data.y[tr],data.H,method,param)
            fit=(intercept_only_solution(X,data.y[tr],data.H,le,lv,l2,method=='v3')
                 if param['ratio']==1 else LogisticProblem(X,data.y[tr],data.H,method=='v3').solve(le,lv,l2,**cfg['solver']))
            fit['lambda_max']=lm
            score=Xt@fit['beta']+fit['intercept']; pred=(score>=0).astype(int); prob=expit(score)
            support=np.flatnonzero(np.abs(fit['beta'])>cfg['support_threshold'])
        beta,z=fit.pop('beta'),fit.pop('z')
        incident=np.unique(np.r_[data.u[support],data.v[support]])
        raw_beta=beta/scale; raw_intercept=fit['intercept']-mean@raw_beta
        atomic_npz(arr,beta=beta,z=z,scaler_mean=mean,scaler_scale=scale,
            raw_beta=raw_beta,raw_intercept=np.array(raw_intercept),
            train_indices=tr,test_indices=te,y_true=data.y[te],score=score,
            predicted=pred,probability=prob,selected_edges=support,
            selected_roi_pairs=np.c_[data.u[support],data.v[support]],incident_rois=incident)
        record=dict(method=method,repeat=split['repeat'],fold=split['fold'],
            train=tr.tolist(),test=te.tolist(),params=param,chosen_candidate=chosen['candidate'],
            selection_rule=cfg['selection_rule'],metrics=prediction_metrics(data.y[te],score,pred),
            selected_edge_count=len(support),selected_edge_fraction=len(support)/len(beta),
            incident_roi_count=len(incident),node_capacity_count=int(np.sum(z>cfg['support_threshold'])),
            threshold_sensitivity={str(t):int(np.sum(np.abs(beta)>t)) for t in [1e-5,1e-4,1e-3]},
            total_method_seconds=time.perf_counter()-start,**fit)
        # JSON is the commit marker, written only after its NPZ is durable.
        atomic_json(final,record)
        print(f"repeat={split['repeat']} fold={split['fold']} {method}: "
              f"AUC={record['metrics']['auc']:.3f} support={len(support)} accepted",flush=True)
    return True


def run(data,cfg,out,jobs=1,max_outer=None):
    with run_lock(out):
        splits=make_splits(data.y,cfg)
        init_manifest(out,dict(config=cfg,data=data.provenance,source_sha256=source_digest(),
                               environment=environment(),splits=splits))
        out=Path(out)
        atomic_npz(out/'dataset_metadata.npz',u=data.u,v=data.v,y=data.y,
                   subject_ids=data.subject_ids,roi_names=data.roi_names)
        from scipy.sparse import save_npz
        save_npz(out/'incidence.npz',data.H)
        chosen=splits if max_outer is None else splits[:max_outer]
        Parallel(n_jobs=jobs)(delayed(_outer)(data,s,cfg,str(out)) for s in chosen)
        from .report import analyze
        return analyze(out)
