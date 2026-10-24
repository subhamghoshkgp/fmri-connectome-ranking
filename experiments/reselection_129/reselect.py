"""Outer-only re-selection. The original package is imported without modification.

No call to make_splits(), run(), _outer(), or any inner-fitting routine is made.
Selection uses saved summary floats exactly: (-mean_auc, mean_support, index).
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'original'))
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from szv3.data import load_dataset, sha256_file
from szv3.experiment import penalties, choose_candidate
from szv3.baselines import fit_svm
from szv3.metrics import prediction_metrics, stability
from szv3.solver import LogisticProblem, intercept_only_solution
from szv3.storage import atomic_json, atomic_npz, init_manifest, environment, source_digest, run_lock

METHODS = ('v3', 'enet', 'lasso', 'rfe_svm')
RULE = 'max_inner_auc_exact_ties_support_then_index'
HUB_DEFINITION = {
    'primary': 'degree_hhi = sum_v (d_v / (2 * number_of_selected_edges))**2',
    'degree': 'Number of selected undirected edges incident on each ROI; saved support is authoritative.',
    'weighted': 'strength_hhi = sum_v (s_v / sum_u s_u)**2, s_v = sum of abs(standardized beta_e) on selected incident edges.',
    'top5': 'Sum of five largest ROI degree/strength shares; every edge contributes to both endpoint totals.',
    'empty': 'Undefined concentrations are JSON null, never zero or perfect concentration.',
    'interpretation': 'Larger HHI means greater concentration; degree HHI also depends on support size. Report together with support.',
    'provenance': 'Explicit new descriptive definition: no numerical hub-concentration formula was present in the supplied original code/document.',
}


class SavedResults:
    """Read ZIP or directory without extracting or modifying original results."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.archive:
            names = self.archive.namelist()
            candidates = [n for n in names if n.endswith('manifest.json') and '/repeat_' not in n]
            if len(candidates) != 1:
                raise ValueError('ZIP must contain exactly one results manifest')
            self.prefix = candidates[0][:-len('manifest.json')]
            if len(names) != len(set(names)):
                raise ValueError('ZIP contains duplicate paths')
        else:
            self.prefix = ''
        self.manifest = self.json('manifest.json')

    def bytes(self, relative):
        if self.archive:
            return self.archive.read(self.prefix + relative)
        return (self.path / relative).read_bytes()

    def json(self, relative):
        return json.loads(self.bytes(relative))

    def arrays(self, relative):
        with np.load(io.BytesIO(self.bytes(relative)), allow_pickle=False) as a:
            return {k: a[k].copy() for k in a.files}

    def close(self):
        if self.archive:
            self.archive.close()


def select_max_auc(rows):
    valid = [r for r in rows if r['valid']]
    if not valid:
        raise ValueError('No valid candidate; no refit or fallback is allowed')
    for row in valid:
        if not math.isfinite(row['mean_auc']) or not math.isfinite(row['mean_support']):
            raise ValueError('Valid candidates must have finite mean AUC and support')
    # No epsilon, rounding, or isclose: the professor requested EXACT ties.
    return min(valid, key=lambda r: (-r['mean_auc'], r['mean_support'], r['candidate']))


def fold_path(split):
    return f"repeat_{split['repeat']:02d}/fold_{split['fold']:02d}"


def validate_original(source):
    """Validate cached summaries/outer arrays; no model fitting occurs here."""
    m = source.manifest
    cfg = m['config']
    if cfg['selection_rule'] != 'one_se' or set(cfg['grids']) != set(METHODS):
        raise ValueError('Expected the completed four-method one-SE experiment')
    expected = cfg['repeats'] * cfg['outer_folds']
    if len(m['splits']) != expected:
        raise ValueError('Saved split count differs from config')
    meta = source.arrays('dataset_metadata.npz')
    n = len(meta['y'])
    seen = set()
    counts = {method: 0 for method in METHODS}
    rows = []
    digest = hashlib.sha256(source.bytes('manifest.json'))
    digest.update(source.bytes('dataset_metadata.npz'))
    digest.update(source.bytes('incidence.npz'))
    for split in m['splits']:
        key = (split['repeat'], split['fold'])
        if key in seen:
            raise ValueError('Duplicate saved outer split')
        seen.add(key)
        tr, te = split['train'], split['test']
        if len(set(tr)) != len(tr) or len(set(te)) != len(te) or set(tr) & set(te) or sorted(tr + te) != list(range(n)):
            raise ValueError('Invalid outer split')
        if len(split['inner']) != cfg['inner_folds']:
            raise ValueError('Wrong inner split count')
        for inner in split['inner']:
            a, b = inner['train'], inner['valid']
            if len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) & set(b) or sorted(a + b) != sorted(tr):
                raise ValueError('Invalid saved inner split')
        root = fold_path(split)
        for method in METHODS:
            files = [f'{root}/{method}{suffix}' for suffix in ('_tuning.json', '.json', '.npz')]
            for name in files:
                digest.update(name.encode())
                digest.update(source.bytes(name))
            tuning = source.json(files[0])
            old = source.json(files[1])
            arrays = source.arrays(files[2])
            if (old['method'], old['repeat'], old['fold']) != (method, *key) or old['train'] != tr or old['test'] != te:
                raise ValueError('Old result does not match saved split')
            for name, indices in [('train_indices', tr), ('test_indices', te)]:
                if not np.array_equal(arrays[name], indices):
                    raise ValueError('Old array indices differ from split')
            if not np.array_equal(arrays['y_true'], meta['y'][te]) or not old['converged']:
                raise ValueError('Invalid old labels/convergence')
            recomputed = prediction_metrics(arrays['y_true'], arrays['score'], arrays['predicted'])
            if any(abs(recomputed[k] - old['metrics'][k]) > 1e-10 for k in recomputed):
                raise ValueError('Stored original metrics differ from predictions')
            support = arrays['selected_edges']
            if len(support) != old['selected_edge_count'] or len(np.unique(support)) != len(support):
                raise ValueError('Stored support count mismatch')
            if np.any(support < 0) or np.any(support >= len(meta['u'])):
                raise ValueError('Support index out of range')
            if method != 'rfe_svm' and not np.array_equal(support, np.flatnonzero(np.abs(arrays['beta']) > cfg['support_threshold'])):
                raise ValueError('Original logistic support threshold mismatch')
            if not np.isfinite(arrays['beta']).all() or not np.isfinite(arrays['score']).all():
                raise ValueError('Nonfinite original result')
            summary = tuning['summary']
            if sorted(r['candidate'] for r in summary) != list(range(len(cfg['grids'][method]))):
                raise ValueError('Incomplete/duplicate tuning candidate indices')
            if len(tuning['inner']) != len(summary) * cfg['inner_folds']:
                raise ValueError('Wrong number of cached inner records')
            for row in summary:
                c = row['candidate']
                if row['params'] != cfg['grids'][method][c]:
                    raise ValueError('Tuning parameters differ from immutable grid')
                cached = [r for r in tuning['inner'] if r['candidate'] == c]
                if sorted(r['inner'] for r in cached) != list(range(cfg['inner_folds'])):
                    raise ValueError('Cached inner folds incomplete')
                for r in cached:
                    inner = split['inner'][r['inner']]
                    if r['params'] != row['params'] or r['train'] != inner['train'] or r['valid_indices'] != inner['valid']:
                        raise ValueError('Cached tuning record differs from saved grid/split')
                valid = all(r['valid'] for r in cached)
                if valid != row['valid']:
                    raise ValueError('Candidate validity differs from cached inner fits')
                if valid:
                    for name, values in [('mean_auc', [r['auc'] for r in cached]), ('mean_support', [r['support'] for r in cached])]:
                        if abs(float(np.mean(values)) - row[name]) > 1e-12:
                            raise ValueError('Cached tuning mean mismatch')
            original_choice = choose_candidate(summary, 'one_se')
            if original_choice['candidate'] != old['chosen_candidate'] or original_choice['params'] != old['params']:
                raise ValueError('Saved old choice does not match original one-SE rule')
            new = select_max_auc(summary)
            counts[method] += int(new['candidate'] != old['chosen_candidate'])
            rows.append(dict(method=method, repeat=split['repeat'], fold=split['fold'],
                old_candidate=old['chosen_candidate'], new_candidate=new['candidate'],
                changed=new['candidate'] != old['chosen_candidate'],
                old_mean_inner_auc=original_choice['mean_auc'], new_mean_inner_auc=new['mean_auc'],
                old_mean_inner_support=original_choice['mean_support'], new_mean_inner_support=new['mean_support'],
                old_params=json.dumps(old['params'], sort_keys=True), new_params=json.dumps(new['params'], sort_keys=True)))
    for repeat in range(cfg['repeats']):
        splits = [s for s in m['splits'] if s['repeat'] == repeat]
        if len(splits) != cfg['outer_folds'] or sorted(i for s in splits for i in s['test']) != list(range(n)):
            raise ValueError('Incomplete per-repeat test coverage')
    return dict(expected_folds_per_method=expected, original_complete=True,
                changed_candidates=counts, input_fingerprint=digest.hexdigest(), selection_rule=RULE), rows


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Write through a temporary sibling to survive disconnection.
    tmp = path.with_suffix(path.suffix + '.pending')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def concentration(support, beta, u, v, n_rois):
    support = np.asarray(support, dtype=int)
    degree = np.bincount(np.r_[u[support], v[support]], minlength=n_rois).astype(float)
    weights = np.abs(beta[support])
    strength = np.bincount(u[support], weights=weights, minlength=n_rois)
    strength += np.bincount(v[support], weights=weights, minlength=n_rois)
    result = dict(selected_edges=len(support), incident_rois=int(np.sum(degree > 0)))
    for prefix, values in [('degree', degree), ('strength', strength)]:
        shares = values / values.sum() if values.sum() else None
        result[prefix + '_hhi'] = float(shares @ shares) if shares is not None else None
        result[prefix + '_top5_share'] = float(np.sort(shares)[-5:].sum()) if shares is not None else None
        result[prefix + '_max_share'] = float(shares.max()) if shares is not None else None
        result[prefix + '_effective_rois'] = float(1 / (shares @ shares)) if shares is not None else None
    return result, degree, strength


def check_data(data, source):
    if data.provenance != source.manifest['data']:
        raise ValueError('Data/preprocessing provenance differs from original run')
    meta = source.arrays('dataset_metadata.npz')
    for name in ('y', 'u', 'v', 'subject_ids', 'roi_names'):
        if not np.array_equal(getattr(data, name), meta[name]):
            raise ValueError('Dataset ordering/atlas mismatch: ' + name)
    from scipy.sparse import load_npz
    original_H = load_npz(io.BytesIO(source.bytes('incidence.npz')))
    if original_H.shape != data.H.shape or (original_H != data.H).nnz:
        raise ValueError('Incidence mapping differs from original')


def refit_selected(data, split, cfg, method, param, old_arrays):
    """Same outer fitting code path/settings as the original _outer_limited."""
    tr, te = np.asarray(split['train']), np.asarray(split['test'])
    scaler = StandardScaler().fit(data.X[tr])
    # Verify unchanged training-only preprocessing before touching test data.
    if not np.allclose(scaler.mean_, old_arrays['scaler_mean'], rtol=0, atol=1e-12) or not np.allclose(scaler.scale_, old_arrays['scaler_scale'], rtol=0, atol=1e-12):
        raise ValueError('Reconstructed scaler differs from original training scaler')
    if method == 'rfe_svm':
        # Preserve original training-only probability calibration. These are its
        # original 3 calibration splits, NOT reruns of any hyperparameter inner CV.
        fit = fit_svm(data.X[tr], data.y[tr], data.X[te], param, split['inner_seed'], True)
        score, pred, prob = fit.pop('score'), fit.pop('predicted'), fit.pop('probability')
        support = fit.pop('support')
        mean, scale = fit.pop('scaler_mean'), fit.pop('scaler_scale')
    else:
        mean, scale = scaler.mean_, scaler.scale_
        X, Xt = scaler.transform(data.X[tr]), scaler.transform(data.X[te])
        le, lv, l2, lm = penalties(X, data.y[tr], data.H, method, param)
        fit = (intercept_only_solution(X, data.y[tr], data.H, le, lv, l2, method == 'v3')
               if param['ratio'] == 1 else
               LogisticProblem(X, data.y[tr], data.H, method == 'v3').solve(le, lv, l2, **cfg['solver']))
        fit['lambda_max'] = lm
        score = Xt @ fit['beta'] + fit['intercept']
        pred, prob = (score >= 0).astype(int), expit(score)
        support = np.flatnonzero(np.abs(fit['beta']) > cfg['support_threshold'])
    beta, z = fit.pop('beta'), fit.pop('z')
    if not fit['converged'] or not np.isfinite(beta).all() or not np.isfinite(score).all():
        raise ValueError('Refit failed finite/convergence checks')
    raw_beta = beta / scale
    incident = np.unique(np.r_[data.u[support], data.v[support]])
    arrays = dict(beta=beta, z=z, scaler_mean=mean, scaler_scale=scale,
        raw_beta=raw_beta, raw_intercept=np.asarray(fit['intercept'] - mean @ raw_beta),
        train_indices=tr, test_indices=te, y_true=data.y[te], score=score,
        predicted=pred, probability=prob, selected_edges=support,
        selected_roi_pairs=np.c_[data.u[support], data.v[support]], incident_rois=incident)
    record = dict(method=method, repeat=split['repeat'], fold=split['fold'],
        train=split['train'], test=split['test'], params=param,
        metrics=prediction_metrics(data.y[te], score, pred),
        selected_edge_count=len(support), selected_edge_fraction=len(support)/len(beta),
        incident_roi_count=len(incident), node_capacity_count=int(np.sum(z > cfg['support_threshold'])),
        threshold_sensitivity={str(t): int(np.sum(np.abs(beta) > t)) for t in [1e-5, 1e-4, 1e-3]}, **fit)
    return record, arrays


def task(data, source_path, split, out):
    source = SavedResults(source_path)
    try:
        with threadpool_limits(limits=1):
            for method in METHODS:
                start = time.perf_counter()
                root = fold_path(split)
                destination = Path(out) / root
                marker = destination / (method + '.json')
                arr_path = destination / (method + '.npz')
                if marker.exists():
                    if not arr_path.exists():
                        raise ValueError('Missing arrays for committed refit: ' + str(marker))
                    continue
                tuning = source.json(f'{root}/{method}_tuning.json')
                selected = select_max_auc(tuning['summary'])
                old = source.json(f'{root}/{method}.json')
                old_arrays = source.arrays(f'{root}/{method}.npz')
                try:
                    record, arrays = refit_selected(data, split, source.manifest['config'], method, selected['params'], old_arrays)
                    if method == 'v3':
                        for prefix, aa in [('old', old_arrays), ('new', arrays)]:
                            hub, degree, strength = concentration(aa['selected_edges'], aa['beta'], data.u, data.v, len(data.roi_names))
                            record[prefix + '_hub_concentration'] = hub
                            arrays[prefix + '_roi_degree'] = degree
                            arrays[prefix + '_roi_strength'] = strength
                    old_summary = next(r for r in tuning['summary'] if r['candidate'] == old['chosen_candidate'])
                    record.update(chosen_candidate=selected['candidate'], selection_rule=RULE,
                        candidate_changed=selected['candidate'] != old['chosen_candidate'],
                        old_choice=old_summary, new_choice=selected, old_result=old,
                        inner_fits_rerun=0, outer_refit_performed=True,
                        total_method_seconds=time.perf_counter() - start,
                        original_record_sha256=hashlib.sha256(source.bytes(f'{root}/{method}.json')).hexdigest(),
                        original_arrays_sha256=hashlib.sha256(source.bytes(f'{root}/{method}.npz')).hexdigest(),
                        tuning_sha256=hashlib.sha256(source.bytes(f'{root}/{method}_tuning.json')).hexdigest())
                    # Full old/new prediction, support and coefficient objects in ONE NPZ.
                    atomic_npz(arr_path, **arrays, **{'old_' + k: v for k, v in old_arrays.items()})
                    atomic_json(marker, record)  # JSON is the durable commit marker.
                    failure = destination / (method + '_failed.json')
                    if failure.exists():
                        failure.unlink()
                    print(f"repeat={split['repeat']:02d} fold={split['fold']} {method}: candidate {old['chosen_candidate']} -> {selected['candidate']}, AUC {old['metrics']['auc']:.3f} -> {record['metrics']['auc']:.3f}", flush=True)
                except Exception as exc:
                    atomic_json(destination / (method + '_failed.json'), dict(method=method,
                        repeat=split['repeat'], fold=split['fold'], selected=selected,
                        error=f'{type(exc).__name__}: {exc}', runtime_seconds=time.perf_counter() - start,
                        inner_fits_rerun=0, policy='Do not select a different candidate after an outer refit failure.'))
                    raise
    finally:
        source.close()


def stats(values):
    values = [float(x) for x in values if x is not None]
    return dict(mean=float(np.mean(values)) if values else None,
                fold_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                defined_folds=len(values))


def full_environment():
    """Record transitive numeric dependencies absent from the original manifest."""
    return {p: importlib.metadata.version(p) for p in
            ('numpy', 'scipy', 'scikit-learn', 'cvxpy', 'clarabel', 'scs',
             'h5py', 'xlrd', 'joblib', 'threadpoolctl')}


def compare(out):
    out = Path(out)
    m = json.loads((out / 'manifest.json').read_text())
    cfg = m['config']
    meta = dict(np.load(out / 'dataset_metadata.npz', allow_pickle=False))
    expected = cfg['repeats'] * cfg['outer_folds']
    summary = dict(selection_rule=RULE, expected_folds_per_method=expected, methods={},
        inner_fits_rerun=0, hub_definition=HUB_DEFINITION,
        interpretation='Descriptive paired comparisons on reused subjects. Fold SD is not a CI; no independent-fold p-values. The comparison is protocol sensitivity, not external confirmation.')
    fold_rows, metric_rows, repeat_rows, hub_rows, pairs_rows = [], [], [], [], []
    report = out / 'comparison'
    report.mkdir(exist_ok=True)
    for method in METHODS:
        saved = []
        for split in m['splits']:
            path = out / fold_path(split) / (method + '.json')
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as a:
                aa = {k: a[k].copy() for k in a.files}
            if r['chosen_candidate'] != r['new_choice']['candidate'] or r['selection_rule'] != RULE:
                raise ValueError('Re-selection commit rule mismatch')
            for prefix, record in [('', r), ('old_', r['old_result'])]:
                if not np.array_equal(aa[prefix + 'test_indices'], split['test']) or not np.array_equal(aa[prefix + 'train_indices'], split['train']):
                    raise ValueError('Comparison indices differ from saved split')
                if not np.array_equal(aa[prefix + 'y_true'], meta['y'][split['test']]):
                    raise ValueError('Comparison true labels differ from metadata')
                metrics = prediction_metrics(aa[prefix + 'y_true'], aa[prefix + 'score'], aa[prefix + 'predicted'])
                if any(abs(metrics[k] - record['metrics'][k]) > 1e-10 for k in metrics):
                    raise ValueError('Comparison metrics differ from saved predictions')
                if len(aa[prefix + 'selected_edges']) != record['selected_edge_count']:
                    raise ValueError('Comparison support count mismatch')
            if not r['converged'] or r['inner_fits_rerun'] != 0:
                raise ValueError('Invalid re-selection commit')
            saved.append((r, aa))
            row = dict(method=method, repeat=r['repeat'], fold=r['fold'],
                old_candidate=r['old_choice']['candidate'], new_candidate=r['chosen_candidate'],
                candidate_changed=r['candidate_changed'],
                old_mean_inner_auc=r['old_choice']['mean_auc'], new_mean_inner_auc=r['new_choice']['mean_auc'],
                new_refit_seconds=r['runtime_seconds'], new_total_seconds=r['total_method_seconds'],
                old_refit_seconds=r['old_result']['runtime_seconds'])
            for metric in ('auc', 'accuracy', 'balanced_accuracy', 'f1', 'sensitivity', 'specificity'):
                old, new = r['old_result']['metrics'][metric], r['metrics'][metric]
                row['old_' + metric], row['new_' + metric], row['delta_' + metric] = old, new, new - old
            row.update(old_support=r['old_result']['selected_edge_count'], new_support=r['selected_edge_count'],
                       delta_support=r['selected_edge_count'] - r['old_result']['selected_edge_count'])
            fold_rows.append(row)
        method_summary = dict(completed_folds=len(saved), expected_folds=expected,
            candidate_changes=sum(r['candidate_changed'] for r, a in saved),
            all_converged=all(r['converged'] for r, a in saved), metrics={})
        if not saved:
            summary['methods'][method] = method_summary
            continue
        method_summary['candidate_change_fraction'] = method_summary['candidate_changes'] / len(saved)
        for metric in ('auc', 'accuracy', 'balanced_accuracy', 'f1', 'sensitivity', 'specificity', 'support'):
            old = [r['old_result']['metrics'][metric] if metric != 'support' else r['old_result']['selected_edge_count'] for r, a in saved]
            new = [r['metrics'][metric] if metric != 'support' else r['selected_edge_count'] for r, a in saved]
            entry = dict(old=stats(old), new=stats(new), paired_delta=stats(np.asarray(new) - old))
            method_summary['metrics'][metric] = entry
            metric_rows.append(dict(method=method, metric=metric, folds=len(saved),
                old_mean=entry['old']['mean'], new_mean=entry['new']['mean'], delta_mean=entry['paired_delta']['mean'],
                old_fold_sd=entry['old']['fold_sd'], new_fold_sd=entry['new']['fold_sd'], paired_delta_fold_sd=entry['paired_delta']['fold_sd']))
        for label, prefix in [('old_one_se', 'old_'), ('new_max_inner_auc', '')]:
            supports = [a[prefix + 'selected_edges'] for r, a in saved]
            st = stability(supports, len(meta['u']))
            pairs_rows += [dict(method=method, selection=label, **pair) for pair in st.pop('pairs')]
            method_summary[label + '_stability'] = st
            frequency = np.zeros(len(meta['u']))
            beta_mean = np.mean([a[prefix + 'beta'] for r, a in saved], axis=0)
            abs_mean = np.mean([np.abs(a[prefix + 'beta']) for r, a in saved], axis=0)
            for support in supports:
                frequency[support] += 1 / len(saved)
            csv_write(report / f'{method}_{label}_edge_frequency.csv', [dict(edge_index=int(e),
                roi_u_id=int(meta['u'][e]) + 1, roi_v_id=int(meta['v'][e]) + 1,
                roi_u=str(meta['roi_names'][meta['u'][e]]), roi_v=str(meta['roi_names'][meta['v'][e]]),
                frequency=float(frequency[e]), mean_beta=float(beta_mean[e]), mean_abs_beta=float(abs_mean[e]))
                for e in np.argsort(-frequency, kind='stable')])
            for repeat in range(cfg['repeats']):
                rr = [(r, a) for r, a in saved if r['repeat'] == repeat]
                if not rr:
                    continue
                st_r = stability([a[prefix + 'selected_edges'] for r, a in rr], len(meta['u']))
                st_r.pop('pairs')
                repeat_row = dict(method=method, selection=label, repeat=repeat, folds=len(rr), **st_r)
                if len(rr) == cfg['outer_folds']:
                    indices = np.concatenate([a[prefix + 'test_indices'] for r, a in rr])
                    if sorted(indices.tolist()) != list(range(len(meta['y']))):
                        raise ValueError('Incomplete OOF coverage')
                    pooled = prediction_metrics(np.concatenate([a[prefix + 'y_true'] for r, a in rr]),
                        np.concatenate([a[prefix + 'score'] for r, a in rr]),
                        np.concatenate([a[prefix + 'predicted'] for r, a in rr]))
                else:
                    pooled = {k: None for k in ('auc', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1')}
                repeat_row.update({'pooled_oof_descriptive_' + k: v for k, v in pooled.items()})
                repeat_rows.append(repeat_row)
        method_summary['stability_delta'] = {k: (method_summary['new_max_inner_auc_stability'][k] - method_summary['old_one_se_stability'][k]
            if method_summary['new_max_inner_auc_stability'][k] is not None and method_summary['old_one_se_stability'][k] is not None else None)
            for k in ('jaccard', 'kuncheva_equal_size', 'adjusted_overlap_min', 'nogueira')}
        if method == 'v3':
            for r, a in saved:
                for label, prefix in [('old_one_se', 'old_'), ('new_max_inner_auc', '')]:
                    hub, degree, strength = concentration(a[prefix + 'selected_edges'], a[prefix + 'beta'], meta['u'], meta['v'], len(meta['roi_names']))
                    hub_rows.append(dict(selection=label, repeat=r['repeat'], fold=r['fold'], **hub))
            for label, prefix in [('old_one_se', 'old_'), ('new_max_inner_auc', '')]:
                degrees, strengths = [], []
                for r, a in saved:
                    _, degree, strength = concentration(a[prefix + 'selected_edges'], a[prefix + 'beta'], meta['u'], meta['v'], len(meta['roi_names']))
                    degrees.append(degree)
                    strengths.append(strength)
                degree, strength = np.mean(degrees, axis=0), np.mean(strengths, axis=0)
                csv_write(report / f'v3_{label}_roi_hubs.csv', [dict(roi_id=i+1,
                    roi=str(meta['roi_names'][i]), mean_selected_degree=float(degree[i]),
                    mean_abs_coefficient_strength=float(strength[i]))
                    for i in np.argsort(-strength, kind='stable')])
            method_summary['hub_concentration'] = {}
            for key in ('degree_hhi', 'strength_hhi', 'degree_top5_share', 'strength_top5_share', 'degree_effective_rois', 'strength_effective_rois'):
                old = [r['old_hub_concentration'][key] for r, a in saved]
                new = [r['new_hub_concentration'][key] for r, a in saved]
                method_summary['hub_concentration'][key] = dict(old=stats(old), new=stats(new),
                    paired_delta=stats([b-a for a, b in zip(old, new) if a is not None and b is not None]))
        summary['methods'][method] = method_summary
    summary['complete'] = all(v['completed_folds'] == expected for v in summary['methods'].values())
    summary['completed_outer_refits'] = sum(v['completed_folds'] for v in summary['methods'].values())
    csv_write(report / 'paired_fold_comparison.csv', fold_rows)
    csv_write(report / 'metric_comparison.csv', metric_rows)
    csv_write(report / 'per_repeat_stability_and_oof.csv', repeat_rows)
    csv_write(report / 'stability_pairs.csv', pairs_rows)
    csv_write(report / 'v3_hub_concentration.csv', hub_rows)
    atomic_json(report / 'summary.json', summary)
    lines = ['# Dataset-1: one-SE versus max-inner-AUC', '', f"Complete: **{summary['complete']}**. Outer refits: **{summary['completed_outer_refits']}/{4*expected}**. Inner tuning fits rerun: **0**.", '',
        'Fold means are descriptive. Paired deltas use the same saved outer splits. Repetitions share the same subjects; SD is not a confidence interval.', '',
        '| Method | Folds | Candidate changes | AUC old/new | Balanced accuracy old/new | F1 old/new | Sensitivity old/new | Specificity old/new | Support old/new |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, s in summary['methods'].items():
        if not s['completed_folds']:
            continue
        cells = [f"{s['metrics'][k]['old']['mean']:.4f} / {s['metrics'][k]['new']['mean']:.4f}" for k in ('auc', 'balanced_accuracy', 'f1', 'sensitivity', 'specificity', 'support')]
        lines.append(f"| {method} | {s['completed_folds']}/{expected} | {s['candidate_changes']} | " + ' | '.join(cells) + ' |')
    lines += ['', '| Method | Jaccard old/new | Nogueira old/new | Kuncheva old/new (equal sizes only) |', '|---|---:|---:|---:|']
    for method, s in summary['methods'].items():
        if s['completed_folds']:
            values = []
            for key in ('jaccard', 'nogueira', 'kuncheva_equal_size'):
                values.append(' / '.join('NA' if s[label + '_stability'][key] is None else f"{s[label + '_stability'][key]:.4f}" for label in ('old_one_se', 'new_max_inner_auc')))
            lines.append('| ' + method + ' | ' + ' | '.join(values) + ' |')
    v3 = summary['methods']['v3']
    if 'hub_concentration' in v3:
        lines += ['', '## V3 hub concentration', '', 'No numerical hub statistic was defined in the supplied original implementation; the following definitions are explicit additions to reporting.', '',
                  '`degree_HHI = sum_v (d_v / (2 * selected_edges))^2`, where d_v is selected-edge incidence. The weighted analogue uses sums of absolute standardized coefficients on selected incident edges. Larger values mean more concentration. Support size affects HHI. Top-5 shares count both endpoints; empty support is undefined.', '']
        for key, e in v3['hub_concentration'].items():
            lines.append(f"- {key}: old {e['old']['mean']}, new {e['new']['mean']}; paired delta {e['paired_delta']['mean']}.")
    lines += ['', 'Selection used only saved inner-CV summaries, without test scores. Every selected candidate was refitted, including unchanged choices. RFE-SVM retains its original training-only 3-fold probability calibration, which is distinct from hyperparameter tuning.', '',
        'Convergence/KKT, parameters, runtime, and old/new predictions/coefficients/support are saved per fold. A failed outer refit cannot trigger a different candidate choice.', '',
        'This comparison assesses sensitivity to the selection rule. Algorithm changes should be considered after reviewing these results; validation on a new cohort would still be needed.']
    (report / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    return summary


def run(source_path, data, out, jobs=1, max_outer=None):
    out = Path(out).resolve()
    source = SavedResults(source_path)
    try:
        if not source.archive and (out == source.path or out.is_relative_to(source.path) or source.path.is_relative_to(out)):
            raise ValueError('New output must be separate from original results')
        if source.archive and out == source.path:
            raise ValueError('New output must not be original ZIP')
        original = source.manifest
        if source_digest() != original['source_sha256']:
            raise ValueError('Original package source differs from completed run; preserve all original .py files')
        if environment() != original['environment']:
            raise ValueError('Python/package environment differs from completed run; use the supplied exact-version environment')
        audit, selections = validate_original(source)
        check_data(data, source)
        with run_lock(out):
            payload = dict(config=original['config'], data=original['data'], splits=original['splits'],
                selection_rule=RULE, parent_signature=original['signature'], input_fingerprint=audit['input_fingerprint'],
                original_source_sha256=source_digest(), environment=environment(),
                extended_environment=full_environment(),
                historical_solver_version_note='The original manifest did not record SCS/joblib/threadpoolctl versions. Solver algorithms/settings and all logged package versions are preserved; historical transitive versions cannot be independently verified.',
                reselect_source_sha256=sha256_file(__file__), inner_fits_rerun=0, hub_definition=HUB_DEFINITION,
                preprocessed_X_sha256=hashlib.sha256(np.ascontiguousarray(data.X).tobytes()).hexdigest())
            init_manifest(out, payload)
            atomic_json(out / 'selection_audit.json', audit)
            csv_write(out / 'candidate_choices.csv', selections)
            atomic_npz(out / 'dataset_metadata.npz', **source.arrays('dataset_metadata.npz'))
            (out / 'original_manifest.json').write_bytes(source.bytes('manifest.json'))
            splits = original['splits'] if max_outer is None else original['splits'][:max_outer]
            Parallel(n_jobs=jobs)(delayed(task)(data, str(source.path), s, str(out)) for s in splits)
            return compare(out)
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True, help='Completed one-SE results directory or results ZIP')
    parser.add_argument('--out', required=True, help='NEW output directory')
    parser.add_argument('--data', help='The same original Dataset-1 MAT file')
    parser.add_argument('--atlas', default=str(HERE / 'original/reference/ParcellationLausanne2008.xls'))
    parser.add_argument('--labels', default=str(HERE / 'original/reference/labels_index_CORTICAL_Laus2008_all_scales.mat'))
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--max-outer', type=int)
    parser.add_argument('--audit-only', action='store_true', help='Selection audit only; no data loading/refits')
    parser.add_argument('--compare-only', action='store_true', help='Rebuild reports from committed refits')
    args = parser.parse_args()
    if args.compare_only:
        print(json.dumps(compare(args.out), indent=2))
        return
    if args.jobs < 1 or (args.max_outer is not None and args.max_outer < 1):
        parser.error('jobs and max-outer must be positive')
    source = SavedResults(args.original)
    if args.audit_only:
        out = Path(args.out).resolve()
        if not source.archive and (out == source.path or out.is_relative_to(source.path) or source.path.is_relative_to(out)):
            raise ValueError('Audit output must be separate from original results')
        audit, rows = validate_original(source)
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'selection_audit.json', audit)
        csv_write(out / 'candidate_choices.csv', rows)
        print(json.dumps(audit, indent=2))
        source.close()
        return
    if not args.data:
        parser.error('--data is required for outer refits')
    data = load_dataset(args.data, args.atlas, args.labels, source.manifest['data']['representation'])
    source.close()
    result = run(args.original, data, args.out, args.jobs, args.max_outer)
    print('Complete:', result['complete'], '| Outer refits:', result['completed_outer_refits'])


if __name__ == '__main__':
    main()
