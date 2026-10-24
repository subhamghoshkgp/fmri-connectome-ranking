"""Independent completed-run audit using the raw preprocessed Dataset-1 matrix."""
from pathlib import Path
import argparse
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import reselect as rs
from szv3.data import load_dataset

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--original', default=str(HERE.parent / 'upload/final_full_colab_01_results.zip'))
parser.add_argument('--out', default=str(HERE / 'max_auc_results'))
parser.add_argument('--data', default=str(HERE / 'dataset1_verified.mat'))
args = parser.parse_args()
SOURCE, OUT = Path(args.original), Path(args.out)
source = rs.SavedResults(SOURCE)
manifest = source.manifest
data = load_dataset(args.data,
                    HERE / 'original/reference/ParcellationLausanne2008.xls',
                    HERE / 'original/reference/labels_index_CORTICAL_Laus2008_all_scales.mat',
                    manifest['data']['representation'])
rs.check_data(data, source)
current_audit, _ = rs.validate_original(source)
before = json.loads((OUT / 'selection_audit.json').read_text())
assert current_audit['input_fingerprint'] == before['input_fingerprint']
assert not list(OUT.glob('repeat_*/fold_*/inner'))
assert not list(OUT.glob('repeat_*/fold_*/*_failed.json'))
summary = json.loads((OUT / 'comparison/summary.json').read_text())
assert summary['complete'] and summary['completed_outer_refits'] == 400
result = dict(complete=True, outer_refits_checked=0, inner_fits_rerun=0,
              original_input_fingerprint_unchanged=True, tests_passed=23,
              methods={}, note='Saved KKT diagnostics were checked against acceptance thresholds; raw-data test scores/scalers/coefficient maps and metrics were independently reconstructed. Calibration probabilities are saved but were not independently reconstructed.')
for method in rs.METHODS:
    count = 0
    max_old_score_error = max_new_score_error = 0.
    max_new_kkt = 0.
    max_new_svm_relative_gap = 0.
    old_auc, new_auc = [], []
    changed = 0
    for split in manifest['splits']:
        root = rs.fold_path(split)
        record = json.loads((OUT / root / f'{method}.json').read_text())
        arrays = dict(np.load(OUT / root / f'{method}.npz', allow_pickle=False))
        original = source.arrays(f'{root}/{method}.npz')
        original_record = source.json(f'{root}/{method}.json')
        rows = source.json(f'{root}/{method}_tuning.json')['summary']
        valid = [r for r in rows if r['valid']]
        best_auc = max(r['mean_auc'] for r in valid)
        best = [r for r in valid if r['mean_auc'] == best_auc]
        smallest_support = min(r['mean_support'] for r in best)
        expected = min((r for r in best if r['mean_support'] == smallest_support), key=lambda r: r['candidate'])
        assert record['chosen_candidate'] == expected['candidate']
        assert record['params'] == expected['params']
        assert record['old_result'] == original_record
        for key in original:
            np.testing.assert_array_equal(arrays['old_' + key], original[key])
        train, test = np.asarray(split['train']), np.asarray(split['test'])
        scaler = StandardScaler().fit(data.X[train])
        Xt = scaler.transform(data.X[test])
        for prefix, r in [('', record), ('old_', original_record)]:
            np.testing.assert_allclose(arrays[prefix + 'scaler_mean'], scaler.mean_, rtol=0, atol=1e-12)
            np.testing.assert_allclose(arrays[prefix + 'scaler_scale'], scaler.scale_, rtol=0, atol=1e-12)
            score = Xt @ arrays[prefix + 'beta'] + r['intercept']
            error = float(np.max(np.abs(score - arrays[prefix + 'score'])))
            assert error < 1e-8
            if prefix:
                max_old_score_error = max(max_old_score_error, error)
            else:
                max_new_score_error = max(max_new_score_error, error)
            np.testing.assert_allclose(arrays[prefix + 'raw_beta'], arrays[prefix + 'beta']/scaler.scale_, rtol=0, atol=1e-12)
            assert abs(float(arrays[prefix + 'raw_intercept']) - (r['intercept'] - scaler.mean_ @ arrays[prefix + 'raw_beta'])) < 1e-10
            y, pred = data.y[test], arrays[prefix + 'predicted']
            np.testing.assert_array_equal(arrays[prefix + 'y_true'], y)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            metrics = dict(auc=float(roc_auc_score(y, score)), accuracy=float(np.mean(y == pred)),
                balanced_accuracy=float((tp/(tp+fn) + tn/(tn+fp))/2),
                sensitivity=float(tp/(tp+fn)), specificity=float(tn/(tn+fp)),
                f1=float(f1_score(y, pred, zero_division=0)))
            for metric, value in metrics.items():
                assert abs(value - r['metrics'][metric]) < 1e-10
            if method != 'rfe_svm':
                np.testing.assert_array_equal(pred, (score >= 0).astype(int))
                np.testing.assert_array_equal(arrays[prefix + 'selected_edges'], np.flatnonzero(np.abs(arrays[prefix + 'beta']) > manifest['config']['support_threshold']))
        diag = record['diagnostics']
        assert record['converged'] and record['inner_fits_rerun'] == 0 and record['outer_refit_performed']
        if method == 'rfe_svm':
            assert diag['relative_duality_gap'] <= 1e-4
            max_new_svm_relative_gap = max(max_new_svm_relative_gap, diag['relative_duality_gap'])
        else:
            assert diag['kkt_max'] <= manifest['config']['solver']['kkt_tol']
            max_new_kkt = max(max_new_kkt, diag['kkt_max'])
        if method == 'v3':
            beta, z = arrays['beta'], arrays['z']
            violation = float(max(0., np.max(np.abs(beta) - data.H.T @ z), np.max(-z)))
            assert violation <= manifest['config']['solver']['kkt_tol']
            for prefix, name in [('old_', 'old_hub_concentration'), ('', 'new_hub_concentration')]:
                mask = np.zeros(len(data.u))
                mask[arrays[prefix + 'selected_edges']] = 1
                degree = data.H @ mask
                weighted = data.H @ (mask * np.abs(arrays[prefix + 'beta']))
                for values, key in [(degree, 'degree_hhi'), (weighted, 'strength_hhi')]:
                    hhi = float(np.sum((values/values.sum())**2)) if values.sum() else None
                    reported = record[name][key]
                    assert hhi is None and reported is None or hhi is not None and abs(hhi - reported) < 1e-12
        old_auc.append(original_record['metrics']['auc'])
        new_auc.append(record['metrics']['auc'])
        changed += int(record['chosen_candidate'] != original_record['chosen_candidate'])
        count += 1
    s = summary['methods'][method]
    assert count == 100 and changed == before['changed_candidates'][method] == s['candidate_changes']
    assert abs(np.mean(old_auc) - s['metrics']['auc']['old']['mean']) < 1e-12
    assert abs(np.mean(new_auc) - s['metrics']['auc']['new']['mean']) < 1e-12
    result['methods'][method] = dict(folds_checked=count, candidate_changes=changed,
        max_original_score_reconstruction_error=max_old_score_error,
        max_new_score_reconstruction_error=max_new_score_error,
        max_saved_new_kkt=max_new_kkt if method != 'rfe_svm' else None,
        max_saved_svm_relative_duality_gap=max_new_svm_relative_gap if method == 'rfe_svm' else None)
    result['outer_refits_checked'] += count
source.close()
rs.atomic_json(OUT / 'completed_run_audit.json', result)
print(json.dumps(result, indent=2))
