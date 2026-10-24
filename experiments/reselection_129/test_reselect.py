import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import reselect as rs
from szv3.data import synthetic_dataset
from szv3.experiment import run as original_run
import szv3.experiment as original_experiment


def candidate(i, auc, support, valid=True):
    return dict(candidate=i, mean_auc=auc, mean_support=support, valid=valid, params={})


def test_exact_selection_and_invalid_candidates():
    assert rs.select_max_auc([candidate(0, .8, 100), candidate(1, .8, 4)])['candidate'] == 1
    assert rs.select_max_auc([candidate(3, .8, 4), candidate(1, .8, 4)])['candidate'] == 1
    # A difference smaller than the old selector's tolerance is NOT a tie.
    assert rs.select_max_auc([candidate(0, .8, 1), candidate(1, .8 + 1e-14, 99)])['candidate'] == 1
    assert rs.select_max_auc([candidate(0, .99, 1, False), candidate(1, .7, 4)])['candidate'] == 1
    with pytest.raises(ValueError, match='No valid candidate'):
        rs.select_max_auc([candidate(0, .9, 1, False)])
    with pytest.raises(ValueError, match='finite'):
        rs.select_max_auc([candidate(0, float('nan'), 1)])


def test_hub_definitions_star_matching_and_empty():
    u = np.r_[np.zeros(5, int), np.arange(0, 10, 2)]
    v = np.r_[np.arange(1, 6), np.arange(1, 10, 2)]
    beta = np.ones(10)
    star, degree, strength = rs.concentration(np.arange(5), beta, u, v, 10)
    matching, _, _ = rs.concentration(np.arange(5, 10), beta, u, v, 10)
    assert star['degree_hhi'] == pytest.approx(.3)
    assert matching['degree_hhi'] == pytest.approx(.1)
    assert star['degree_hhi'] == star['strength_hhi']
    assert degree.sum() == 10 and strength.sum() == 10
    empty, _, _ = rs.concentration([], beta, u, v, 10)
    assert empty['degree_hhi'] is None and empty['strength_top5_share'] is None


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def original(tmp_path):
    cfg = json.loads((rs.HERE / 'original/configs/smoke.json').read_text())
    cfg.update(repeats=1, outer_folds=2, inner_folds=2)
    cfg['grids'] = {m: rows[:2] for m, rows in cfg['grids'].items()}
    data = synthetic_dataset(n=60)
    root = tmp_path / 'one_se'
    original_run(data, cfg, root, jobs=1)
    return data, root


def test_outer_only_resume_and_original_protection(original, tmp_path, monkeypatch):
    data, source_path = original
    old_snapshot = snapshot(source_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('No split regeneration or inner fitting is permitted')
    for name in ('make_splits', '_outer', '_outer_limited', 'run'):
        monkeypatch.setattr(original_experiment, name, forbidden)
    calls = []
    actual_refit = rs.refit_selected
    def counted(data, split, cfg, method, params, arrays):
        calls.append((split['repeat'], split['fold'], method))
        return actual_refit(data, split, cfg, method, params, arrays)
    monkeypatch.setattr(rs, 'refit_selected', counted)
    out = tmp_path / 'max_auc'
    result = rs.run(source_path, data, out)
    assert result['complete'] and result['completed_outer_refits'] == 8
    assert len(calls) == 8 and len(set(calls)) == 8
    assert result['inner_fits_rerun'] == 0
    assert snapshot(source_path) == old_snapshot
    for method in rs.METHODS:
        assert result['methods'][method]['completed_folds'] == 2
    src = rs.SavedResults(source_path)
    for split in src.manifest['splits']:
        for method in rs.METHODS:
            root = rs.fold_path(split)
            selected = rs.select_max_auc(src.json(f'{root}/{method}_tuning.json')['summary'])
            r = json.loads((out / root / f'{method}.json').read_text())
            assert r['chosen_candidate'] == selected['candidate']
            assert r['params'] == selected['params']
            with np.load(out / root / f'{method}.npz', allow_pickle=False) as a:
                old = src.arrays(f'{root}/{method}.npz')
                for key in old:
                    np.testing.assert_array_equal(a['old_' + key], old[key])
    rs.run(source_path, data, out)
    assert len(calls) == 8  # No already committed outer fit rerun.
    assert snapshot(source_path) == old_snapshot
    with pytest.raises(ValueError, match='separate'):
        rs.run(source_path, data, source_path / 'bad')
    assert not (source_path / 'bad').exists()


def test_failed_refit_never_falls_back(original, tmp_path, monkeypatch):
    data, source_path = original
    called = []
    def fail(data, split, cfg, method, params, arrays):
        called.append((method, params))
        raise RuntimeError('intentional convergence failure')
    monkeypatch.setattr(rs, 'refit_selected', fail)
    out = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='intentional'):
        rs.run(source_path, data, out)
    assert len(called) == 1
    failed = list(out.glob('repeat_*/fold_*/*_failed.json'))
    assert len(failed) == 1
    assert not list(out.glob('repeat_*/fold_*/v3.json'))
    r = json.loads(failed[0].read_text())
    assert r['inner_fits_rerun'] == 0 and 'different candidate' in r['policy']


def test_resume_rejects_changed_data(original, tmp_path):
    data, source_path = original
    changed = copy.deepcopy(data)
    changed.provenance['seed'] += 1
    with pytest.raises(ValueError, match='provenance'):
        rs.run(source_path, changed, tmp_path / 'bad_data')
