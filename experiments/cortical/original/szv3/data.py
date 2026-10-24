"""Strict loaders. MATLAB IDs remain authoritative; no cortical-only filtering."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from scipy.io import loadmat
from .math import edge_map


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    roi_names: np.ndarray
    u: np.ndarray
    v: np.ndarray
    H: object
    provenance: dict


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atlas_labels(xls_path, cortical_path):
    import xlrd
    sheet = xlrd.open_workbook(str(xls_path)).sheet_by_name('LABELS')
    labels = {}
    for row in range(2, sheet.nrows):
        for col, prefix in [(6, 'rh_'), (16, 'lh_')]:
            idx, label = sheet.cell_value(row, col), sheet.cell_value(row, col+1)
            if isinstance(idx, (float, int)) and idx > 0 and label:
                idx = int(idx)
                if idx in labels:
                    raise ValueError(f'Duplicate atlas ID {idx}')
                # The atlas places brainstem in its LEFT column; it is midline.
                name = str(label).strip()
                labels[idx] = name if name.lower() == 'brainstem' else prefix + name
    if set(labels) != set(range(1, 130)):
        raise ValueError('Expected exactly atlas IDs 1..129 at scale60')
    cortical = loadmat(cortical_path, simplify_cells=True)
    for idx, name in zip(cortical['ixc'][1], cortical['llist'][1]):
        if labels[int(idx)] != str(name).strip():
            raise ValueError(f'Cortical label mismatch at MATLAB ID {idx}')
    return np.array([labels[i] for i in range(1, 130)])


def canonical_matrices(a, n_rois=129):
    a = np.asarray(a, dtype=float)
    if a.ndim != 3:
        raise ValueError(f'Expected a 3D FC array; got {a.shape}')
    axes = [i for i, dim in enumerate(a.shape) if dim != n_rois]
    if len(axes) != 1:
        raise ValueError(f'Ambiguous subject axis: {a.shape}')
    a = np.moveaxis(a, axes[0], 0)
    if a.shape[1:] != (n_rois, n_rois):
        raise ValueError(f'Wrong FC scale: {a.shape}')
    if not np.isfinite(a).all():
        raise ValueError('FC has NaN or Inf: explicit investigation required')
    if np.max(np.abs(a - a.transpose(0, 2, 1))) > 1e-7:
        raise ValueError('FC matrices are not symmetric within 1e-7')
    if np.max(np.abs(np.diagonal(a, axis1=1, axis2=2)-1)) > 1e-6:
        raise ValueError('FC diagonal is not 1; verify correlation representation')
    if np.max(np.abs(a)) > 1+1e-8:
        raise ValueError('FC contains values outside [-1,1]')
    return a


def _hdf5_arrays(path):
    import h5py
    with h5py.File(path, 'r') as f:
        # Compact files from export_fc129.m may also be saved in v7.3 format.
        if all(k in f for k in ['FC_ctrl_129', 'FC_schz_129']):
            return [np.array(f[k]) for k in ['FC_ctrl_129', 'FC_schz_129']]
        base = f['SC_FC_Connectomes/FC_correlation']
        result = []
        for group in ['ctrl', 'schz']:
            node = base[group]
            if h5py.check_dtype(ref=node.dtype) is None:
                raise ValueError('Expected scale references in FC_correlation/'+group)
            matches = []
            for ref in np.array(node).ravel():
                a = np.array(f[ref])
                if a.ndim == 3 and sum(d == 129 for d in a.shape) == 2:
                    matches.append(a)
            if len(matches) != 1:
                raise ValueError(f'Expected one 129-ROI scale for {group}; got {len(matches)}')
            result.append(matches[0])
        return result


def load_dataset(path, xls_path, cortical_path, representation):
    import h5py
    if representation not in ['raw', 'fisher']:
        raise ValueError('Explicitly choose raw or fisher FC; baseline convention is unconfirmed')
    if h5py.is_hdf5(path):
        arrays = _hdf5_arrays(path)
    else:
        d = loadmat(path, simplify_cells=True)
        if all(k in d for k in ['FC_ctrl_129', 'FC_schz_129']):
            arrays = [d[k] for k in ['FC_ctrl_129', 'FC_schz_129']]
        else:
            fc = d['SC_FC_Connectomes']['FC_correlation']
            arrays = []
            for group in ['ctrl', 'schz']:
                matches = [a for a in np.asarray(fc[group], dtype=object).ravel()
                           if hasattr(a, 'shape') and len(a.shape) == 3
                           and sum(d == 129 for d in a.shape) == 2]
                if len(matches) != 1:
                    raise ValueError('Could not identify one 129-ROI scale; use export_fc129.m')
                arrays.append(matches[0])
    ctrl, schz = [canonical_matrices(a) for a in arrays]
    if len(ctrl) != 27 or len(schz) != 27:
        raise ValueError(f'Expected 27 CTRL + 27 SCHZ, got {len(ctrl)} + {len(schz)}')
    u, v, H = edge_map(129)
    X = np.concatenate([ctrl, schz])[:, u, v]
    if representation == 'fisher':
        if np.any(np.abs(X) >= 1):
            raise ValueError('Off-diagonal |r| >= 1; investigate before Fisher transform')
        X = np.arctanh(X)
    names = atlas_labels(xls_path, cortical_path)
    ids = np.array([f'{g}_{i+1:02d}' for g in ['ctrl', 'schz'] for i in range(27)])
    # Group/position IDs are not clinical identifiers; retain the original ordering.
    prov = dict(kind='dataset1', data_sha256=sha256_file(path),
                atlas_sha256=sha256_file(xls_path), labels_sha256=sha256_file(cortical_path),
                representation=representation, shape=list(X.shape),
                diagnosis_mapping={'ctrl': 0, 'schz': 1},
                baseline_representation_confirmed=False,
                subject_ids='group plus original within-group position',
                FC_range=[float(X.min()), float(X.max())])
    return Dataset(X, np.r_[np.zeros(27, int), np.ones(27, int)], ids, names, u, v, H, prov)


def synthetic_dataset(seed=71, n=60, n_rois=8):
    from scipy.special import expit
    rng = np.random.default_rng(seed)
    u, v, H = edge_map(n_rois)
    X = rng.normal(size=(n, len(u)))
    beta = np.zeros(len(u)); beta[u == 0] = 0.7
    y = rng.binomial(1, expit(X@beta))
    return Dataset(X, y, np.array([f'synthetic_{i:03d}' for i in range(n)]),
                   np.array([f'roi_{i+1}' for i in range(n_rois)]), u, v, H,
                   dict(kind='synthetic_only', seed=seed, n=n, n_rois=n_rois))
