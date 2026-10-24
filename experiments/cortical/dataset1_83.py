"""Verified 83-region Dataset-1 scale; preserve all 54 subjects and signed FC."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.io import loadmat
from szv3.data import Dataset, canonical_matrices, sha256_file
from szv3.math import edge_map

HERE = Path(__file__).resolve().parent


def atlas83(xls_path, cortical_path):
    import xlrd
    sheet = xlrd.open_workbook(str(xls_path)).sheet_by_name('LABELS')
    labels = {}
    for row in range(2, sheet.nrows):
        for col, prefix in ((8, 'rh_'), (18, 'lh_')):
            idx, label = sheet.cell_value(row, col), sheet.cell_value(row, col + 1)
            if isinstance(idx, (float, int)) and idx > 0 and label:
                idx = int(idx)
                if idx in labels:
                    raise ValueError('Duplicate atlas ID')
                name = str(label).strip()
                labels[idx] = name if name.lower() == 'brainstem' else prefix + name
    if set(labels) != set(range(1, 84)):
        raise ValueError('Expected scale33 atlas IDs 1..83')
    cortical = loadmat(cortical_path, simplify_cells=True)
    for idx, name in zip(cortical['ixc'][0], cortical['llist'][0]):
        if labels[int(idx)] != str(name).strip():
            raise ValueError('83-scale cortical label mismatch')
    return np.array([labels[i] for i in range(1, 84)])


def load_dataset83(path, representation='fisher', verify_source=True):
    import h5py
    path = Path(path)
    protocol = json.loads((HERE/'reference/dataset1_129_protocol.json').read_text())
    if representation != protocol['representation']:
        raise ValueError('Preserve the verified 129-region representation: fisher')
    xls = HERE/'reference/ParcellationLausanne2008.xls'
    cortical = HERE/'reference/labels_index_CORTICAL_Laus2008_all_scales.mat'
    if sha256_file(xls) != protocol['atlas_sha256'] or sha256_file(cortical) != protocol['labels_sha256']:
        raise ValueError('Atlas source checksum mismatch')
    input_sha = sha256_file(path)
    if verify_source and input_sha not in (protocol['compact_mat_sha256'], protocol['source_original_sha256']):
        raise ValueError('Dataset source checksum mismatch')
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            base = f['SC_FC_Connectomes/FC_correlation']
            arrays = []
            for group in ('ctrl', 'schz'):
                matches = [np.asarray(f[ref]) for ref in np.asarray(base[group]).ravel()
                           if len(f[ref].shape) == 3 and sum(d == 83 for d in f[ref].shape) == 2]
                if len(matches) != 1:
                    raise ValueError('Expected exactly one 83-region FC scale per group')
                arrays.append(matches[0])
    else:
        mat = loadmat(path, simplify_cells=True)
        arrays = [mat['FC_ctrl_83'], mat['FC_schz_83']]
    ctrl, schz = [canonical_matrices(a, n_rois=83) for a in arrays]
    if len(ctrl) != 27 or len(schz) != 27:
        raise ValueError('Expected 27 controls and 27 schizophrenia subjects')
    for group, a in zip(('ctrl', 'schz'), (ctrl, schz)):
        if verify_source and hashlib.sha256(a.tobytes()).hexdigest() != protocol['matrix_sha256']['FC_'+group+'_83']:
            raise ValueError('83-region matrix content differs from the verified extraction')
    u, v, H = edge_map(83)
    raw = np.concatenate((ctrl, schz))[:, u, v]
    if np.any(np.abs(raw) >= 1):
        raise ValueError('Off-diagonal |correlation| >= 1; cannot Fisher-transform')
    X = np.arctanh(raw)
    ids = np.array([f'{g}_{i+1:02d}' for g in ('ctrl', 'schz') for i in range(27)])
    provenance = dict(kind='dataset1_83', original_data_sha256=protocol['source_original_sha256'],
        input_sha256=input_sha, representation=representation, shape=list(X.shape), n_rois=83,
        atlas_scale='scale33', atlas_sha256=protocol['atlas_sha256'], labels_sha256=protocol['labels_sha256'],
        diagnosis_mapping={'ctrl': 0, 'schz': 1}, class_counts={'ctrl': 27, 'schz': 27},
        subject_ids='group plus original within-group position; same ordering as 129-region study',
        raw_FC_range=[float(raw.min()), float(raw.max())], FC_range=[float(X.min()), float(X.max())],
        retained_rois='All 83: 68 cortical, 14 subcortical, 1 brainstem',
        preprocessing='Fisher arctanh once, followed by StandardScaler fit within each training fold only')
    return Dataset(X, np.r_[np.zeros(27, int), np.ones(27, int)], ids, atlas83(xls, cortical), u, v, H, provenance)
