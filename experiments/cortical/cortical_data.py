"""Atlas-verified cortical-to-cortical features; no outcome-based filtering."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.io import loadmat
from szv3.data import Dataset,canonical_matrices,sha256_file
from szv3.math import edge_map

HERE=Path(__file__).resolve().parent
SCALES={83:dict(cell=0,columns=(8,18),cortex=68,atlas='scale33'),
        129:dict(cell=1,columns=(6,16),cortex=114,atlas='scale60')}

def protocol():return json.loads((HERE/'reference/dataset1_129_protocol.json').read_text())

def cortical_atlas(resolution):
    import xlrd
    spec=SCALES[resolution];pr=protocol()
    xls=HERE/'reference/ParcellationLausanne2008.xls'
    mat=HERE/'reference/labels_index_CORTICAL_Laus2008_all_scales.mat'
    if sha256_file(xls)!=pr['atlas_sha256'] or sha256_file(mat)!=pr['labels_sha256']:
        raise ValueError('Atlas metadata checksum mismatch')
    sheet=xlrd.open_workbook(str(xls)).sheet_by_name('LABELS');names={}
    for row in range(2,sheet.nrows):
        for col,prefix in zip(spec['columns'],('rh_','lh_')):
            idx,label=sheet.cell_value(row,col),sheet.cell_value(row,col+1)
            if isinstance(idx,(float,int)) and idx>0 and label:
                if int(idx)!=idx or int(idx) in names:raise ValueError('Duplicate/noninteger atlas ID')
                label=str(label).strip();names[int(idx)]=label if label.lower()=='brainstem' else prefix+label
    if set(names)!=set(range(1,resolution+1)):raise ValueError('Full atlas IDs do not cover this resolution')
    cells=loadmat(mat,simplify_cells=True)
    ids=np.asarray(cells['ixc'][spec['cell']],dtype=int)
    labels=np.asarray([str(x).strip() for x in cells['llist'][spec['cell']]])
    if len(ids)!=spec['cortex'] or len(set(ids))!=len(ids) or np.any(ids<1) or np.any(ids>resolution):
        raise ValueError('Unexpected cortical ROI index set')
    if len(labels)!=len(ids) or any(names[int(i)]!=label for i,label in zip(ids,labels)):
        raise ValueError('Cortical MAT index/label does not match XLS')
    forbidden=('thalamus','caudate','putamen','pallidum','hippocampus','amygdala','accumbens','brainstem')
    if any(any(s in name.lower() for s in forbidden) for name in labels):
        raise ValueError('Noncortical label in the cortical index list')
    return ids,labels,{i:names[i] for i in sorted(set(names)-set(ids))}

def read_full_arrays(path,resolution):
    import h5py
    path=Path(path)
    keys=[f'FC_{g}_{resolution}' for g in ('ctrl','schz')]
    if h5py.is_hdf5(path):
        with h5py.File(path,'r') as f:
            if all(k in f for k in keys):arrays=[np.asarray(f[k]) for k in keys]
            else:
                base=f['SC_FC_Connectomes/FC_correlation'];arrays=[]
                for group in ('ctrl','schz'):
                    refs=base[group]
                    if h5py.check_dtype(ref=refs.dtype) is None:raise ValueError('Expected MATLAB scale references')
                    matches=[ref for ref in np.asarray(refs).ravel()
                             if len(f[ref].shape)==3 and sum(d==resolution for d in f[ref].shape)==2]
                    if len(matches)!=1:raise ValueError(f'Expected exactly one {resolution}-region FC scale for {group}')
                    arrays.append(np.asarray(f[matches[0]]))
    else:
        d=loadmat(path,simplify_cells=True)
        if not all(k in d for k in keys):raise ValueError('Expected verified compact FC arrays or original v7.3 MAT')
        arrays=[d[k] for k in keys]
    a=[canonical_matrices(x,n_rois=resolution) for x in arrays]
    if any(len(x)!=27 for x in a):raise ValueError('Expected original 27 controls and 27 patients')
    return a

def cortical_features(arrays,resolution):
    """MATLAB IDs -> Python indices on BOTH ROI axes before upper-triangle vectorization."""
    ids,labels,excluded=cortical_atlas(resolution)
    full=[canonical_matrices(x,n_rois=resolution) for x in arrays]
    matrices=np.concatenate(full,axis=0)
    cortical=matrices[:,ids-1,:][:,:,ids-1]
    u,v,H=edge_map(len(ids));raw=cortical[:,u,v]
    np.testing.assert_array_equal(raw,matrices[:,ids[u]-1,ids[v]-1])
    if not np.isfinite(raw).all() or np.any(np.abs(raw)>=1):raise ValueError('Invalid signed off-diagonal Pearson correlations')
    return np.arctanh(raw),ids,labels,u,v,H,excluded,raw

def load_cortical(path,resolution):
    path=Path(path);source=sha256_file(path)
    cache_manifest=HERE/'data/CORTICAL_DATA_MANIFEST.json'
    if cache_manifest.exists():
        cm=json.loads(cache_manifest.read_text())
        if source==cm['files'].get(str(resolution),{}).get('sha256'):
            rec=cm['files'][str(resolution)]
            with np.load(path,allow_pickle=False) as a:
                X=a['X'].copy();ids=a['original_roi_ids'].copy();labels=a['roi_names'].copy();y=a['y'].copy();subjects=a['subject_ids'].copy()
            atlas_ids,atlas_names,excluded=cortical_atlas(resolution)
            np.testing.assert_array_equal(ids,atlas_ids);np.testing.assert_array_equal(labels,atlas_names)
            np.testing.assert_array_equal(y,np.r_[np.zeros(27,int),np.ones(27,int)])
            u,v,H=edge_map(len(ids))
            if X.shape!=(54,len(u)) or not np.isfinite(X).all() or hashlib.sha256(X.tobytes()).hexdigest()!=rec['X_sha256']:
                raise ValueError('Cortical feature content mismatch')
            prov=rec['provenance']
            return Dataset(X,y,subjects,labels,u,v,H,prov)
    pr=protocol()
    if source!=pr['source_original_sha256']:
        raise ValueError('Source checksum mismatch: use the original shared 27_SCHZ_CTRL_dataset.mat')
    arrays=read_full_arrays(path,resolution)
    return dataset_from_arrays(arrays,resolution,source)

def dataset_from_arrays(arrays,resolution,source_sha):
    X,ids,names,u,v,H,excluded,raw=cortical_features(arrays,resolution)
    ids_subject=np.array([f'{g}_{i+1:02d}' for g in ('ctrl','schz') for i in range(27)])
    pr=protocol()
    prov=dict(kind='dataset1_cortical_only',original_resolution=resolution,n_rois=len(ids),
        available_edges=len(u),shape=list(X.shape),representation='fisher',
        original_data_sha256=source_sha,atlas_sha256=pr['atlas_sha256'],labels_sha256=pr['labels_sha256'],
        original_roi_ids_matlab_1based=ids.tolist(),excluded_roi_ids_matlab_1based=list(excluded),
        excluded_roi_labels=list(excluded.values()),roi_names=names.tolist(),atlas_scale=SCALES[resolution]['atlas'],
        diagnosis_mapping={'ctrl':0,'schz':1},class_counts={'ctrl':27,'schz':27},
        subject_order='27 ctrl followed by 27 schz; original within-group positions',
        subset_rule='Supplied cortical ixc at the matching scale; both matrix axes; unique off-diagonal upper triangle only',
        preprocessing='Signed Pearson -> Fisher arctanh exactly once; StandardScaler fitted within each relevant training split only',
        raw_FC_range=[float(raw.min()),float(raw.max())],FC_range=[float(X.min()),float(X.max())])
    return Dataset(X,np.r_[np.zeros(27,int),np.ones(27,int)],ids_subject,names,u,v,H,prov)

def prepare(source,destination):
    from szv3.storage import atomic_json,atomic_npz
    source=Path(source);dest=Path(destination);pr=protocol();digest=sha256_file(source)
    if digest!=pr['source_original_sha256']:raise ValueError('Original MAT SHA256 mismatch; do not substitute another source')
    dest.mkdir(parents=True,exist_ok=True);manifest=dict(original_sha256=digest,files={})
    for resolution in SCALES:
        d=dataset_from_arrays(read_full_arrays(source,resolution),resolution,digest)
        if resolution==83:
            for group,a in zip(('ctrl','schz'),read_full_arrays(source,83)):
                if hashlib.sha256(a.tobytes()).hexdigest()!=pr['matrix_sha256'][f'FC_{group}_83']:
                    raise ValueError('83-region extraction differs from prior verified subject order')
        path=dest/f'cortical_{resolution}.npz'
        atomic_npz(path,X=d.X,y=d.y,subject_ids=d.subject_ids,roi_names=d.roi_names,
                   original_roi_ids=np.asarray(d.provenance['original_roi_ids_matlab_1based'],int))
        manifest['files'][str(resolution)]=dict(file=path.name,sha256=sha256_file(path),
            X_sha256=hashlib.sha256(d.X.tobytes()).hexdigest(),provenance=d.provenance)
    atomic_json(dest/'CORTICAL_DATA_MANIFEST.json',manifest)
    return manifest
