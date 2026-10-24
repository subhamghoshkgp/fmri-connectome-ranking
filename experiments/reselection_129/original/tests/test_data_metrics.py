from pathlib import Path
import numpy as np
import pytest
from scipy.io import savemat
from szv3.data import canonical_matrices,atlas_labels,load_dataset
from szv3.metrics import stability
from szv3.storage import init_manifest


def test_axes_and_rejections():
    a=np.repeat(np.eye(129)[None,:,:],27,axis=0)
    np.testing.assert_array_equal(canonical_matrices(a),canonical_matrices(np.moveaxis(a,0,2)))
    a[0,0,1]=.2
    with pytest.raises(ValueError,match='symmetric'): canonical_matrices(a)


def test_stability_conventions():
    assert stability([[1,2],[1,2]],10)['jaccard']==1
    assert stability([[1,2],[1,2]],10)['kuncheva_equal_size']==1
    assert stability([[1],[1,2]],10)['kuncheva_equal_size'] is None
    assert stability([[],[]],10)['jaccard'] is None
    assert stability([[],[]],10)['nogueira'] is None


def test_manifest_guard(tmp_path):
    init_manifest(tmp_path,{'test':1}); init_manifest(tmp_path,{'test':1})
    with pytest.raises(ValueError,match='Resume rejected'): init_manifest(tmp_path,{'test':2})


def test_supplied_atlas():
    root=Path(__file__).resolve().parents[1]/'reference'
    names=atlas_labels(root/'ParcellationLausanne2008.xls',root/'labels_index_CORTICAL_Laus2008_all_scales.mat')
    assert len(names)==129 and len(set(names))==129
    assert names[57]=='rh_thalamusproper'
    assert names[64]=='lh_lateralorbitofrontal_1'
    assert names[121]=='lh_thalamusproper'
    assert names[128]=='brainstem'


def test_compact_loader(tmp_path):
    root=Path(__file__).resolve().parents[1]/'reference'
    # SYNTHETIC correlation matrices, not the unavailable real file.
    a=np.repeat(np.eye(129)[None,:,:],27,axis=0)
    path=tmp_path/'compact.mat'; savemat(path,{'FC_ctrl_129':a,'FC_schz_129':a})
    d=load_dataset(path,root/'ParcellationLausanne2008.xls',
                   root/'labels_index_CORTICAL_Laus2008_all_scales.mat','raw')
    assert d.X.shape==(54,8256) and d.y.sum()==27


def test_hdf5_loader(tmp_path):
    import h5py
    root=Path(__file__).resolve().parents[1]/'reference'; path=tmp_path/'fixture.mat'
    a=np.repeat(np.eye(129)[None,:,:],27,axis=0)
    with h5py.File(path,'w') as f:
        base=f.create_group('SC_FC_Connectomes/FC_correlation')
        for key in ['ctrl','schz']:
            target=f.create_dataset('scale_'+key,data=a)
            refs=base.create_dataset(key,(1,1),dtype=h5py.ref_dtype); refs[0,0]=target.ref
    d=load_dataset(path,root/'ParcellationLausanne2008.xls',
                   root/'labels_index_CORTICAL_Laus2008_all_scales.mat','raw')
    assert d.X.shape==(54,8256)
