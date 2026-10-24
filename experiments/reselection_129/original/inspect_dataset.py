"""Print HDF5 paths/shapes or MATLAB variable metadata without dumping data."""
import argparse
import h5py
from scipy.io import whosmat
p=argparse.ArgumentParser(); p.add_argument('data'); a=p.parse_args()
if h5py.is_hdf5(a.data):
    with h5py.File(a.data,'r') as f:
        f.visititems(lambda name,obj: print(name,getattr(obj,'shape','group'),getattr(obj,'dtype','')))
else:
    print(whosmat(a.data))
