import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import numpy as np


def jsonable(obj):
    if isinstance(obj, dict): return {str(k):jsonable(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)): return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    return obj


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp = tempfile.mkstemp(dir=path.parent, prefix='.pending_', suffix='.json')
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(jsonable(value),f,sort_keys=True,indent=2,allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp = tempfile.mkstemp(dir=path.parent, prefix='.pending_', suffix='.npz')
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,**arrays); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def digest_obj(obj):
    return hashlib.sha256(json.dumps(jsonable(obj),sort_keys=True).encode()).hexdigest()


def source_digest():
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        digest.update(str(p.relative_to(root)).encode()); digest.update(p.read_bytes())
    return digest.hexdigest()


def environment():
    packages = ['numpy','scipy','scikit-learn','cvxpy','clarabel','h5py','xlrd']
    return dict(python=platform.python_version(),
                packages={p:importlib.metadata.version(p) for p in packages})


def init_manifest(out, payload):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    path=out/'manifest.json'; signature=digest_obj(payload)
    if path.exists():
        old=json.loads(path.read_text())
        if old['signature']!=signature:
            raise ValueError('Resume rejected: data/config/source/environment changed. Use a NEW output directory.')
        return old
    payload={**payload,'signature':signature}
    atomic_json(path,payload)
    return payload


@contextlib.contextmanager
def run_lock(out):
    """Advisory OS lock; releases on crash. Linux/macOS/Colab/WSL deployment."""
    import fcntl
    Path(out).mkdir(parents=True,exist_ok=True)
    with open(Path(out)/'.run.lock','a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another process is writing this run directory')
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)
