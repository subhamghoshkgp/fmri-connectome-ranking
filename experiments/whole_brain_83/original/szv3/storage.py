import contextlib
import errno
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


def sync_parent(path):
    """Persist rename metadata where directory fsync is available (POSIX)."""
    if os.name == 'nt': return
    fd = os.open(str(Path(path).parent), os.O_RDONLY)
    try:
        try: os.fsync(fd)
        except OSError as exc:
            # Drive's FUSE mount may not implement directory fsync. File writes
            # still flush before rename; do not suppress actual I/O failures.
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS): raise
    finally: os.close(fd)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp = tempfile.mkstemp(dir=path.parent, prefix='.pending_', suffix='.json')
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(jsonable(value),f,sort_keys=True,indent=2,allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
        sync_parent(path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp = tempfile.mkstemp(dir=path.parent, prefix='.pending_', suffix='.npz')
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,**arrays); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
        sync_parent(path)
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
    """Nonblocking OS lock; automatically released when a process exits/crashes."""
    Path(out).mkdir(parents=True,exist_ok=True)
    with open(Path(out)/'.run.lock','a+b') as f:
        if os.name == 'nt':
            import msvcrt
            # Windows locks a byte range, so retain a permanent byte in this file.
            if os.fstat(f.fileno()).st_size == 0:
                f.write(b'0'); f.flush(); os.fsync(f.fileno())
            f.seek(0)
            try: msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as exc:
                raise RuntimeError('Another process is writing this run directory') from exc
        else:
            import fcntl
            try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Another process is writing this run directory') from exc
        try: yield
        finally:
            if os.name == 'nt':
                f.seek(0); msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f,fcntl.LOCK_UN)


class RunPaused(Exception):
    """Control-flow signal, never an invalid hyperparameter candidate."""


def check_stop(path):
    """Look up the current experiment root and pause between committed fits."""
    path=Path(path)
    for parent in (path, *path.parents):
        if (parent/'manifest.json').is_file():
            if (parent/'STOP_REQUESTED').exists():
                raise RunPaused('Safe stop requested; committed checkpoints are retained')
            return
