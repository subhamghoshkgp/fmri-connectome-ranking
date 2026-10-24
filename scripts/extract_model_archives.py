"""Verify and restore the original fitted-model paths from per-repeat archives."""
from pathlib import Path
import hashlib,json,shutil,zipfile

ROOT=Path(__file__).resolve().parents[1]
index=json.loads((ROOT/'model_archives/INDEX.json').read_text())
restored=0
for record in index:
    archive=ROOT/record['archive']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==record['sha256'],archive
    with zipfile.ZipFile(archive) as z:
        assert len(z.infolist())==record['members'],archive
        assert z.testzip() is None,archive
        for item in z.infolist():
            target=(ROOT/item.filename).resolve()
            assert target.is_relative_to(ROOT.resolve()),item.filename
            if item.is_dir():
                target.mkdir(parents=True,exist_ok=True)
                continue
            if target.exists():
                with z.open(item) as source:
                    assert hashlib.sha256(source.read()).digest()==hashlib.sha256(target.read_bytes()).digest(),f'Existing file differs: {target}'
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(item) as source,target.open('wb') as output:shutil.copyfileobj(source,output)
            restored+=1
print(f'Verified/restored {restored} files from {len(index)} archives.')
