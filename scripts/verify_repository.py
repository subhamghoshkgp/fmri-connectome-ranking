"""Verify the packaged study without fitting or modifying any model."""
from pathlib import Path
from collections import Counter
import argparse,ast,csv,json,re,statistics,sys

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--require-models',action='store_true',help='Also require every fitted-model pair available in the full staged repository.')
args=parser.parse_args()
folds={}
for scope in ['whole_brain_83','whole_brain_129','cortical_68','cortical_114']:
    rows=list(csv.DictReader((ROOT/f'results/{scope}/analysis/fold_metrics.csv').open()))
    counts=Counter(r['method'] for r in rows)
    assert len(rows)==1200 and len(counts)==12 and set(counts.values())=={100},scope
    assert len({(r['method'],r['repeat'],r['fold']) for r in rows})==1200,scope
    folds[scope]=rows
name_to_id={'LASSO':'max_auc_lasso','Elastic net':'max_auc_enet','IGL-ref':'max_auc_v3','RFE-SVM':'max_auc_rfe_svm','IGL':'v3_expanded','IGR':'v3_rank','IGR-R':'v3_rank_refit','DRR (dense)':'ridge_rank'}
overview=list(csv.DictReader((ROOT/'results/primary_results_overview.csv').open()))
assert len(overview)==32
for row in overview:
    scope=f"{row['scope']}_{row['regions']}"
    target=name_to_id[row['method']]
    rr=[r for r in folds[scope] if r['method'].removeprefix('saved_')==target]
    assert len(rr)==100
    for name,column in [('auc','auc_mean'),('support','mean_edges'),('balanced_accuracy','balanced_accuracy'),('f1','f1')]:
        assert abs(statistics.mean(float(r[name]) for r in rr)-float(row[column]))<1e-11,(scope,row['method'],name)
    assert abs(statistics.stdev(float(r['auc']) for r in rr)-float(row['auc_fold_sd']))<1e-11

if args.require_models:
    base=ROOT/'results/whole_brain_83/outer_models'
    for method in {r['method'] for r in folds['whole_brain_83']}:
        pairs=list(base.glob(f'repeat_*/fold_*/{method}.json'))
        assert len(pairs)==100 and all(p.with_suffix('.npz').is_file() for p in pairs),method
    base=ROOT/'results/whole_brain_129/outer_models_partial_65'
    for method in ['v3_expanded','v3_rank','v3_rank_refit','ridge_rank']:
        pairs=list(base.glob(f'repeat_*/fold_*/{method}.json'))
        assert len(pairs)==65 and all(p.with_suffix('.npz').is_file() for p in pairs),method
    base=ROOT/'results/reselection_129'
    for method in ['v3','enet','lasso','rfe_svm']:
        pairs=list(base.glob(f'repeat_*/fold_*/{method}.json'))
        assert len(pairs)==100 and all(p.with_suffix('.npz').is_file() for p in pairs),method

python_files=list(ROOT.rglob('*.py'))
for p in python_files:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
notebooks=list((ROOT/'notebooks').glob('*.ipynb'))
for p in notebooks:
    n=json.loads(p.read_text());assert n['nbformat']==4 and n['cells'],p.name

patterns=[re.compile(r'(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(r'github_pat_[A-Za-z0-9_]{40,}'),re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
checked=0
for p in ROOT.rglob('*'):
    if not p.is_file() or '.git' in p.parts:continue
    assert p.stat().st_size<100*1024*1024, f'Large file requires a separate storage decision: {p.relative_to(ROOT)}'
    if p.suffix in ['.py','.ipynb','.md','.txt','.json','.yaml','.yml','.toml']:
        text=p.read_text(encoding='utf-8-sig');checked+=1
        assert not any(pattern.search(text) for pattern in patterns), f'Credential-pattern match: {p.relative_to(ROOT)}'

print(json.dumps(dict(status='passed',representations=4,completed_fold_metric_rows=4800,primary_summary_rows=32,python_files_parsed=len(python_files),notebooks_validated=len(notebooks),text_files_checked_for_credential_patterns=checked,fitted_model_archive_checks=bool(args.require_models)),indent=2))
