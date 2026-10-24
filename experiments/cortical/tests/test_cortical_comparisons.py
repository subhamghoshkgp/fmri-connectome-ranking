from pathlib import Path
import csv,json,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'original')]
from run_cortical import read_metrics,compare_fullbrain,METHODS
from run_rank_v3 import csv_file


def test_reference_folds_resolutions_and_method_mapping():
    for resolution in (83,129):
        indexed=read_metrics(ROOT/f'reference/full_brain_{resolution}_fold_metrics.csv')
        assert len(indexed)==1200
        for m in METHODS:assert sum(k[0]==m for k in indexed)==100


def test_matched_comparison_delta_direction_and_partial_labels(tmp_path):
    a=read_metrics(ROOT/'reference/full_brain_83_fold_metrics.csv')
    rows=[]
    for key in sorted(a):
        m,r,f=key
        if r!=0 or f!=0:continue
        metrics=a[key].copy();metrics['auc']+=.0123
        rows.append(dict(method=m,repeat=r,fold=f,**metrics))
    csv_file(tmp_path/'analysis/fold_metrics.csv',rows)
    compare_fullbrain(tmp_path,83,False)
    result=list(csv.DictReader((tmp_path/'analysis/cortical_vs_fullbrain_summary.csv').open()))
    assert len(result)==12
    for row in result:
        assert row['complete_study']=='False' and row['paired_folds']=='1'
        assert abs(float(row['auc_cortical_minus_fullbrain_mean'])-.0123)<1e-12
        assert row['auc_wins']=='1' and row['auc_losses']=='0'
    with pytest.raises(ValueError,match='coverage'):compare_fullbrain(tmp_path,83,True)
