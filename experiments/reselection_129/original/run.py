#!/usr/bin/env python3
"""CLI: set thread limits BEFORE importing numerical libraries."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/smoke.json')
    p.add_argument('--data'); p.add_argument('--atlas'); p.add_argument('--labels')
    p.add_argument('--representation',choices=['raw','fisher'])
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--out',required=True); p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--max-outer',type=int,help='Execution limit only; resume same run later without this option')
    args=p.parse_args()
    from szv3.data import load_dataset,synthetic_dataset
    from szv3.experiment import run
    if args.synthetic:
        data=synthetic_dataset()
    else:
        if not all([args.data,args.atlas,args.labels,args.representation]):
            p.error('Real runs require --data --atlas --labels and explicit --representation')
        data=load_dataset(args.data,args.atlas,args.labels,args.representation)
    cfg=json.loads(Path(args.config).read_text())
    # Small synthetic graphs require feature counts within their feature universe.
    for row in cfg['grids'].get('rfe_svm',[]):
        row['k']=min(row['k'],data.X.shape[1])
    result=run(data,cfg,args.out,args.jobs,args.max_outer)
    print('Complete:',result['complete'],'Data:',result['kind'])
    print('Report:',str(Path(args.out)/'analysis'/'SUMMARY.md'))


if __name__=='__main__': main()
