"""Recreate the merged table for both resolutions and anatomical scopes."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import csv
PKG=Path(__file__).resolve().parent
full=list(csv.DictReader((PKG/'paper_results.csv').open()))
cortex=list(csv.DictReader((PKG/'evidence/cortical_paper_results.csv').open()))
order=['LASSO','Elastic net','IGL-ref','RFE-SVM','IGR-R','IGR','IGL','DRR (dense)']
def num(v,n=4):return f'{float(v):.{n}f}'
def edge(v):return str(Decimal(v).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP))
lines=[r'\begin{table*}[t]',r'\centering',
r'\caption{Sparse-to-dense results on the same 54 participants and 100 saved outer folds per representation. AUC: mean $\pm$ fold SD; edges, BA and F1: means; $\widehat\Phi$: Nogueira stability. Bold marks highest AUC and adjusted stability within each representation. All primary rows use maximum-inner-AUC tuning.}',
r'\label{tab:primary}',r'\fontsize{9}{10.5}\selectfont',r'\setlength{\tabcolsep}{2.4pt}',
r'\begin{tabular}{@{}lcrrrrcrrrr@{}}',r'\toprule']
for scope in ['whole','cortex']:
    headers= ['83 whole-brain regions; 3,403 edges','129 whole-brain regions; 8,256 edges'] if scope=='whole' else ['68 cortical regions; 2,278 edges','114 cortical regions; 6,441 edges']
    lines += [r'& \multicolumn{5}{c}{'+headers[0]+r'} & \multicolumn{5}{c}{'+headers[1]+r'} \\',r'\cmidrule(lr){2-6}\cmidrule(l){7-11}',r'Method & AUC & Edges & BA & F1 & $\widehat\Phi$ & AUC & Edges & BA & F1 & $\widehat\Phi$ \\',r'\midrule']
    for name in order:
        fields=[name.replace('RFE-SVM','RFE--SVM')]
        for res in [83,129]:
            rr=full if scope=='whole' else cortex
            r=next(x for x in rr if x['method']==name and int(x['regions'] if scope=='whole' else x['original_regions'])==res)
            auc=num(r['auc_mean'])+r' $\pm$ '+num(r['auc_fold_sd'])
            if name=='DRR (dense)':auc=r'\textbf{'+auc+'}'
            phi=num(r['nogueira']) if r['nogueira'] else '--'
            if (scope=='whole' and name=='IGR') or (scope=='cortex' and ((res==83 and name=='IGR-R') or (res==129 and name=='Elastic net'))):phi=r'\textbf{'+phi+'}'
            fields += [auc,edge(r['support_mean']),num(r['balanced_accuracy_mean']),num(r['f1_mean']),phi]
        lines += [' & '.join(fields)+r' \\']
    if scope=='whole':lines += [r'\midrule']
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(PKG/'results_table.tex').write_text('\n'.join(lines)+'\n')

print('Generated one merged table with 32 primary method/representation results.')
