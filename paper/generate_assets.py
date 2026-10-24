"""Regenerate the publication table and vector figures from the archived results."""
from pathlib import Path
import csv,json,math,statistics,xml.etree.ElementTree as ET
from decimal import Decimal,ROUND_HALF_UP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,PathPatch
from matplotlib.path import Path as MPath
from matplotlib.ticker import FixedLocator,FuncFormatter

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
rows=list(csv.DictReader((ROOT/'paper_results.csv').open(newline='')))
for r in rows:
 r['resolution']=int(r['regions'])
 for k in r:
  if k.endswith(('_mean','_sd')) or k in ('jaccard','nogueira'):
   r[k]=float(r[k]) if r[k] else None

def normalize(m):return m.replace('saved_max_auc_','max_auc_').replace('saved_one_se_','one_se_')
checks=[]
for resolution in (83,129):
 fold=list(csv.DictReader((ROOT/f'evidence/outer_fold_metrics_{resolution}.csv').open(newline='')))
 for row in [r for r in rows if r['resolution']==resolution]:
  rr=[r for r in fold if normalize(r['method'])==normalize(row['saved_method_id'])]
  assert len(rr)==100 and len({(r['repeat'],r['fold']) for r in rr})==100
  for metric in ('auc','accuracy','balanced_accuracy','f1','sensitivity','specificity','support'):
   mean=statistics.mean(float(r[metric]) for r in rr)
   assert abs(mean-row[metric+'_mean'])<1e-12,(resolution,row['method'],metric)
   if row.get(metric+'_fold_sd') is not None:
    assert abs(statistics.stdev(float(r[metric]) for r in rr)-row[metric+'_fold_sd'])<1e-12
  checks.append(dict(resolution=resolution,method=row['method'],folds=100,means_and_available_sds_match=True))
(ROOT/'evidence/RESULTS_VERIFICATION.json').write_text(json.dumps(checks,indent=2)+'\n')
order=['LASSO','Elastic net','IGL-ref','RFE-SVM','IGR-R','IGR','IGL','DRR (dense)']
import runpy
runpy.run_path(str(ROOT/'generate_results_table.py'))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8.5,
 'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
styles={'LASSO':('#6b7280','o'),'Elastic net':('#6b7280','s'),'IGL-ref':('#92400e','v'),
        'RFE-SVM':('#434b56','D'),'IGL':('#cc7813','^'),'IGR':('#4268a6','o'),
        'IGR-R':('#a94b7e','s'),'DRR (dense)':('#16816c','P')}
# Coordinates are label anchors; marker locations are always the exact CSV means.
label_positions={83:{'LASSO':(23,.636),'Elastic net':(78,.675),'IGL-ref':(112,.716),
 'RFE-SVM':(290,.737),'IGR-R':(325,.777),'IGR':(1200,.741),'IGL':(1350,.782),'DRR (dense)':(2300,.821)},
 129:{'LASSO':(23,.630),'Elastic net':(82,.744),'IGL-ref':(330,.691),
 'RFE-SVM':(260,.797),'IGR-R':(520,.827),'IGR':(1400,.761),'IGL':(1750,.809),'DRR (dense)':(5500,.845)}}
fig,axes=plt.subplots(1,2,figsize=(7.06,2.30),sharey=True)
fig.subplots_adjust(left=.075,right=.98,bottom=.23,top=.87,wspace=.12)
for ax,resolution in zip(axes,(83,129)):
 ax.set_xscale('log');ax.set_xlim(13,15000);ax.set_ylim(.617,.858)
 ax.xaxis.set_major_locator(FixedLocator([20,100,500,2000,10000]));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:,.0f}'))
 ax.minorticks_off();ax.set_yticks([.65,.70,.75,.80,.85]);ax.grid(axis='y',color='#e4e8ed',lw=.5)
 ax.spines[['top','right']].set_visible(False)
 ax.set_title(f'{resolution} regions ({resolution*(resolution-1)//2:,} available edges)',pad=5)
 ax.set_xlabel('Mean selected / used edges (log scale)',labelpad=5)
 for method in order:
  r=next(x for x in rows if x['resolution']==resolution and x['method']==method)
  color,marker=styles[method];x,y=r['support_mean'],r['auc_mean']
  ax.scatter([x],[y],s=26 if method!='DRR (dense)' else 44,c=[color],marker=marker,linewidth=.5,edgecolors='white',zorder=4)
  label='DRR (all edges)' if method=='DRR (dense)' else method
  ax.annotate(label,xy=(x,y),xytext=label_positions[resolution][method],textcoords='data',
     color=color,fontsize=7.3,ha='center',va='center',
     arrowprops=dict(arrowstyle='-',lw=.55,color=color,shrinkA=2,shrinkB=4))
axes[0].set_ylabel('Mean outer-test ROC-AUC',labelpad=5)
fig.savefig(FIG/'sparse_to_dense.pdf');fig.savefig(FIG/'sparse_to_dense.svg');fig.savefig(FIG/'sparse_to_dense.png',dpi=240)
plt.close(fig)

# Compact full-width diagram. Dense ridge receives all X directly, never sparse-refit output.
fig,ax=plt.subplots(figsize=(7.06,1.53));fig.subplots_adjust(0,0,1,1)
ax.set_xlim(0,1000);ax.set_ylim(0,218);ax.axis('off')
boxes={
 'input':(35,104,185,68,'Training features',r'Signed Fisher FC $X$'+'\nTraining-only scaling','#f1f4f7','#536273'),
 'gate':(280,104,185,68,'Incidence gating',r'$|\beta|\leq H^{\mathsf{T}}z,\ z\geq0$'+'\nAtlas endpoints $H$','#eaf1fa','#4268a6'),
 'rank':(525,104,185,68,'Pairwise ranking',r'$\ell_{\mathrm{rank}}(\beta)$'+'\nIGR selector','#eaf1fa','#4268a6'),
 'refit':(770,104,195,68,'Sparse ridge refit',r'Selected mask $S$'+'\nIGR-R predictor','#f9edf4','#a94b7e'),
 'dense':(770,9,195,64,'Dense ridge ranking',r'$\ell_{\mathrm{rank}}+\rho\|\beta\|_2^2/2$'+'\nDRR: all $p$ edges','#e8f5f0','#16816c')}
for k,(x,y,w,h,title,body,fill,color) in boxes.items():
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=5',facecolor=fill,edgecolor=color,lw=.8))
 ax.text(x+w/2,y+h-15,title,ha='center',va='center',fontsize=8.2,fontweight='bold',color=color)
 ax.text(x+w/2,y+25,body,ha='center',va='center',fontsize=7.7,color='#1f2937',linespacing=1.25)
for start,end in [((220,138),(280,138)),((465,138),(525,138)),((710,138),(770,138))]:
 ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=8,lw=.85,color='#536273'))
# Joint-fit bracket above the gating and pairwise-loss boxes.
ax.plot([275,275,715,715],[178,190,190,178],color='#4268a6',lw=.7)
ax.text(495,192,'Joint incidence-gated ranking fit (IGR)',ha='center',va='bottom',fontsize=7.7,color='#4268a6')
# An independent all-feature branch begins at X, not at the sparse predictor.
ax.plot([125,125,740],[104,41,41],color='#16816c',lw=.9)
ax.add_patch(FancyArrowPatch((740,41),(770,41),arrowstyle='-|>',mutation_scale=8,lw=.9,color='#16816c'))
ax.text(430,46,'Same training data; all features retained',ha='center',va='bottom',fontsize=7.8,color='#16816c')
ax.text(505,87,'IGL control: replace pairwise ranking by pointwise logistic loss',ha='center',va='center',fontsize=7.5,color='#536273')
fig.savefig(FIG/'ranking_framework.pdf');fig.savefig(FIG/'ranking_framework.svg');fig.savefig(FIG/'ranking_framework.png',dpi=260)
plt.close(fig)

# Editable diagrams.net / draw.io source, using the same box positions and branch topology.
mxfile=ET.Element('mxfile',host='app.diagrams.net',version='26.0.0')
diagram=ET.SubElement(mxfile,'diagram',id='ranking-framework',name='Sparse-to-dense ranking')
model=ET.SubElement(diagram,'mxGraphModel',dx='1100',dy='450',grid='1',gridSize='10',page='1',pageWidth='1100',pageHeight='300',math='1')
root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
texts={'input':'Training features<br>Signed Fisher FC X<br>Training-only scaling',
 'gate':'Incidence gating<br>|β| ≤ Hᵀz, z ≥ 0<br>Atlas endpoints H',
 'rank':'Pairwise ranking<br>Positive–negative training pairs<br>IGR selector',
 'refit':'Sparse ridge refit<br>Selected mask S<br>IGR-R predictor',
 'dense':'Dense ridge ranking<br>Pairwise loss + ridge penalty<br>DRR: all p edges'}
for k,(x,y,w,h,title,body,fill,color) in boxes.items():
 cell=ET.SubElement(root,'mxCell',id=k,value=texts[k],vertex='1',parent='1',style=f'rounded=1;whiteSpace=wrap;html=1;fontSize=13;fillColor={fill};strokeColor={color};')
 ET.SubElement(cell,'mxGeometry',x=str(x),y=str(210-y-h),width=str(w),height=str(h),**{'as':'geometry'})
for i,(a,b) in enumerate([('input','gate'),('gate','rank'),('rank','refit'),('input','dense')]):
 style='edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeColor=#536273;'
 if b=='dense':style+='exitX=0.5;exitY=1;entryX=0;entryY=0.5;strokeColor=#16816c;'
 cell=ET.SubElement(root,'mxCell',id=f'e{i}',edge='1',parent='1',source=a,target=b,style=style,
  value='All p edges' if b=='dense' else '')
 ET.SubElement(cell,'mxGeometry',relative='1',**{'as':'geometry'})
for k,text,x,y,w,h in [('joint','Joint incidence-gated ranking fit (IGR)',275,0,440,24),('control','IGL control: pointwise logistic loss',300,119,410,24)]:
 cell=ET.SubElement(root,'mxCell',id=k,value=text,vertex='1',parent='1',style='text;html=1;fontSize=12;align=center;')
 ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
ET.indent(mxfile);ET.ElementTree(mxfile).write(FIG/'ranking_framework.drawio',encoding='utf-8',xml_declaration=True)
print('Verified 16 primary rows; generated merged table and two vector figures.')
