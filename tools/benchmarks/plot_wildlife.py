"""Render measured transfer-learning results; no hard-coded performance values."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
a=p.parse_args();e=Path(a.evidence);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
s=json.loads((e/'summary.json').read_text());curves=json.loads((e/'curves.json').read_text())
names=list(s['models']);labels=['Bear-B · shared','YOLO26-x · shared','YOLO11-x · shared','YOLO26-x · default','YOLO11-x · default']
colors=['#2765a4','#248b70','#c47b20','#248b70','#c47b20']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig=plt.figure(figsize=(14,10),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1.1])
ax=fig.add_subplot(gs[0,0]);values=[s['models'][n]['AP'] for n in names]
ci=s['bootstrap']['AP_95CI'];lower=[max(0,v-ci[n][0]) for n,v in zip(names,values)];upper=[max(0,ci[n][1]-v) for n,v in zip(names,values)]
y=np.arange(len(names));ax.barh(y,values,color=colors,alpha=.88,xerr=[lower,upper],error_kw=dict(capsize=3,lw=1.2))
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xticks(range(0,101,20));ax.set_xlabel('Held-out test COCO AP50:95 (%)');ax.set_title('Common evaluator · 224 test images',loc='left',weight='bold');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for i,v in enumerate(values):ax.text(ci[names[i]][1]+1.2,i,f'{v:.2f}',va='center',fontsize=10,weight='bold')
ax=fig.add_subplot(gs[0,1]);classes=['buffalo','elephant','rhino','zebra'];matrix=np.array([[s['models'][n]['per_class_AP'][c] for c in classes] for n in names])
im=ax.imshow(matrix,vmin=50,vmax=100,cmap='YlGnBu',aspect='auto');ax.set_xticks(range(4),classes);ax.set_yticks(y,labels);ax.set_title('Per-class test AP50:95 (%)',loc='left',weight='bold')
for i in y:
 for j in range(4):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]>80 else '#172433')
fig.colorbar(im,ax=ax,fraction=.035,pad=.03)
ax=fig.add_subplot(gs[1,:]);
for i,(n,c) in enumerate(zip(names,colors)):
 r=curves[n];x=[v['epoch'] for v in r];v=[z['native_val_AP'] for z in r]
 ax.plot(x,v,color=c,ls='--' if 'native' in n else '-',lw=2,label=labels[i])
 best=int(np.argmax(v));ax.scatter([x[best]],[v[best]],color=c,s=24,zorder=4)
ax.set_xlim(1,50);ax.set_ylim(0,100);ax.set_xlabel('Fine-tuning epoch');ax.set_ylabel('Native validation AP50:95 (%)');ax.set_title('Learning curves · validation selects checkpoints',loc='left',weight='bold');ax.grid(alpha=.15);ax.legend(loc='lower right',ncol=2,frameon=False)
fig.suptitle('Bear Detector transfer to African Wildlife',fontsize=19,weight='bold')
fig.supxlabel('50 epochs · seed 42 · batch 32 · fixed 640 input. Test bars: paired image-bootstrap 95% CI (200 samples).\nNative validation implementations differ; use the common test evaluator for cross-model conclusions. One seed; unequal model sizes and pretraining.',fontsize=10)
fig.savefig(out.with_suffix('.svg'));fig.savefig(out.with_suffix('.png'),dpi=150)
