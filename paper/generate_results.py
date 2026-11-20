from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[1];out=root/'paper';data=root/'results/local-run-2026-11-20'
a=pd.read_csv(data/'results_cv.csv',index_col=0);b=pd.read_csv(data/'results_loso_pooled.csv',index_col=0)
models=[('LR|Text-only','LR text'),('LR|Speech-only','LR speech'),('LR|Text+Speech (early fusion)','LR early fusion'),('LR|Multimodal (+mismatch)','LR fusion + mismatch'),('NET|Text-only','NET text'),('NET|Speech-only','NET speech'),('NET|Multimodal gated (+mismatch)','NET gated + mismatch'),('NET|Multimodal gated -mismatch','NET gated, no mismatch')]
def num(v):return f'{v:.3f}'
lines=[r'\begin{table*}[t]',r'\caption{Main results on MUStARD. CV entries are means over 15 folds; the CV F1 column includes fold standard deviations. Cross-show metrics pool the 636 held-out utterances. F1 uses sarcasm as the positive class.}',r'\label{tab:main}',r'\centering',r'\begin{tabular}{lccccc}',r'\hline',r'Model & CV accuracy & CV F1 (mean $\pm$ SD) & Cross-show accuracy & Cross-show F1 & Cross-show macro-F1 \\',r'\hline']
for model,label in models:
 cv=a.loc[model];cs=b.loc[model];c=f'{cv.f1_mean:.3f} $\\pm$ {cv.f1_std:.3f}';f=num(cs.f1)
 if model=='NET|Multimodal gated (+mismatch)':c=r'\textbf{'+c+'}'
 if model=='LR|Speech-only':f=r'\textbf{'+f+'}'
 lines.append(f'{label} & {num(cv.acc_mean)} & {c} & {num(cs.acc)} & {f} & {num(cs.macro_f1)} '+r'\\')
lines += [r'\hline',r'\end{tabular}',r'\end{table*}'];(out/'table-main.tex').write_text('\n'.join(lines)+'\n')
abl=[('LR|Multimodal (+mismatch)','All blocks'),('LR|Multimodal -mismatch','Without mismatch'),('LR|Multimodal -prosody','Without prosody'),('LR|Multimodal -WavLM','Without WavLM'),('LR|Prosody-only','Prosody only'),('LR|Mismatch-only','Sentiment, emotion, mismatch'),('LR|Text+Context','Text with context')]
lines=[r'\begin{table}[t]',r'\caption{LR ablations. CV F1 is the mean over 15 folds. Cross-show F1 is pooled.}',r'\label{tab:ablations}',r'\centering',r'\begin{tabular}{lcc}',r'\hline',r'Feature configuration & CV F1 & Cross-show F1 \\',r'\hline']
for model,label in abl:lines.append(f'{label} & {a.loc[model,"f1_mean"]:.3f} & {b.loc[model,"f1"]:.3f} '+r'\\')
lines +=[r'\hline',r'\end{tabular}',r'\end{table}'];(out/'table-ablations.tex').write_text('\n'.join(lines)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.1,2.6),sharey=True)
labels=[v for _,v in models];ix=np.arange(len(models))
for ax,col,title in zip(axs,['#244f65','#b86b40'],['Random utterance CV','Pooled held-out shows']):
 vals=[a.loc[m,'f1_mean'] if ax is axs[0] else b.loc[m,'f1'] for m,_ in models]
 errs=[a.loc[m,'f1_std'] for m,_ in models] if ax is axs[0] else None
 ax.barh(ix,vals,xerr=errs,color=col,height=.65,error_kw={'elinewidth':.7,'capsize':2})
 ax.set_xlim(.45,.85);ax.set_title(title,fontsize=10);ax.set_xlabel('Sarcastic-class F1');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
axs[0].set_yticks(ix,labels);axs[0].invert_yaxis();fig.tight_layout(pad=.6)
fig.savefig(out/'figures/model-comparison.pdf',bbox_inches='tight');fig.savefig(out/'figures/model-comparison.png',dpi=180,bbox_inches='tight')
print('Generated tables and figure directly from fresh-run CSVs')
