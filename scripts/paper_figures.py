"""Publication figures derived only from saved held-out predictions and metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/paper'

def build_figures():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':8,'axes.labelsize':8,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False})
    data=ROOT/'results/mustardpp';summary=pd.read_csv(data/'summary.csv');pred=pd.read_csv(data/'predictions.csv')
    matrices={}
    for training in ['MUStARD','MUStARD++']:
        part=pred[(pred.training==training)&(pred['mode']=='Text + speech')].groupby('id').agg(label=('label','first'),score=('score','mean'))
        row=summary[(summary.training==training)&(summary['mode']=='Text + speech')].iloc[0]
        for metric,fn in [('accuracy',accuracy_score),('f1',f1_score),('precision',precision_score),('recall',recall_score)]:
            assert np.isclose(fn(part.label,part.score>=.5),row[metric])
        matrices[training]=confusion_matrix(part.label,part.score>=.5,labels=[0,1]).tolist()
    modes=['Text only','Speech reference','Text + speech'];conditions=['MUStARD','MUStARD++ size-matched','MUStARD++']
    fig,ax=plt.subplots(figsize=(3.54,2.35));x=np.arange(3)
    for i,condition in enumerate(conditions):
        values=[summary[(summary.training==condition)&(summary['mode']==m)].f1.iloc[0] for m in modes]
        ax.bar(x+(i-1)*.24,values,.23,label=condition,color=['#a3a3a3','#c7d8e6','#294c68'][i])
    ax.set_xticks(x,['Text','Speech ref.','Text + speech']);ax.set_ylabel('Sarcastic-class F1');ax.set_ylim(0,.9);ax.legend(loc='upper left',frameon=False,fontsize=6.5);fig.tight_layout();fig.savefig(OUT/'f1_comparison.png',dpi=320);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(3.54,1.95))
    for ax,(name,counts) in zip(axes,matrices.items()):
        matrix=np.array(counts);ax.imshow(matrix,cmap='Blues',vmin=0,vmax=344)
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',fontsize=9,color='white' if v>180 else 'black')
        ax.set_title(name,fontsize=8);ax.set_xticks([0,1],['Non-sarc.','Sarc.'],fontsize=7);ax.set_yticks([0,1],['Non-sarc.','Sarc.'],fontsize=7);ax.set_xlabel('Predicted');ax.set_ylabel('Reference')
    fig.tight_layout(pad=.4);fig.savefig(OUT/'confusion_matrices.png',dpi=320);plt.close(fig)
    ci=pd.read_csv(data/'paired_bootstrap.csv');ci=ci[ci.comparison=='MUStARD++ minus MUStARD'].set_index('mode').loc[modes]
    fig,ax=plt.subplots(figsize=(3.54,1.8));d=ci.delta_f1.to_numpy();ax.errorbar(d,np.arange(3),xerr=np.vstack([d-ci.ci_low,ci.ci_high-d]),fmt='o',color='#294c68',capsize=3)
    ax.axvline(0,color='#777',lw=.8,ls='--');ax.set_yticks(range(3),['Text','Speech ref.','Text + speech']);ax.set_xlabel('F1 difference: MUStARD++ minus MUStARD');fig.tight_layout();fig.savefig(OUT/'paired_intervals.png',dpi=320);plt.close(fig)
    emotion=json.loads((ROOT/'results/emotion/evaluation.json').read_text());matrix=np.array(emotion['confusion_matrix'])
    assert matrix.sum()==1202
    fig,ax=plt.subplots(figsize=(3.54,3.15));ax.imshow(matrix,cmap='Blues')
    for (i,j),v in np.ndenumerate(matrix):
        if v:ax.text(j,i,str(v),ha='center',va='center',fontsize=5.5,color='white' if v>matrix.max()*.55 else 'black')
    ax.set_xticks(range(9),emotion['labels'],rotation=65,ha='right',fontsize=6);ax.set_yticks(range(9),emotion['labels'],fontsize=6);ax.set_xlabel('Predicted surface emotion');ax.set_ylabel('Annotated surface emotion');fig.tight_layout(pad=.3);fig.savefig(OUT/'emotion_confusion.png',dpi=320);plt.close(fig)
    (OUT/'figure_data.json').write_text(json.dumps({'sarcasm_confusion_matrices':matrices,'emotion_counts':emotion['counts'],'note':'Figures derive from saved out-of-fold predictions; no epoch loss curves or training accuracy invented.'},indent=2)+'\n')
    return OUT
if __name__=='__main__':build_figures()
