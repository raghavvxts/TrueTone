"""Update the existing final deck in place, preserving its layouts and artwork.

Requires matplotlib and defusedxml. Result charts are derived from saved CSVs.
"""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from defusedxml import minidom

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / 'presentations/TrueTone-PID-02.pptx'
DATA = ROOT / 'results/mustardpp'
PAPER, GREEN, ORANGE, SAGE = '#f3efe4', '#173f35', '#c45e2b', '#93a58c'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'text.color': GREEN,
                     'axes.labelcolor': GREEN, 'xtick.color': GREEN, 'ytick.color': GREEN})


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width/160, height/160), dpi=160)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=7)
    return fig, ax


def png(fig):
    out = BytesIO()
    fig.savefig(out, format='png', transparent=True, dpi=160)
    plt.close(fig)
    return out.getvalue()


def charts():
    summary = pd.read_csv(DATA/'summary.csv')
    def value(training, mode, metric='f1'):
        return float(summary.loc[(summary.training==training)&(summary['mode']==mode), metric].iloc[0])
    result = {}
    fig, ax = canvas(575,409)
    wedges, _ = ax.pie([601,601], colors=[ORANGE,SAGE], startangle=90,
                      wedgeprops={'width':.35,'edgecolor':PAPER})
    ax.text(0,0,'50 / 50',ha='center',va='center',fontsize=16,fontweight='bold')
    ax.legend(wedges,['Sarcastic: 601','Non-sarcastic: 601'], loc='lower center',
              bbox_to_anchor=(.5,-.12),ncol=1,frameon=False,fontsize=9)
    fig.subplots_adjust(top=.99,bottom=.20,left=.06,right=.94)
    result['image2.png']=png(fig)
    fig, ax = canvas(1310,816)
    names=['Big Bang Theory','Friends','Silicon Valley','Golden Girls','Sarcasmoholics']
    total=np.array([716,354,78,40,14]);pos=np.array([360,151,37,39,14]);y=np.arange(5)
    ax.barh(y-.18,total,.32,color=SAGE,label='All targets')
    ax.barh(y+.18,pos,.32,color=ORANGE,label='Sarcastic')
    for yy,t,p in zip(y,total,pos):
        ax.text(t+10,yy-.18,str(t),va='center',fontsize=13)
        ax.text(p+10,yy+.18,str(p),va='center',fontsize=13)
    ax.set_yticks(y,names,fontsize=14);ax.invert_yaxis();ax.set_xlim(0,820)
    ax.xaxis.grid(True,alpha=.13);ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2,frameon=False,fontsize=13)
    ax.set_title('Target utterances by show',fontsize=16,pad=62)
    fig.subplots_adjust(left=.30,right=.94,bottom=.24,top=.71)
    result['image3.png']=png(fig)
    fig, ax = canvas(958,777)
    modes=['Text only','Speech reference','Text + speech'];y=np.arange(3)
    for offset,training,color in [(-.18,'MUStARD',SAGE),(.18,'MUStARD++',ORANGE)]:
        values=[value(training,m) for m in modes]
        ax.barh(y+offset,values,.30,color=color,label=training)
        for yy,v in zip(y+offset,values):ax.text(v+.008,yy,f'{v:.3f}',va='center',fontsize=13)
    ax.set_yticks(y,modes,fontsize=12);ax.invert_yaxis();ax.set_xlim(.5,.85)
    ax.set_xticks([.5,.6,.7,.8]);ax.set_xlabel('Sarcastic-class F1',fontsize=12)
    ax.xaxis.grid(True,alpha=.13);ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=2,frameon=False,fontsize=11)
    ax.set_title('Same 688 shared targets',fontsize=15,pad=58)
    fig.subplots_adjust(left=.34,right=.90,bottom=.20,top=.68)
    result['image4.png']=png(fig)
    fig, ax = canvas(958,777)
    labels=['F1','Macro-F1','Accuracy','Precision','Recall'];vals=[value('MUStARD++','Text + speech',m) for m in ['f1','macro_f1','accuracy','precision','recall']]
    y=np.arange(5);ax.barh(y,vals,.60,color=GREEN)
    for yy,v in zip(y,vals):ax.text(v+.008,yy,f'{v:.3f}',va='center',fontsize=13)
    ax.set_yticks(y,labels,fontsize=12);ax.invert_yaxis();ax.set_xlim(.5,.87)
    ax.set_xticks([.5,.6,.7,.8]);ax.xaxis.grid(True,alpha=.13);ax.set_axisbelow(True)
    ax.set_title('MUStARD++: Text + speech',fontsize=15,pad=30)
    ax.set_xlabel('Shared-target scores',fontsize=12)
    fig.subplots_adjust(left=.28,right=.90,bottom=.20,top=.74)
    result['image5.png']=png(fig)
    fig,ax=canvas(1847,926)
    conditions=['MUStARD','MUStARD++','MUStARD++ size-matched'];vals=[value(c,'Text + speech') for c in conditions]
    ax.bar(np.arange(3),vals,color=[SAGE,ORANGE,GREEN],width=.52)
    for i,v in enumerate(vals):ax.text(i,v+.006,f'{v:.3f}',ha='center',fontsize=24,fontweight='bold')
    ax.set_xticks(range(3),['MUStARD baseline','MUStARD++ final','Size-matched\nMUStARD++'],fontsize=16)
    ax.set_ylim(.65,.80);ax.set_yticks([.65,.70,.75,.80]);ax.tick_params(axis='y',labelsize=15)
    ax.yaxis.grid(True,alpha=.16);ax.set_axisbelow(True);ax.set_ylabel('Text + speech F1',fontsize=17)
    interval=pd.read_csv(DATA/'paired_bootstrap.csv').query("mode == 'Text + speech' and comparison == 'MUStARD++ minus MUStARD'").iloc[0]
    ax.set_title(f'Observed gain: {interval.delta_f1:+.3f} F1\n95% group-bootstrap CI [{interval.ci_low:+.3f}, {interval.ci_high:+.3f}]',fontsize=17,pad=25)
    fig.text(.5,.07,'Same shared targets. The interval includes zero.',ha='center',fontsize=16)
    fig.subplots_adjust(left=.15,right=.92,top=.66,bottom=.28)
    result['image6.png']=png(fig)
    fig,ax=canvas(1220,801)
    held=pd.read_csv(DATA/'held_out_show.csv');shows=['BBT','FRIENDS','SV','GOLDENGIRLS','SARCASMOHOLICS'];labels=['Big Bang Theory','Friends','Silicon Valley','Golden Girls','Sarcasmoholics'];y=np.arange(5)
    for offset,training,color in [(-.17,'MUStARD',SAGE),(.17,'MUStARD++',GREEN)]:
        vals=[held.loc[(held.show==s)&(held.training==training)&(held['mode']=='Text + speech'),'f1'].iloc[0] for s in shows]
        ax.barh(y+offset,vals,.29,color=color,label=training)
        for yy,v in zip(y+offset,vals):ax.text(v+.015,yy,f'{v:.3f}',va='center',fontsize=13)
    ax.set_yticks(y,labels,fontsize=13);ax.invert_yaxis();ax.set_xlim(0,1.1)
    ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('Held-out-show F1: Text + speech',fontsize=14)
    ax.xaxis.grid(True,alpha=.14);ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.13),ncol=2,frameon=False,fontsize=13)
    fig.subplots_adjust(left=.30,right=.95,top=.80,bottom=.16)
    result['image7.png']=png(fig)
    return result


UPDATES = {
3:{1:'Two comparisons test what speech contributes and what the expanded training set changes.',6:'Does MUStARD++ improve detection over the MUStARD training baseline?',7:'MUStARD  →  MUStARD++'},
4:{2:'1,202',4:'601 sarcastic · 601 non-sarcastic'},
6:{4:'1,202 targets',5:'FINAL REFIT',6:'Local app weights',8:'TRAINING TO APP'},
7:{2:'Repeated grouped 5-fold CV',3:'3 repeats · seeds 42, 43, 44',4:'688 shared test targets',5:'Exact duplicate groups kept apart',8:'Hold out each of the five shows in turn',9:'1,202 total test targets',10:'Class imbalance limits interpretation'},
8:{1:'Matched grouped evaluation · average 3 out-of-fold probabilities per clip'},
9:{0:'DOES MORE TRAINING DATA HELP?',1:'A modest observed gain; the confidence interval includes zero.'},
10:{0:'GENERALISING TO UNSEEN SHOWS',1:'READ WITH CAUTION',2:'39 / 40',3:'Golden Girls targets',4:'are labeled sarcastic',5:'14 / 14',6:'Sarcasmoholics: all sarcastic',7:'High F1 on a skewed show does not establish reliable generalisation.'},
11:{2:'1  Text + speech outperforms text alone on the shared-target test.',3:'2  MUStARD++ has a modest observed gain; uncertainty includes zero.',4:'3  Final app weights use all 1,202 MUStARD++ targets.',7:'• Shared-show folds retain domain overlap; five held-out shows are uneven',8:'• Frozen sentiment and emotion models come from different domains'},
12:{6:'CSV results + baseline archive',7:'IEEE paper PDF + final presentation',11:'3  Official MUStARD++ annotations — github.com/cfiltnlp/MUStARD_Plus_Plus'}
}


def main():
    media=charts()
    with ZipFile(DECK) as source:
        entries={item.filename:(item,source.read(item.filename)) for item in source.infolist()}
    for i,changes in UPDATES.items():
        name=f'ppt/slides/slide{i}.xml';info,data=entries[name]
        doc=minidom.parseString(data);texts=doc.getElementsByTagName('a:t')
        for index,value in changes.items():
            node=texts[index]
            for child in list(node.childNodes):node.removeChild(child)
            node.appendChild(doc.createTextNode(value))
        if i==4:
            # The original 112.5pt count wrapped "690" across lines in LibreOffice.
            # Keep its position and box, fit the new count on one line.
            parent=texts[2].parentNode
            parent.getElementsByTagName('a:rPr')[0].setAttribute('sz','7200')
        entries[name]=(info,doc.toxml(encoding='UTF-8'))
    for name,data in media.items():
        key='ppt/media/'+name;entries[key]=(entries[key][0],data)
    temp=DECK.with_suffix('.tmp.pptx')
    with ZipFile(temp,'w') as target:
        for info,data in entries.values():target.writestr(info,data)
    with ZipFile(temp) as check:
        assert check.testzip() is None
        for name in check.namelist():
            if name.endswith('.xml'):minidom.parseString(check.read(name))
    temp.replace(DECK)
    print('Updated final PPTX, preserving slide geometry, layouts, cover and relationships.')


if __name__=='__main__':main()
