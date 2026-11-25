"""Private, speaker-disjoint microphone adaptation; never overwrites serving weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from inference import Predictor, prepare_audio, feature_runtime, feature_signature, predict_features
from experiment.train_emotion import LABELS, features as emotion_features

COLUMNS=['id','audio','speaker','split','consent','transcript','sarcasm_rater1','sarcasm_rater2','emotion_rater1','emotion_rater2']

def validate(frame, directory):
    if set(COLUMNS)-set(frame.columns):raise ValueError('Missing manifest columns')
    if (len(frame)==0 or frame[COLUMNS].isna().any().any() or
            frame[COLUMNS].astype(str).apply(lambda col:col.str.strip().eq('')).any().any()):
        raise ValueError('Complete every manifest field')
    if not frame.id.is_unique:raise ValueError('Recording IDs must be unique')
    if not frame.consent.astype(str).str.lower().eq('yes').all():raise ValueError('Explicit consent=yes required for every recording')
    if set(frame.split)!={'train','calibration','test'}:raise ValueError('Use train, calibration and test splits')
    if frame.groupby('speaker').split.nunique().max()!=1:raise ValueError('A speaker must occur in only one split')
    # No duplicate waveform may cross splits, even under a different filename.
    seen={}
    for row in frame.itertuples():
        path=(directory/row.audio).resolve()
        if not path.is_relative_to(directory.resolve()):raise ValueError('Keep audio inside the private study directory')
        x=prepare_audio(path)
        digest=hashlib.sha256(x.tobytes()).hexdigest()
        if digest in seen and seen[digest]!=row.split:raise ValueError('Duplicate audio crosses splits')
        seen[digest]=row.split
    for split, part in frame.groupby('split'):
        if part.speaker.nunique()<2:raise ValueError(f'{split}: at least two independent speakers required')
        if set(part.sarcasm_rater1.astype(str))!={'0','1'}:raise ValueError(f'{split}: both sarcasm classes required')
    for a,b in [('sarcasm_rater1','sarcasm_rater2'),('emotion_rater1','emotion_rater2')]:
        if not frame[a].astype(str).eq(frame[b].astype(str)).all():raise ValueError('Rater disagreement: adjudicate before training; do not silently discard')
    if not set(frame.emotion_rater1)<=set(LABELS):raise ValueError('Unknown surface emotion label')
    if frame.transcript.str.strip().eq('').any():raise ValueError('Human-checked transcripts required')

def augment(samples,seed):
    rng=np.random.default_rng(seed)
    rms=float(np.sqrt(np.mean(samples*samples)))
    return [('original',samples),('quiet',samples*.25),
            ('noise_20db',np.clip(samples+rng.normal(0,rms/10,samples.shape),-1,1))]

def calibrated(probabilities, temperature):
    logits=np.log(np.clip(probabilities,1e-12,1))
    return softmax(logits/temperature,axis=1)

def temperature(probabilities,y,labels):
    return float(minimize_scalar(lambda t:log_loss(y,calibrated(probabilities,t),labels=labels),bounds=(.2,10),method='bounded').x)

def metrics(y,p,labels):
    pred=np.array(labels)[p.argmax(axis=1)]; confidence=p.max(axis=1);correct=(pred==np.asarray(y))
    ece=0.
    for lo in np.arange(0,1,.1):
        mask=(confidence>=lo)&(confidence<(lo+.1) if lo<.9 else confidence<=1)
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    return dict(accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=labels,average='macro',zero_division=0),
                log_loss=log_loss(y,p,labels=labels),ece_10_bins=float(ece),
                confusion_matrix=confusion_matrix(y,pred,labels=labels).tolist())

def extract(samples,transcript,bundle,temp):
    path=Path(temp)/'input.wav';sf.write(path,samples,16000,subtype='FLOAT')
    r=feature_runtime(temp);c=bundle['config']
    wavlm=r['wavlm_embed']([str(path)],c['speech_model'])
    emo=r['speech_emotion']([str(path)],c['emotion_model'])
    import re
    pro=r['prosody_feats'](str(path),len(re.findall(r"[A-Za-z0-9']+",transcript)))
    text=r['embed_text']([transcript],c['text_model']);sent=r['text_sentiment']([transcript],c['sentiment_model'])
    vt=float(sent[0,2]-sent[0,0]);vs=float(emo[0]@np.array([r['emo_valence'](s) for s in bundle['emotion_labels']]))
    return dict(wavlm=wavlm,speech_emo=emo,prosody=np.array([[pro[n] for n in bundle['prosody_names']]]),text_emb=text,text_sent=sent,
                mismatch=np.array([[vt,vs,vt-vs,abs(vt-vs),vt*vs,float(vt*vs<0 and abs(vt)>.1 and abs(vs)>.1)]]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'truetone/microphone/manifest.csv')
    parser.add_argument('--init',action='store_true');args=parser.parse_args()
    directory=args.manifest.resolve().parent;directory.mkdir(parents=True,exist_ok=True)
    if not directory.is_relative_to(ROOT/'truetone'):
        parser.error('Keep private recordings, labels and candidates under ignored truetone/')
    if args.init:
        if args.manifest.exists():parser.error('Manifest already exists; will not overwrite labels')
        pd.DataFrame(columns=COLUMNS).to_csv(args.manifest,index=False);print(args.manifest);return
    if not args.manifest.exists():parser.error('Run --init and collect labelled recordings first')
    if (directory/'evaluation.json').exists():parser.error('Test set already evaluated. Do not tune and reuse it; collect a new independent test set.')
    frame=pd.read_csv(args.manifest,dtype=str,keep_default_na=False)
    validate(frame,directory)
    serving=Predictor().bundle
    dataset=pd.read_csv(ROOT/'truetone/pp/metadata.csv');base=dict(np.load(ROOT/'truetone/pp/features.npz'))
    records=[]
    with tempfile.TemporaryDirectory(prefix='truetone-adapt-') as temp:
        for i,row in enumerate(frame.itertuples()):
            samples=prepare_audio(directory/row.audio)
            variants=augment(samples,42+i) if row.split=='train' else [('original',samples)]
            for variant,wave in variants:
                records.append((row,variant,extract(wave,row.transcript,serving,temp)))
                print(row.id,variant,flush=True)
    blocks=list(serving['models']['Text + speech']['blocks'])
    def matrix(f,task):
        return emotion_features(f) if task=='emotion' else np.concatenate([f[k]/np.sqrt(f[k].shape[1]) for k in blocks],axis=1)
    report={'note':'Single locked test evaluation; any later tuning requires a fresh speaker test set. Reference transcripts used; not end-to-end ASR accuracy.',
            'feature_signature':feature_signature(),'splits':frame.groupby('split').size().to_dict(),'tasks':{}}
    candidates={}
    for task in ['sarcasm','emotion']:
        field='sarcasm_rater1' if task=='sarcasm' else 'emotion_rater1'
        labels=[0,1] if task=='sarcasm' else LABELS
        convert=int if task=='sarcasm' else str
        bx=matrix(base,task); by=dataset.label.to_numpy() if task=='sarcasm' else dataset.Explicit_Emotion.to_numpy()
        parts={}
        for split in ['train','calibration','test']:
            selected=[(row,feat) for row,_,feat in records if row.split==split]
            parts[split]=(np.concatenate([matrix(f,task) for _,f in selected]),np.array([convert(getattr(r,field)) for r,_ in selected]))
        # All augmentation stays on training speakers; calibration/test remain original.
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,random_state=42))
        model.fit(np.concatenate([bx,parts['train'][0]]),np.concatenate([by,parts['train'][1]]))
        assert list(model.classes_)==labels
        calx,caly=parts['calibration'];testx,testy=parts['test']
        t=temperature(model.predict_proba(calx),caly,labels)
        raw=model.predict_proba(testx)
        test_features=[feat for row,_,feat in records if row.split=='test']
        if task=='sarcasm':
            scores=np.array([predict_features(serving,f,'Text + speech') for f in test_features])
            previous=np.column_stack([1-scores,scores])
        else:
            previous=np.full((len(test_features),len(labels)),1e-12)
            mapping={'neu':'Neutral','hap':'Happiness','ang':'Anger','sad':'Sadness'}
            for i,f in enumerate(test_features):
                for j,label in enumerate(serving['emotion_labels']):previous[i,labels.index(mapping[label])]=f['speech_emo'][0,j]
            previous/=previous.sum(axis=1,keepdims=True)
        report['tasks'][task]={'labels':labels,'temperature':t,'existing_model':metrics(testy,previous,labels),'uncalibrated':metrics(testy,raw,labels),'calibrated':metrics(testy,calibrated(raw,t),labels)}
        candidates[task]={'model':model,'temperature':t,'labels':labels,'blocks':blocks if task=='sarcasm' else ['wavlm','speech_emo','prosody']}
    # Private candidates only. No automatic promotion based on test scores.
    joblib.dump({'tasks':candidates,'feature_signature':feature_signature(),'dataset':'MUStARD++ + consented microphone training speakers'},directory/'candidate.joblib')
    (directory/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved private candidate and evaluation; serving model unchanged.')
if __name__=='__main__':main()
