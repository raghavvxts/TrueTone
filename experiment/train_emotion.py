"""Train audio-only MUStARD++ surface emotion; evaluate with held-out shows."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,log_loss,confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

LABELS=['Anger','Disgust','Excitement','Fear','Frustration','Happiness','Neutral','Sadness','Surprise']
BLOCKS=['wavlm','speech_emo','prosody']

def features(blocks):
    # Exclude the three transcript-count features from this audio-only head.
    prosody=np.delete(blocks['prosody'],[24,25,26],axis=1)
    x=np.concatenate([blocks['wavlm'],blocks['speech_emo'],prosody],axis=1)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite emotion features')
    return x

def classifier():
    return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=42))

def main():
    rows=pd.read_csv(ROOT/'truetone/pp/metadata.csv')
    z=np.load(ROOT/'truetone/pp/features.npz')
    x=features(z); y=rows.Explicit_Emotion.to_numpy(); groups=rows.show.to_numpy()
    assert len(rows)==1202 and set(y)==set(LABELS) and rows.id.is_unique
    probabilities=np.zeros((len(y),len(LABELS))); baseline=z['speech_emo']
    old=np.array(['Neutral','Happiness','Anger','Sadness'])[baseline.argmax(axis=1)]
    majority=np.empty(len(y),dtype=object); folds=[]
    for train,test in LeaveOneGroupOut().split(x,y,groups):
        text=rows.utterance.str.lower().str.replace(r'[^a-z0-9]','',regex=True)
        train=train[~text.iloc[train].isin(set(text.iloc[test])).to_numpy()]
        model=classifier().fit(x[train],y[train]); p=model.predict_proba(x[test])
        for j,label in enumerate(model.classes_): probabilities[test,LABELS.index(label)]=p[:,j]
        majority[test]=pd.Series(y[train]).mode()[0]
        folds.append(dict(show=groups[test][0],train=len(train),test=len(test),accuracy=accuracy_score(y[test],model.predict(x[test]))))
        print(folds[-1],flush=True)
    predicted=np.array(LABELS)[probabilities.argmax(axis=1)]
    report={'task':'MUStARD++ Explicit_Emotion / audio only','labels':LABELS,'counts':rows.Explicit_Emotion.value_counts().to_dict(),'split':'leave-one-show-out; fixed C=0.1; no hyperparameter search','folds':folds,'metrics':{}}
    for name,pred in [('trained',predicted),('frozen_four_class',old),('majority',majority)]:
        report['metrics'][name]={'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,labels=LABELS,average='macro',zero_division=0)}
    probabilities/=probabilities.sum(axis=1,keepdims=True)
    report['log_loss']=log_loss(y,probabilities,labels=LABELS)
    report['confusion_matrix']=confusion_matrix(y,predicted,labels=LABELS).tolist()
    out=ROOT/'results/emotion';out.mkdir(exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.DataFrame(dict(id=rows.id,show=groups,expected=y,predicted=predicted,baseline=old)).to_csv(out/'held_out_predictions.csv',index=False)
    model=classifier().fit(x,y)
    artifact=ROOT/'models/truetone_emotion.joblib'
    from inference import feature_signature
    joblib.dump(dict(model=model,labels=LABELS,blocks=BLOCKS,feature_signature=feature_signature(),target='Explicit_Emotion',training_examples=len(y)),artifact)
    report['artifact_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['metrics'],indent=2),flush=True)
if __name__=='__main__':main()
