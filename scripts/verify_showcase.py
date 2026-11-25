"""Verify all prepared showcase recordings; never suppress prediction mismatches."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from inference import Predictor

def main():
    rows=json.loads((ROOT/'truetone/demos/manifest.json').read_text());model=Predictor();checks=[]
    for row in rows:
        path=ROOT/'truetone/demos'/row['file']
        for mode in ['Existing four-class','MUStARD++ nine-class (experimental)']:
            score,words,duration,voice,emotions,contrast=model.predict(path,row['transcript'],emotion_mode=mode)
            assert words==row['transcript'] and 1<=duration<=30 and np.isfinite(score)
            assert np.isclose(sum(emotions.values()),1)
            checks.append(dict(id=row['id'],reference_sarcasm=row['label'],reference_emotion=row['emotion'],mode=mode,
                               score=score,predicted_sarcasm=int(score>=.5),predicted_emotion=max(emotions,key=emotions.get)))
        print(row['id'],'verified both modes',flush=True)
    report={'note':'18 labelled MUStARD++ training examples, chosen by annotation/ID/duration/content, not prediction success. Reference transcripts supplied. Mismatches preserved. This is not held-out accuracy or ASR verification.','checks':checks}
    (ROOT/'results/app-audit/showcase_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(checks),'requests',flush=True)
if __name__=='__main__':main()
