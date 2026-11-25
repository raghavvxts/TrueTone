"""Prepare local showcase clips by annotation, never by prediction correctness."""
import json
from pathlib import Path
import re
import shutil
import pandas as pd
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
# Keep short classroom examples; selection never examines model scores.
EXCLUDE=re.compile(r'\b(fuck|shit|bitch|sex|pervert|ass|damn|hell|bastard|kill|dead|fat|bed|pants|genital|cheating|cop a feel)\b',re.I)

def select_examples(rows):
    examples=[]
    for (emotion,label),part in rows.groupby(['Explicit_Emotion','label'],sort=True):
        for row in part.sort_values('id').itertuples():
            info=sf.info(row.wav)
            if not 2<=info.duration<=10 or len(row.utterance)>160 or EXCLUDE.search(row.utterance):continue
            examples.append(dict(id=row.id,file=f'{row.id}.wav',transcript=row.utterance,label=int(label),
                                 emotion=emotion,show=row.show,duration=info.duration,source=row.wav,
                                 note='Seen in final refit; dataset annotation is not a model prediction'))
            break
    return examples

def main():
    rows=pd.read_csv(ROOT/'truetone/pp/metadata.csv')
    out=ROOT/'truetone/demos';out.mkdir(exist_ok=True)
    examples=select_examples(rows)
    assert len(examples)==18 and {x['label'] for x in examples}=={0,1}
    for row in examples:shutil.copyfile(row.pop('source'),out/row['file'])
    (out/'manifest.json').write_text(json.dumps(examples,indent=2)+'\n')
    print('Prepared',len(examples),'local MUStARD++ clips: both sarcasm labels across nine surface emotions.')
if __name__=='__main__':main()
