"""Prepare local dataset playback examples, chosen without checking model predictions."""
import json
from pathlib import Path
import shutil
import pandas as pd
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]

def main():
    rows=pd.read_csv(ROOT/'truetone/pp/metadata.csv')
    out=ROOT/'truetone/demos';out.mkdir(exist_ok=True)
    examples=[]
    for label in (0,1):
        selected=0
        for row in rows[rows.label==label].sort_values('id').itertuples():
            info=sf.info(row.wav)
            if not 3<=info.duration<=10:continue
            file=f'{"sarcastic" if label else "non-sarcastic"}-{selected+1}.wav'
            shutil.copyfile(row.wav,out/file)
            examples.append(dict(id=row.id,file=file,transcript=row.utterance,label=label,show=row.show,
                                 note='Seen in final refit; playback demonstration only'))
            selected+=1
            if selected==2:break
        assert selected==2
    (out/'manifest.json').write_text(json.dumps(examples,indent=2)+'\n')
    print('Prepared four local public-dataset demos; no personal audio used.')
if __name__=='__main__':main()
