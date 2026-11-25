"""Synthetic speech integration/stability checks. No ground-truth emotion labels."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from inference import Predictor,prepare_audio
TEXTS=['This project is pretty good.', 'Wonderful, I have another meeting.',
       'The meeting starts at nine tomorrow.', 'Thank you for helping me today.',
       'I cannot believe that happened again.', 'This is exactly what I needed.',
       'Please send the report when it is ready.', 'I am waiting for the next train.']

def main():
    out=ROOT/'truetone/generated-check';out.mkdir(parents=True,exist_ok=True)
    model=Predictor();rng=np.random.default_rng(42);checks=[]
    for voice in ['Albert','Samantha']:
        for i,text in enumerate(TEXTS):
            file=out/f'{voice}-{i}.aiff'
            subprocess.run(['say','-v',voice,'-r','155','-o',str(file),text],check=True)
            x=prepare_audio(file);rms=np.sqrt(np.mean(x*x))
            for condition,wave in [('original',x),('noise_20db',np.clip(x+rng.normal(0,rms/10,x.shape),-1,1))]:
                for mode in ['Existing four-class','MUStARD++ nine-class (experimental)']:
                    start=time.perf_counter()
                    score,words,duration,description,emotions,contrast=model.predict((16000,wave),text,emotion_mode=mode)
                    assert np.isfinite(score) and 0<=score<=1 and np.isclose(sum(emotions.values()),1)
                    assert len(emotions)==(4 if mode.startswith('Existing') else 9)
                    checks.append(dict(voice=voice,sentence=i,condition=condition,mode=mode,score=score,
                                       emotion=emotions,seconds=time.perf_counter()-start))
                    print(voice,i,condition,mode,flush=True)
    report={'note':'Synthesized by macOS say. Reference transcript supplied. No human emotion or sarcasm ground truth; not accuracy evidence. Two voices are not independent human speakers. Audio stays local.',
            'checks':checks}
    public=ROOT/'results/app-audit/generated_audio.json'
    public.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(checks),'finite complete inference results',flush=True)
if __name__=='__main__':main()
