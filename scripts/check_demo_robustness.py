"""Local playback smoke checks, not generalization accuracy."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
from inference import Predictor

def main():
    model=Predictor(); manifest=json.loads((ROOT/'truetone/demos/manifest.json').read_text())
    rng=np.random.default_rng(42);results=[]
    for row in manifest:
        x,rate=sf.read(ROOT/'truetone/demos'/row['file']);rms=np.sqrt(np.mean(x*x))
        for name,wave in [('original',x),('quiet',x*.25),('noise_20db',np.clip(x+rng.normal(0,rms/10,x.shape),-1,1))]:
            start=time.perf_counter()
            p,words,duration,voice,emotions,contrast=model.predict((rate,wave),row['transcript'])
            assert np.isfinite(p) and 0<=p<=1 and np.isclose(sum(emotions.values()),1)
            results.append(dict(id=row['id'],condition=name,reference_label=row['label'],score=p,predicted=int(p>=.5),seconds=time.perf_counter()-start))
            print(results[-1],flush=True)
    out=ROOT/'results/app-audit';out.mkdir(exist_ok=True)
    (out/'demo_robustness.json').write_text(json.dumps({'note':'Training-set demos with reference transcripts; stability check only, not accuracy. Noise is synthetic, not microphone ground truth.','checks':results},indent=2)+'\n')
if __name__=='__main__':main()
