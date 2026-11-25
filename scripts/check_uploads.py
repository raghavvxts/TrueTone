"""Generate speech files and upload them through the running Gradio app API.

No microphone use, human emotion labels or training-accuracy claims.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import imageio_ffmpeg
import numpy as np
import soundfile as sf
from gradio_client import Client,handle_file
from inference import prepare_audio
from scripts.check_generated_audio import TEXTS
from experiment.check_transcripts import normalized_words,word_errors

def scores(value):
    return {item['label']:item['confidence'] for item in value.get('confidences',[])}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--url',default='http://127.0.0.1:7862');args=parser.parse_args()
    client=Client(args.url,verbose=False)
    out=ROOT/'truetone/upload-check';out.mkdir(parents=True,exist_ok=True)
    fixtures=[]
    for voice in ['Albert','Samantha']:
        for i,text in enumerate(TEXTS):
            source=out/f'{voice}-{i}.aiff'
            subprocess.run(['say','-v',voice,'-r','155','-o',str(source),text],check=True)
            x=prepare_audio(source);wav=source.with_suffix('.wav');sf.write(wav,x,16000)
            fixtures.append(dict(id=f'{voice}-{i}',voice=voice,sentence=i,text=text,path=wav,condition='clean'))
            if i==0:
                for suffix in ['mp3','m4a']:
                    path=wav.with_suffix('.'+suffix)
                    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-y','-i',str(wav),str(path)],check=True)
                    fixtures.append(dict(id=f'{voice}-{i}-{suffix}',voice=voice,sentence=i,text=text,path=path,condition='compressed'))
    checks=[]
    for row in fixtures:
        for mode in ['Existing four-class','MUStARD++ nine-class (experimental)']:
            start=time.perf_counter()
            result=client.predict(handle_file(str(row['path'])),'','Text + speech',mode,api_name='/analyze')
            sarcasm=scores(result[0]);emotion=scores(result[4]);accepted=bool(sarcasm)
            item=dict(id=row['id'],voice=row['voice'],sentence=row['sentence'],format=row['path'].suffix[1:],condition=row['condition'],mode=mode,accepted=accepted,seconds=time.perf_counter()-start)
            if accepted:
                assert set(sarcasm)=={'Sarcastic','Not sarcastic'}
                assert np.isclose(sum(sarcasm.values()),1) and np.isclose(sum(emotion.values()),1)
                assert len(emotion)==(4 if mode.startswith('Existing') else 9)
                expected=normalized_words(row['text']);actual=normalized_words(result[1])
                item.update(reference=row['text'],transcript=result[1],reference_words=len(expected),word_errors=word_errors(expected,actual),sarcasm_score=sarcasm['Sarcastic'],top_emotion=max(emotion,key=emotion.get))
            else:
                assert 'Recording needs attention' in result[2]
                item['recovery_message']=result[2]
            checks.append(item)
            print(row['id'],mode,'accepted' if accepted else 'rejected with instructions',flush=True)
    unique=[r for r in checks if r['mode']=='Existing four-class']
    accepted=[r for r in unique if r['accepted']]
    reference_words=sum(r['reference_words'] for r in accepted)
    report={'note':'Actual file uploads to Gradio; blank transcript triggers ASR. Synthesized speech has known words but no validated sarcasm/emotion labels. Not human microphone accuracy; no model training. WER excludes rejected uploads and counts each file once, not each mode.',
            'summary':dict(files=len(fixtures),requests=len(checks),accepted=sum(r['accepted'] for r in checks),rejected=sum(not r['accepted'] for r in checks),accepted_file_reference_words=reference_words,accepted_file_word_errors=sum(r['word_errors'] for r in accepted),accepted_file_wer=sum(r['word_errors'] for r in accepted)/reference_words if reference_words else None),
            'checks':checks}
    (ROOT/'results/app-audit/upload_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary']),flush=True)
if __name__=='__main__':main()
