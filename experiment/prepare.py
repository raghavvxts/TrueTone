"""Prepare official MUStARD++ targets and identical frozen features for the final project."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault('HF_HOME', str(ROOT / 'truetone/hf'))
sys.path.insert(0, str(ROOT))
import imageio_ffmpeg
import joblib
import numpy as np
import pandas as pd
from inference import feature_runtime, feature_signature

OUT = ROOT / 'truetone/pp'
OUT.mkdir(parents=True, exist_ok=True)


def normalize(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())


def targets():
    raw = pd.read_csv(ROOT / 'data/mustardpp.csv', keep_default_na=False)
    rows = raw[raw.Sarcasm.astype(str).isin(['0', '1'])].copy()
    assert len(rows) == 1202 and rows.SCENE.is_unique
    rows = rows.rename(columns={'SCENE': 'id', 'SENTENCE': 'utterance', 'SHOW': 'show', 'SPEAKER': 'speaker'})
    rows['label'] = rows.Sarcasm.astype(int)
    rows['group'] = rows.utterance.map(normalize)
    return rows.reset_index(drop=True)


def extract_videos(rows):
    wanted = set(rows.id)
    video_dir = OUT / 'videos'
    video_dir.mkdir(exist_ok=True)
    for archive in sorted((ROOT / 'data/mirror/MUStARD++').glob('*.zip')):
        print('Inspecting', archive.name, flush=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                name = Path(member.filename).name
                stem = Path(name).stem
                clip_id = stem[:-2] if stem.endswith('_u') else stem
                # Extract only selected target video files; never execute archive contents.
                if clip_id in wanted and name.lower().endswith(('.mp4', '.mkv')):
                    dest = video_dir / (clip_id + '.mp4')
                    if not dest.exists():
                        dest.write_bytes(z.read(member))
    existing = ROOT / 'truetone/audio16k'
    missing = []
    paths = []
    audio_dir = OUT / 'audio16k'
    audio_dir.mkdir(exist_ok=True)
    for row in rows.itertuples():
        base = existing / f'{row.id}.wav'
        output = audio_dir / f'{row.id}.wav'
        # Shared clips use a private copy of the original audio for exact comparison.
        if base.exists():
            paths.append(str(base))
            continue
        if not output.exists():
            video = video_dir / f'{row.id}.mp4'
            if not video.exists():
                missing.append(row.id)
                paths.append('')
                continue
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-loglevel', 'error', '-y',
                            '-i', str(video), '-vn', '-ac', '1', '-ar', '16000', str(output)], check=True)
        paths.append(str(output))
    if missing:
        (OUT / 'missing_audio.json').write_text(json.dumps(missing, indent=2))
        raise RuntimeError(f'{len(missing)} target clips missing. See {OUT / "missing_audio.json"}')
    return paths


def main():
    rows = targets()
    baseline = json.loads((ROOT / 'truetone/data/sarcasm_data.json').read_text())
    bundle = joblib.load(ROOT / 'models/baseline/truetone_lr.joblib')
    assert bundle['feature_signature'] == feature_signature()
    ids = bundle['ids']
    assert ids == list(baseline)
    paths = extract_videos(rows)
    rows['wav'] = paths
    rows.to_csv(OUT / 'metadata.csv', index=False)
    old_index = {key: n for n, key in enumerate(ids)}
    feature_file = OUT / 'baseline_features.npz'
    if not feature_file.exists():
        raise RuntimeError('Run experiment/prepare_baseline.py first.')
    old = dict(np.load(feature_file))
    labels = bundle['emotion_labels']
    config = bundle['config']
    runtime = feature_runtime(OUT)
    rowcache = OUT / 'features'
    rowcache.mkdir(exist_ok=True)
    changed_text = 0
    for n, row in enumerate(rows.itertuples()):
        fingerprint = hashlib.sha256((row.id + row.utterance + feature_signature()).encode()).hexdigest()[:20]
        file = rowcache / f'{fingerprint}.npz'
        if file.exists():
            continue
        index = old_index.get(row.id)
        same_text = index is not None and ' '.join(baseline[row.id]['utterance'].split()) == ' '.join(row.utterance.split())
        features = {}
        if index is not None:
            for key in ['wavlm', 'speech_emo']:
                features[key] = old[key][index:index+1]
        else:
            features['wavlm'] = runtime['wavlm_embed']([row.wav], config['speech_model'])
            features['speech_emo'] = runtime['speech_emotion']([row.wav], config['emotion_model'])
            assert json.loads((OUT / 'emo_labels_live.json').read_text()) == labels
        if same_text:
            for key in ['text_emb', 'text_sent', 'prosody']:
                features[key] = old[key][index:index+1]
        else:
            changed_text += index is not None
            features['text_emb'] = runtime['embed_text']([row.utterance], config['text_model'])
            features['text_sent'] = runtime['text_sentiment']([row.utterance], config['sentiment_model'])
            prosody = runtime['prosody_feats'](row.wav, len(re.findall(r"[A-Za-z0-9']+", row.utterance)))
            features['prosody'] = np.array([[prosody[name] for name in bundle['prosody_names']]], dtype=np.float32)
        vt = features['text_sent'][:, 2] - features['text_sent'][:, 0]
        vs = features['speech_emo'] @ np.array([runtime['emo_valence'](s) for s in labels])
        diff = vt - vs
        features['mismatch'] = np.stack([vt, vs, diff, abs(diff), vt*vs,
            ((vt*vs < 0) & (abs(vt) > .1) & (abs(vs) > .1)).astype(float)], axis=1).astype(np.float32)
        assert all(np.isfinite(value).all() for value in features.values())
        np.savez(file, **features)
        print(f'FEATURES {n+1}/{len(rows)} {row.id}', flush=True)
    all_features = []
    for row in rows.itertuples():
        fingerprint = hashlib.sha256((row.id + row.utterance + feature_signature()).encode()).hexdigest()[:20]
        with np.load(rowcache / f'{fingerprint}.npz') as item:
            all_features.append(dict(item))
    np.savez_compressed(OUT / 'features.npz', **{key: np.vstack([f[key] for f in all_features]) for key in all_features[0]})
    # Original rows retain their own original text/prosody features for training.
    vt = old['text_sent'][:, 2] - old['text_sent'][:, 0]
    vs = old['speech_emo'] @ np.array([runtime['emo_valence'](s) for s in labels])
    old['mismatch'] = np.stack([vt,vs,vt-vs,abs(vt-vs),vt*vs,
        ((vt*vs<0)&(abs(vt)>.1)&(abs(vs)>.1)).astype(float)],axis=1).astype(np.float32)
    np.savez_compressed(OUT / 'baseline_features.npz', **old)
    audit = {'official_csv_sha256': hashlib.sha256((ROOT / 'data/mustardpp.csv').read_bytes()).hexdigest(),
             'targets': len(rows), 'classes': rows.label.value_counts().to_dict(),
             'shared_ids': len(set(rows.id) & set(ids)), 'added_ids': len(set(rows.id)-set(ids)),
             'removed_original_ids': sorted(set(ids)-set(rows.id)),
             'changed_shared_transcripts': int(sum(' '.join(baseline[r.id]['utterance'].split()) != ' '.join(r.utterance.split()) for r in rows.itertuples() if r.id in baseline)),
             'duplicate_normalized_targets': int(rows.group.duplicated().sum()),
             'feature_signature': feature_signature(), 'baseline_feature_source': 'truetone/pp/baseline_features.npz',
             'split_note': 'Group identical normalized transcripts across both corpora; exclude every held-out group from both training conditions.'}
    (ROOT / 'experiment/data_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('READY',audit,flush=True)


if __name__ == '__main__':
    main()
