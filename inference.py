"""Local inference using the feature functions defined in the research notebook."""
import ast
import hashlib
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(Path(os.environ.get('TRUETONE_DIR', ROOT / 'truetone')) / 'hf'))

import joblib
import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(os.environ.get('TRUETONE_MODEL', ROOT / 'models/truetone_lr.joblib'))
FEATURE_FUNCTIONS = {'embed_text', 'text_sentiment', 'load_wav', 'wavlm_embed', 'speech_emotion', 'prosody_feats', 'emo_valence'}
BLOCKS = {
    'Text + speech': ['text_emb', 'text_sent', 'wavlm', 'prosody', 'speech_emo', 'mismatch'],
    'Speech reference': ['wavlm', 'prosody', 'speech_emo'],
}


def notebook_definitions():
    notebook = json.loads((ROOT / 'TrueTone.ipynb').read_text())
    nodes = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            nodes.extend(n for n in ast.parse(''.join(cell['source'])).body
                         if isinstance(n, ast.FunctionDef) and n.name in FEATURE_FUNCTIONS)
    if {n.name for n in nodes} != FEATURE_FUNCTIONS or len(nodes) != len(FEATURE_FUNCTIONS):
        raise RuntimeError('Notebook feature definitions are missing or duplicated.')
    return ast.Module(body=nodes, type_ignores=[])


def feature_signature():
    return hashlib.sha256(ast.dump(notebook_definitions(), include_attributes=False).encode()).hexdigest()


@lru_cache(maxsize=12)
def pretrained_component(class_name, name):
    """Keep frozen encoders in RAM while the app is open; prefer downloaded files."""
    import transformers
    factory = getattr(transformers, class_name)
    try:
        return factory.from_pretrained(name, local_files_only=True)
    except OSError:
        return factory.from_pretrained(name)


class CachedPretrained:
    def __init__(self, class_name):
        self.class_name = class_name

    def from_pretrained(self, name):
        return pretrained_component(self.class_name, name)


def feature_runtime(cache_dir):
    from tqdm.auto import tqdm
    # Execute only the named feature definitions from our trusted project notebook.
    # Setup, downloads, evaluation cells and arbitrary notebook outputs are not executed.
    namespace = dict(np=np, torch=torch, sf=sf, librosa=librosa, json=json, tqdm=tqdm,
                     DEVICE='cuda' if torch.cuda.is_available() else 'cpu', CACHE=Path(cache_dir), CACHE_KEY='live',
                     **{name: CachedPretrained(name) for name in ('AutoTokenizer', 'AutoModel',
                     'AutoModelForSequenceClassification', 'AutoFeatureExtractor', 'AutoModelForAudioClassification')})
    exec(compile(notebook_definitions(), str(ROOT / 'TrueTone.ipynb'), 'exec'), namespace)
    return namespace


def prepare_audio(audio):
    if audio is None:
        raise ValueError('Record or upload an audio clip first.')
    if not isinstance(audio, (tuple, list)) or len(audio) != 2:
        raise ValueError('Use a valid audio recording.')
    sr, raw = audio
    raw = np.asarray(raw)
    if not isinstance(sr, (int, np.integer)) or not 8000 <= sr <= 192000:
        raise ValueError('Unsupported audio sample rate.')
    if raw.ndim not in (1, 2) or not raw.size or (raw.ndim == 2 and raw.shape[1] not in (1, 2)):
        raise ValueError('Use a mono or stereo audio clip.')
    duration = raw.shape[0] / sr
    if not 1 <= duration <= 30:
        raise ValueError('Use a recording between 1 and 30 seconds.')
    if raw.dtype.kind not in 'iuf':
        raise ValueError('Audio contains invalid sample values.')
    if np.issubdtype(raw.dtype, np.unsignedinteger):
        midpoint = (np.iinfo(raw.dtype).max + 1) / 2
        samples = (raw.astype(np.float32) - midpoint) / midpoint
    elif np.issubdtype(raw.dtype, np.integer):
        limit = max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max)
        samples = raw.astype(np.float32) / limit
    else:
        samples = raw.astype(np.float32)
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1.1:
        raise ValueError('Audio contains invalid sample values.')
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if np.sqrt(np.mean((samples - samples.mean())**2)) < 1e-4:
        raise ValueError('The recording is silent or too quiet. Check your microphone.')
    return librosa.resample(samples, orig_sr=sr, target_sr=16000).astype(np.float32)


def describe_voice(emotion, labels, prosody):
    """Describe delivery using the existing emotion encoder, without identity guesses."""
    names = {'neu': 'Neutral', 'hap': 'Happy', 'ang': 'Angry', 'sad': 'Sad'}
    values = np.asarray(emotion, dtype=float).reshape(-1)
    if (len(values) != len(labels) or set(labels) != set(names) or
            not np.isfinite(values).all() or np.any(values < 0) or
            not np.isclose(values.sum(), 1, atol=.01)):
        raise RuntimeError('Invalid voice emotion scores.')
    order = np.argsort(values)[::-1]
    best, second = order[:2]
    uncertain = values[best] < .6 or values[best] - values[second] < .15
    title = f'Mixed / uncertain (closest: {names[labels[best]].lower()})' if uncertain else names[labels[best]] + '-like'
    pitch = prosody.get('f0_mean', 0)
    pitch_text = (f'{pitch:.0f} Hz estimated mean pitch' if pitch > 0 and
                  prosody.get('f0_voiced_frac', 0) >= .1 else 'Pitch unavailable')
    pace = prosody.get('words_per_sec', 0) * 60
    description = (f'### Voice category estimate: {title}\n'
                   f'**Top emotion score:** {values[best]:.1%}\n\n'
                   f'{pitch_text} · {pace:.0f} words/min estimated from transcript\n\n'
                   'Closest match among neutral, happy, angry and sad. Frustration is not a supported class. '
                   'These uncalibrated scores do not establish the speaker’s feelings.')
    return description, {names[label]: float(value) for label, value in zip(labels, values)}


def cue_contrast(text_valence, speech_valence):
    """Flag strong opposing polarities; this heuristic is not a sarcasm classifier."""
    if not all(np.isfinite(v) and -1.00001 <= v <= 1.00001 for v in (text_valence, speech_valence)):
        raise ValueError('Invalid polarity values.')
    if text_valence > .5 and speech_valence < -.5:
        return 'Positive wording / negative vocal delivery'
    if text_valence < -.5 and speech_valence > .5:
        return 'Negative wording / positive vocal delivery'
    return ''


def fit_bundle(features, labels, config, prosody_names, emotion_labels, ids, path=MODEL_PATH, dataset='MUStARD++'):
    labels = np.asarray(labels)
    if len(ids) != len(labels) or len(set(ids)) != len(ids) or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Training rows must have unique IDs and both binary classes.')
    if min(np.bincount(labels)) < 3:
        raise ValueError('At least three examples per class are required for inner CV.')
    models = {}
    for name, blocks in BLOCKS.items():
        scalers, transformed = {}, []
        for block in blocks:
            values = np.asarray(features[block], dtype=np.float32)
            if values.ndim != 2 or len(values) != len(labels) or not np.isfinite(values).all():
                raise ValueError(f'Invalid training features: {block}')
            scaler = StandardScaler().fit(values)
            scalers[block] = scaler
            transformed.append(np.clip(scaler.transform(values), -5, 5) / np.sqrt(values.shape[1]))
        classifier = LogisticRegressionCV(Cs=config['lr_Cs'],
            cv=StratifiedKFold(3, shuffle=True, random_state=config['seed']),
            scoring='neg_log_loss', max_iter=3000, class_weight='balanced')
        classifier.fit(np.hstack(transformed), labels)
        models[name] = dict(blocks=blocks, scalers=scalers, classifier=classifier)
    bundle = dict(version=1, feature_signature=feature_signature(), config=config,
                  prosody_names=list(prosody_names), emotion_labels=list(emotion_labels),
                  ids=list(ids), n_samples=len(labels), dataset=dataset, models=models)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    joblib.dump(bundle, temp)
    temp.replace(path)
    return bundle


def predict_features(bundle, features, model_name):
    if model_name not in bundle['models']:
        raise ValueError('Choose a supported model.')
    model = bundle['models'][model_name]
    transformed = []
    for block in model['blocks']:
        values = np.asarray(features[block], dtype=np.float32)
        scaler = model['scalers'][block]
        if values.shape != (1, scaler.n_features_in_) or not np.isfinite(values).all():
            raise ValueError(f'Invalid inference features: {block}')
        transformed.append(np.clip(scaler.transform(values), -5, 5) / np.sqrt(values.shape[1]))
    index = int(np.where(model['classifier'].classes_ == 1)[0][0])
    return float(model['classifier'].predict_proba(np.hstack(transformed))[0, index])


@lru_cache(maxsize=1)
def transcriber():
    from transformers import pipeline
    return pipeline('automatic-speech-recognition',
                    model=pretrained_component('AutoModelForSpeechSeq2Seq', 'openai/whisper-tiny.en'),
                    tokenizer=pretrained_component('AutoTokenizer', 'openai/whisper-tiny.en'),
                    feature_extractor=pretrained_component('AutoFeatureExtractor', 'openai/whisper-tiny.en'), device=-1)


class Predictor:
    def __init__(self, path=MODEL_PATH):
        path = Path(path)
        if not path.is_file():
            raise RuntimeError('Prepare the classifier first: .venv/bin/python scripts/prepare_live_model.py')
        # joblib is executable serialization: load only the locally trained, trusted bundle.
        self.bundle = joblib.load(path)
        if self.bundle.get('version') != 1 or self.bundle.get('feature_signature') != feature_signature():
            raise RuntimeError('Feature code has changed. Run scripts/prepare_live_model.py again.')

    def predict(self, audio, transcript='', model_name='Text + speech', progress=lambda value, desc: None):
        if model_name not in BLOCKS:
            raise ValueError('Choose a supported model.')
        if len(transcript or '') > 2000:
            raise ValueError('Keep the transcript below 2,000 characters.')
        progress(.05, desc='Preparing audio')
        samples = prepare_audio(audio)
        transcript = (transcript or '').strip()
        if not transcript:
            progress(.15, desc='Transcribing your recording')
            transcript = transcriber()({'raw': samples, 'sampling_rate': 16000})['text'].strip()
        if not re.search(r'[A-Za-z0-9]', transcript):
            raise ValueError('No speech was recognised. Try again or enter the transcript.')
        contrast = ''
        config = self.bundle['config']
        with tempfile.TemporaryDirectory(prefix='truetone-live-') as temp:
            path = Path(temp) / 'recording.wav'
            sf.write(path, samples, 16000, subtype='FLOAT')
            runtime = feature_runtime(temp)
            progress(.35, desc='Extracting speech and prosody')
            wavlm = runtime['wavlm_embed']([str(path)], config['speech_model'])
            emotion = runtime['speech_emotion']([str(path)], config['emotion_model'])
            labels = json.loads((Path(temp) / 'emo_labels_live.json').read_text())
            if labels != self.bundle['emotion_labels']:
                raise RuntimeError('Emotion checkpoint labels changed. Prepare the classifier again.')
            prosody = runtime['prosody_feats'](str(path), len(re.findall(r"[A-Za-z0-9']+", transcript)))
            features = dict(wavlm=wavlm, speech_emo=emotion,
                            prosody=np.array([[prosody[n] for n in self.bundle['prosody_names']]], dtype=np.float32))
            if model_name == 'Text + speech':
                progress(.65, desc='Extracting text and sentiment')
                features['text_emb'] = runtime['embed_text']([transcript], config['text_model'])
                sentiment = runtime['text_sentiment']([transcript], config['sentiment_model'])
                vt = float(sentiment[0, 2] - sentiment[0, 0])
                vs = float(emotion[0] @ np.array([runtime['emo_valence'](s) for s in labels], dtype=np.float32))
                contrast = cue_contrast(vt, vs)
                features['text_sent'] = sentiment
                features['mismatch'] = np.array([[vt, vs, vt-vs, abs(vt-vs), vt*vs,
                    float(vt*vs < 0 and abs(vt) > .1 and abs(vs) > .1)]], dtype=np.float32)
            progress(.95, desc='Running the trained classifier')
            probability = predict_features(self.bundle, features, model_name)
        progress(1, desc='Complete')
        voice, emotion_scores = describe_voice(emotion, labels, prosody)
        return probability, transcript, len(samples)/16000, voice, emotion_scores, contrast
