"""Compare local ASR against nine official target transcripts, not classifier accuracy.

Requires prepared MUStARD++ data. Detailed transcripts stay under ignored truetone/.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import soundfile as sf
import torch
from inference import prepare_audio, pretrained_component, validate_auto_transcript

# Fixed diagnostic targets cover every available show/label stratum; not a random benchmark.
FIXTURE_IDS = ('1_12320', '1_4760', '2_413', '2_287', '2_591', '2_222', '2_485',
               '3_S02E04_162', '3_S06E03_366')


def normalized_words(text):
    return re.findall(r"[a-z0-9']+", text.lower().replace('\u2019', "'"))


def word_errors(reference, hypothesis):
    """Unit-cost Levenshtein edits; WER can exceed 1 for excessive insertions."""
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j-1] + (expected != actual)))
        previous = current
    return previous[-1]


def main():
    from transformers import pipeline
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['openai/whisper-tiny.en', 'openai/whisper-base.en', 'openai/whisper-small.en'])
    parser.add_argument('--output', type=Path, default=ROOT/'truetone/transcript-check')
    args = parser.parse_args()
    metadata = ROOT/'truetone/pp/metadata.csv'
    if not metadata.exists():
        parser.error('Prepare MUStARD++ data with experiment/prepare.py first.')
    targets = pd.read_csv(metadata).set_index('id').loc[list(FIXTURE_IDS)]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    results, summaries = [], []
    for name in args.models:
        asr = pipeline('automatic-speech-recognition',
                       model=pretrained_component('AutoModelForSpeechSeq2Seq', name),
                       tokenizer=pretrained_component('AutoTokenizer', name),
                       feature_extractor=pretrained_component('AutoFeatureExtractor', name), device=-1)
        first = targets.iloc[0]
        samples, rate = sf.read(first.wav)
        asr({'raw': prepare_audio((rate, samples)), 'sampling_rate': 16000})
        for clip_id, row in targets.iterrows():
            samples, rate = sf.read(row.wav)
            audio = prepare_audio((rate, samples))
            start = time.perf_counter()
            text = asr({'raw': audio, 'sampling_rate': 16000})['text'].strip()
            elapsed = time.perf_counter() - start
            expected = normalized_words(row.utterance)
            errors = word_errors(expected, normalized_words(text))
            try:
                validate_auto_transcript(text, len(audio)/16000)
                guard = 'accepted'
            except ValueError:
                guard = 'rejected'
            results.append(dict(model=name, id=clip_id, show=row.show, label=int(row.label),
                                reference_words=len(expected), word_errors=errors,
                                wer=errors/len(expected), seconds=elapsed, guard=guard,
                                reference=row.utterance, hypothesis=text))
            print(name, clip_id, f'WER={errors/len(expected):.3f}', flush=True)
        frame = pd.DataFrame(results)
        group = frame[frame.model == name]
        summaries.append(dict(model=name, clips=len(group), reference_words=int(group.reference_words.sum()),
                              word_errors=int(group.word_errors.sum()),
                              wer=float(group.word_errors.sum()/group.reference_words.sum()),
                              mean_seconds=float(group.seconds.mean())))
        frame.to_csv(args.output/'details.csv', index=False)
        (args.output/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    print('Detailed output remains local:', args.output)


if __name__ == '__main__':
    main()
