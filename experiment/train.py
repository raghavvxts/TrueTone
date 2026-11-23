"""Refit the final MUStARD++ classifiers from prepared target features."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from inference import fit_bundle, MODEL_PATH


def main():
    metadata = pd.read_csv(ROOT / 'truetone/pp/metadata.csv')
    features = dict(np.load(ROOT / 'truetone/pp/features.npz'))
    previous = joblib.load(ROOT / 'models/baseline/truetone_lr.joblib')
    assert len(metadata) == 1202 and metadata.label.value_counts().to_dict() == {0: 601, 1: 601}
    bundle = fit_bundle(features, metadata.label.to_numpy(), previous['config'],
                        previous['prosody_names'], previous['emotion_labels'], metadata.id.tolist(),
                        MODEL_PATH, dataset='MUStARD++')
    manifest = dict(dataset='MUStARD++', n_samples=1202, training_examples=1202,
                    classifier='LogisticRegressionCV', threshold=.5, sklearn_version='1.9.1',
                    feature_signature=bundle['feature_signature'], training_only=True,
                    sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                    official_annotation_sha256=hashlib.sha256((ROOT/'data/mustardpp.csv').read_bytes()).hexdigest(),
                    transcription='openai/whisper-tiny.en',
                    note='Final refit on all targets; separate fold-trained models produce the reported evaluation.')
    MODEL_PATH.with_name('manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Saved MUStARD++ classifiers:', MODEL_PATH)


if __name__ == '__main__':
    main()
