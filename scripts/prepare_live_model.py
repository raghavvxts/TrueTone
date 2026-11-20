"""Extract notebook features and fit final LR classifiers on all available MUStARD rows."""
import json
import hashlib
import sklearn
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import fit_bundle, MODEL_PATH


def main():
    os.chdir(ROOT)
    runtime_root = Path(os.environ.get('TRUETONE_DIR', ROOT / 'truetone')).resolve()
    os.environ['TRUETONE_DIR'] = str(runtime_root)
    os.environ.setdefault('TRUETONE_JOBS', '4')
    os.environ.setdefault('HF_HOME', str(runtime_root / 'hf'))
    if shutil.which('ffmpeg') is None:
        import imageio_ffmpeg
        binaries = runtime_root / 'bin'
        binaries.mkdir(parents=True, exist_ok=True)
        binary = binaries / 'ffmpeg'
        if not binary.exists():
            binary.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
        os.environ['PATH'] = str(binaries) + os.pathsep + os.environ['PATH']
    notebook = json.loads((ROOT / 'TrueTone.ipynb').read_text())
    namespace = {'__name__': '__main__'}
    # Reuse the actual notebook setup/data/features; never run its evaluation protocols.
    for index in (2, 3, 4, 5, 6, 8, 9):
        print(f'Preparing notebook cell {index}', flush=True)
        exec(compile(''.join(notebook['cells'][index]['source']), f'notebook_cell_{index}', 'exec'), namespace)
    model_path = MODEL_PATH
    bundle = fit_bundle(namespace['B'], namespace['y'], namespace['CFG'], namespace['PROS_NAMES'],
                        namespace['EMO_LABELS'], namespace['meta'].id.tolist(), model_path)
    manifest = dict(dataset=bundle['dataset'], training_examples=bundle['n_samples'],
                    classifier='LogisticRegressionCV', models={name: value['blocks'] for name, value in bundle['models'].items()},
                    feature_signature=bundle['feature_signature'], sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    sklearn_version=sklearn.__version__, threshold=.5, transcription='openai/whisper-tiny.en',
                    note='Final refits on all examples; live microphone/ASR accuracy has not been benchmarked.')
    (model_path.parent / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(bundle["models"])} LR classifiers trained on {bundle["n_samples"]} examples to {model_path}', flush=True)


if __name__ == '__main__':
    main()
