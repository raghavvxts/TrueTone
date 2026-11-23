"""Reproduce original MUStARD features for the archived, controlled baseline."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import imageio_ffmpeg
    import numpy as np
    os.chdir(ROOT)
    os.environ['TRUETONE_DIR'] = str(ROOT / 'truetone')
    os.environ.setdefault('HF_HOME', str(ROOT / 'truetone/hf'))
    binaries = ROOT / 'truetone/bin'
    binaries.mkdir(parents=True, exist_ok=True)
    ffmpeg = binaries / 'ffmpeg'
    if not ffmpeg.exists():
        ffmpeg.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ['PATH'] = str(binaries) + os.pathsep + os.environ['PATH']
    notebook = json.loads((ROOT / 'research/baseline/TrueTone_MUStARD.ipynb').read_text())
    namespace = {'__name__': '__main__'}
    for index in (2, 3, 4, 5, 6, 8, 9):
        print(f'Preparing baseline cell {index}', flush=True)
        exec(compile(''.join(notebook['cells'][index]['source']), f'baseline_cell_{index}', 'exec'), namespace)
    destination = ROOT / 'truetone/pp/baseline_features.npz'
    destination.parent.mkdir(parents=True, exist_ok=True)
    blocks = ('text_emb', 'text_sent', 'wavlm', 'speech_emo', 'prosody', 'mismatch')
    np.savez_compressed(destination, **{b: namespace['B'][b] for b in blocks})
    print('Saved original MUStARD features:', destination)


if __name__ == '__main__':
    main()
