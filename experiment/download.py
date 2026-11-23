"""Download official annotations and verified video mirror; no Git network writes."""
import hashlib
import json
from pathlib import Path
import urllib.request

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data'
OFFICIAL = 'https://raw.githubusercontent.com/cfiltnlp/MUStARD_Plus_Plus/main/mustard%2B%2B_text.csv'


def main():
    DEST.mkdir(exist_ok=True)
    csv = DEST / 'mustardpp.csv'
    if not csv.exists():
        with urllib.request.urlopen(OFFICIAL, timeout=60) as source:
            csv.write_bytes(source.read())
    mirror_csv = Path(hf_hub_download('sifan077/MUStARD_plus_plus', 'MUStARD++/mustard++_text.csv',
                                   repo_type='dataset', local_dir=DEST/'mirror'))
    if csv.read_bytes() != mirror_csv.read_bytes():
        raise RuntimeError('Official and mirror annotations differ; review before training.')
    hashes = json.loads((ROOT/'experiment/archive_hashes.json').read_text())
    for name, expected in hashes.items():
        file = Path(hf_hub_download('sifan077/MUStARD_plus_plus', f'MUStARD++/{name}',
                                   repo_type='dataset', local_dir=DEST/'mirror'))
        digest = hashlib.sha256()
        with file.open('rb') as source:
            for chunk in iter(lambda: source.read(8*1024*1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(f'Archive checksum failed: {name}')
        print('Verified', name, flush=True)


if __name__ == '__main__':
    main()
