"""Execute the complete research pipeline; downloads data and pretrained models."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('sarcastone/executed.ipynb'))
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / 'SarcasTone_vastaiClaude-2.ipynb', as_version=4)
args.output.parent.mkdir(parents=True, exist_ok=True)
try:
    NotebookClient(notebook, timeout=None, kernel_name='sarcastone', resources={'metadata': {'path': str(root)}}).execute()
finally:
    nbformat.write(notebook, args.output)
