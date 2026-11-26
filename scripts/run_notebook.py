"""Execute notebook review mode; --retrain explicitly rebuilds the research pipeline."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('truetone/executed.ipynb'))
parser.add_argument('--retrain', action='store_true', help='Download/prepare data and rerun full training/evaluation')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / 'TrueTone.ipynb', as_version=4)
if args.retrain:
    for cell in notebook.cells:
        if cell.cell_type == 'code' and 'RETRAIN = False' in cell.source:
            cell.source = cell.source.replace('RETRAIN = False', 'RETRAIN = True', 1)
            break
    else:
        raise RuntimeError('Notebook retraining switch is missing.')
args.output.parent.mkdir(parents=True, exist_ok=True)
def report_cell(cell, cell_index, **kwargs):
    print(f"Starting cell {cell_index}: {''.join(cell.source).splitlines()[0][:100]}", flush=True)


def save_progress(cell, cell_index, **kwargs):
    nbformat.write(notebook, args.output)
    print(f"Completed cell {cell_index}", flush=True)


try:
    NotebookClient(notebook, timeout=None, kernel_name='truetone', on_cell_execute=report_cell, on_cell_executed=save_progress, resources={'metadata': {'path': str(root)}}).execute()
finally:
    nbformat.write(notebook, args.output)
