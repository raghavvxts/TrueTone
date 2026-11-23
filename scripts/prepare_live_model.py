"""Refit the final MUStARD++ classifiers from prepared features; no evaluation run."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'experiment/train.py'), run_name='__main__')
