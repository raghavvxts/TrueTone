"""Exercise the actual notebook functions without downloading models or data."""
import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'SarcasTone_vastaiClaude-2.ipynb').read_text())


def source(i):
    return ''.join(NOTEBOOK['cells'][i]['source'])


def definitions(i):
    tree = ast.parse(source(i))
    tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    return compile(tree, f'cell_{i}', 'exec')


class NotebookTests(unittest.TestCase):
    def test_all_cells_compile(self):
        import nbformat
        nbformat.validate(nbformat.from_dict(NOTEBOOK))
        for i, cell in enumerate(NOTEBOOK['cells']):
            if cell['cell_type'] == 'code':
                compile(source(i), f'cell_{i}', 'exec')

    def test_prosody_audio(self):
        import librosa
        ns = {'np': np, 'librosa': librosa}
        exec(definitions(8), ns)
        with tempfile.TemporaryDirectory() as temp:
            for name, audio in [('tone', .2*np.sin(2*np.pi*180*np.arange(16000)/16000)), ('silence', np.zeros(4000))]:
                path = Path(temp) / f'{name}.wav'
                sf.write(path, audio, 16000)
                features = ns['prosody_feats'](path, 4)
                self.assertEqual(len(features), 58)
                self.assertTrue(np.isfinite(list(features.values())).all())
                self.assertAlmostEqual(features['articulation_rate'], features['words_per_sec_speech'])

    def test_classifiers_and_train_only_scaling(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(42)
        y = np.tile([0, 1], 30)
        blocks = {'text': rng.normal(size=(60, 4)).astype('float32'), 'audio': rng.normal(size=(60, 3)).astype('float32')}
        blocks['text'][:, 0] += y*3
        ns = dict(np=np, pd=pd, torch=torch, B=blocks, y=y, DEVICE='cpu', CFG=dict(lr_Cs=[.1, 1], net_seeds=1, net_hidden=8, net_dropout=.1, net_lr=.001, net_wd=.01, net_bs=16, net_epochs=2))
        exec(definitions(10), ns)
        tr, te = np.arange(48), np.arange(48, 60)
        before = ns['make_X'](['text'], tr, te, True)[0][0].copy()
        blocks['text'][te] += 1000
        np.testing.assert_array_equal(before, ns['make_X'](['text'], tr, te, True)[0][0])
        for function in ('fit_predict_lr', 'fit_predict_net'):
            p, gate = ns[function](['text', 'audio'], tr, te, 42)
            self.assertEqual(p.shape, (12,))
            self.assertTrue(((p >= 0) & (p <= 1)).all())
            self.assertTrue(np.isfinite(list(ns['fold_metrics'](y[te], p).values())).all())
            if gate is not None:
                self.assertAlmostEqual(float(gate.sum()), 1, places=5)

    def test_bootstrap_identical_predictions(self):
        ns = dict(np=np, CFG={'bootstrap': 20}, rng=np.random.default_rng(42), y=np.tile([0, 1], 10))
        from sklearn.metrics import f1_score
        ns['f1_score'] = f1_score
        exec(definitions(13), ns)
        p = ns['y'].astype(float)
        delta, lo, hi, probability = ns['boot_delta'](p, p)
        self.assertEqual((delta, lo, hi, probability), (0, 0, 0, 1))

    def test_saved_results(self):
        from sklearn.metrics import f1_score
        for protocol, repeats in [('cv', 3), ('loso', 1)]:
            folds = pd.read_csv(ROOT / f'results/folds_{protocol}.csv')
            summary = pd.read_csv(ROOT / f'results/results_{protocol}.csv', index_col=0)
            with np.load(ROOT / f'results/oof_{protocol}.npz', allow_pickle=False) as predictions:
                self.assertEqual(set(predictions.files), set(summary.index))
                for model in predictions.files:
                    values = predictions[model]
                    self.assertEqual(values.shape, (repeats, 690))
                    valid = values[np.isfinite(values)]
                    self.assertTrue(((valid >= 0) & (valid <= 1)).all())
                    self.assertEqual(np.isnan(values).sum(), 0 if protocol == 'cv' else 54)
                    selected = folds[folds.model == model]
                    self.assertAlmostEqual(selected.f1.mean(), summary.loc[model, 'f1_mean'])
                    self.assertAlmostEqual(selected.f1.std(), summary.loc[model, 'f1_std'])


if __name__ == '__main__':
    unittest.main()
