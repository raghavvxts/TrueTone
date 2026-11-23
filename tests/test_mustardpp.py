"""Verify the deployed dataset identity and saved paired evaluation evidence."""
import hashlib
import json
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from inference import Predictor, feature_signature

ROOT = Path(__file__).resolve().parents[1]


class MustardPlusPlusTests(unittest.TestCase):
    def test_default_bundle_matches_manifest_and_keeps_original_baseline(self):
        active = Predictor(ROOT/'models/truetone_lr.joblib').bundle
        manifest = json.loads((ROOT/'models/manifest.json').read_text())
        self.assertEqual((active['dataset'], active['n_samples']), ('MUStARD++', 1202))
        self.assertEqual(len(set(active['ids'])), 1202)
        self.assertEqual(manifest['feature_signature'], feature_signature())
        self.assertEqual(manifest['sha256'], hashlib.sha256((ROOT/'models/truetone_lr.joblib').read_bytes()).hexdigest())
        baseline = joblib.load(ROOT/'models/baseline/truetone_lr.joblib')
        self.assertEqual((baseline['dataset'], baseline['n_samples']), ('MUStARD', 690))
        self.assertEqual(len(set(active['ids']) & set(baseline['ids'])), 688)
        for model in active['models'].values():
            np.testing.assert_array_equal(model['classifier'].classes_, [0, 1])
            for scaler in model['scalers'].values():
                self.assertEqual(int(scaler.n_samples_seen_), 1202)

    def test_saved_metrics_recompute_from_paired_out_of_fold_predictions(self):
        predictions = pd.read_csv(ROOT/'results/mustardpp/predictions.csv')
        summary = pd.read_csv(ROOT/'results/mustardpp/summary.csv')
        self.assertEqual(len(predictions), 688*3*3*3)
        for row in summary.itertuples():
            part = predictions[(predictions.training==row.training)&(predictions['mode']==row.mode)]
            self.assertEqual(part.groupby('id').size().unique().tolist(), [3])
            avg = part.groupby('id').agg(label=('label','first'),score=('score','mean'))
            self.assertEqual(len(avg), 688)
            self.assertAlmostEqual(f1_score(avg.label, avg.score>=.5), row.f1)
            self.assertAlmostEqual(accuracy_score(avg.label, avg.score>=.5), row.accuracy)
        # One common test population/assignment across all training conditions and modes.
        grouped=predictions.groupby(['repeat','fold','training','mode']).id.agg(lambda x:tuple(sorted(x)))
        for (repeat,fold),group in grouped.groupby(level=[0,1]):
            self.assertEqual(len(set(group)),1)
        for repeat,part in predictions.groupby('repeat'):
            self.assertEqual(part.groupby('group').fold.nunique().max(),1)

    def test_dataset_audit_and_uncertainty_are_explicit(self):
        audit=json.loads((ROOT/'experiment/data_audit.json').read_text())
        self.assertEqual(audit['targets'],1202)
        self.assertEqual(audit['classes'], {'0':601,'1':601})
        self.assertEqual(audit['shared_ids'],688)
        intervals=pd.read_csv(ROOT/'results/mustardpp/paired_bootstrap.csv')
        row=intervals[(intervals['mode']=='Text + speech')&(intervals.comparison=='MUStARD++ minus MUStARD')].iloc[0]
        self.assertLess(row.ci_low,0)
        self.assertGreater(row.ci_high,0)
        self.assertAlmostEqual(row.delta_f1,.01732331069506854)


if __name__=='__main__':unittest.main()
