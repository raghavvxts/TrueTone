import unittest
from unittest.mock import patch
from types import SimpleNamespace
from app import predictor, create_app

class AppDatasetTests(unittest.TestCase):
    def tearDown(self):predictor.cache_clear()
    def test_interface_rejects_baseline_weights(self):
        predictor.cache_clear()
        with patch('app.Predictor',return_value=SimpleNamespace(bundle={'dataset':'MUStARD','n_samples':690})):
            with self.assertRaisesRegex(RuntimeError,'MUStARD\\+\\+'):
                predictor()
    def test_demo_examples_build_without_prediction_cache(self):
        app=create_app()
        self.assertTrue(any(getattr(b,'value',None)=='Showcase demos · sarcasm and emotions' or getattr(b,'label',None)=='Showcase demos · sarcasm and emotions' for b in app.blocks.values()))

    def test_bad_recording_shows_recovery_message_without_stale_predictions(self):
        from app import analyze
        with patch('app.predictor') as factory:
            factory.return_value.predict.side_effect=ValueError('No speech was recognised. Try again or enter the transcript.')
            result=analyze(None,'','Text + speech')
        self.assertEqual(result[0],None)
        self.assertEqual(result[1],'')
        self.assertIn('Recording needs attention',result[2])
        self.assertIn('enter the transcript',result[2])
        self.assertEqual(result[3:],('',None))
