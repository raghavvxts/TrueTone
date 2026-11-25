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
        self.assertTrue(any(getattr(b,'value',None)=='Try a MUStARD++ dataset clip' or getattr(b,'label',None)=='Try a MUStARD++ dataset clip' for b in app.blocks.values()))
