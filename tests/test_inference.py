"""Fast live-demo boundary and classifier artifact tests; no network downloads."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from inference import BLOCKS, Predictor, fit_bundle, predict_features, prepare_audio, describe_voice, cue_contrast, validate_auto_transcript


class InferenceTests(unittest.TestCase):
    def test_auto_transcript_rejects_observed_repetition_and_runaway_output(self):
        for text, duration in [('Thank you. ' * 150, 14), ('Thank you. ' * 15, 20),
                               (' '.join(f'word{i}' for i in range(90)), 5), ('a' * 2001, 30)]:
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(ValueError, 'transcription looks unreliable'):
                    validate_auto_transcript(text, duration)
        for text, duration in [('Thank you.', 1), ('This is a microphone test for TrueTone.', 5),
                               ('Wonderful, I have another meeting.', 5)]:
            validate_auto_transcript(text, duration)
        wave = .2 * np.sin(2*np.pi*180*np.arange(16000)/16000)
        with patch('inference.transcriber', return_value=Mock(return_value={'text': 'Thank you. ' * 150})), \
             patch('inference.feature_runtime') as encoders:
            with self.assertRaisesRegex(ValueError, 'transcription looks unreliable'):
                Predictor().predict((16000, wave))
            encoders.assert_not_called()

    def test_audio_validation_and_resampling(self):
        rate = 48000
        wave = (12000*np.sin(2*np.pi*180*np.arange(rate)/rate)).astype(np.int16)
        converted = prepare_audio((rate, np.column_stack([wave, wave])))
        self.assertEqual(converted.shape, (16000,))
        self.assertTrue(np.isfinite(converted).all())
        self.assertLessEqual(float(np.max(np.abs(converted))), 1)
        for audio in [None, (16000, np.zeros(16000)), (16000, np.ones(16000)*np.nan),
                      (16000, np.ones(16000*31)), (0, wave), (16000, np.zeros((16000, 3))),
                      (16000, np.ones(16000)*2), (16000, np.ones(16000)*.2),
                      (16000, np.ones(16000, dtype=complex)), (16000, np.full(16000, 128, dtype=np.uint8))]:
            with self.subTest(audio=str(type(audio))):
                with self.assertRaises(ValueError):
                    prepare_audio(audio)

    def test_unsigned_pcm_and_voice_description(self):
        wave = (128 + 60*np.sin(2*np.pi*180*np.arange(16000)/16000)).astype(np.uint8)
        self.assertLess(abs(float(prepare_audio((16000, wave)).mean())), .01)
        labels = ['neu', 'hap', 'ang', 'sad']
        text, scores = describe_voice([[.1, .7, .1, .1]], labels,
                                     {'f0_mean': 180, 'f0_voiced_frac': .8, 'words_per_sec': 2})
        self.assertIn('Voice category estimate: Happy-like', text)
        self.assertIn('180 Hz', text)
        self.assertIn('120 words/min', text)
        self.assertEqual(scores['Happy'], .7)
        text, _ = describe_voice([[.3, .3, .2, .2]], labels, {})
        self.assertIn('Mixed / uncertain', text)
        self.assertIn('Pitch unavailable', text)
        with self.assertRaises(RuntimeError):
            describe_voice([[float('nan')]*4], labels, {})

    def test_audio_files_decode_without_system_ffprobe(self):
        import subprocess
        import imageio_ffmpeg
        import soundfile as sf
        from gradio.data_classes import FileData
        from app import create_app
        app = create_app()
        component = next(b for b in app.blocks.values() if b.__class__.__name__ == 'Audio')
        wave = .2*np.sin(2*np.pi*180*np.arange(32000)/16000)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'tone.wav'
            sf.write(source, wave, 16000)
            for suffix in ('wav', 'mp3', 'm4a'):
                path = source if suffix == 'wav' else Path(temp)/f'tone.{suffix}'
                if suffix != 'wav':
                    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y',
                                    '-i', str(source), str(path)], check=True)
                with patch('gradio.processing_utils.audio_from_file', side_effect=AssertionError('system decoder used')):
                    self.assertEqual(component.preprocess(FileData(path=str(path))), str(path))
                decoded = prepare_audio(path)
                self.assertTrue(1.9*16000 <= len(decoded) <= 2.1*16000)
                self.assertTrue(np.isfinite(decoded).all())
                self.assertGreater(float(np.std(decoded)), .01)
            source.write_bytes(b'not an audio file')
            with self.assertRaisesRegex(ValueError, 'Could not decode'):
                prepare_audio(source)

    def test_contrast_flag_does_not_override_classifier_score(self):
        self.assertEqual(cue_contrast(.95, -.98), 'Positive wording / negative vocal delivery')
        self.assertEqual(cue_contrast(-.8, .9), 'Negative wording / positive vocal delivery')
        self.assertEqual(cue_contrast(.1, -.98), '')
        self.assertEqual(cue_contrast(.8, .8), '')
        with self.assertRaises(ValueError):
            cue_contrast(float('nan'), -.9)
        from app import analyze
        contrast = cue_contrast(.95, -.98)
        with patch('app.predictor') as predictor:
            predictor.return_value.predict.return_value = (.073, 'Test utterance', 5.2, 'Voice estimate', {'Sad': .98}, contrast)
            scores, words, details, voice, emotions = analyze(None, '', 'Text + speech')
        self.assertEqual(scores['Sarcastic'], .073)
        self.assertIn('Conflicting cues', details)
        self.assertIn('Classifier prediction:** Not sarcastic', details)
        self.assertIn('polarity heuristic', details)

    def test_frozen_components_reuse_local_weights_and_fallback(self):
        from inference import pretrained_component, CachedPretrained
        pretrained_component.cache_clear()

        factory = Mock()
        with patch.dict('sys.modules', {'transformers': SimpleNamespace(AutoModel=factory)}):
            first = CachedPretrained('AutoModel').from_pretrained('checkpoint')
            self.assertIs(first, CachedPretrained('AutoModel').from_pretrained('checkpoint'))
            factory.from_pretrained.assert_called_once_with('checkpoint', local_files_only=True)
            factory.from_pretrained.reset_mock()
            factory.from_pretrained.side_effect = [OSError('not downloaded'), object()]
            pretrained_component('AutoModel', 'new-checkpoint')
            self.assertEqual(factory.from_pretrained.call_count, 2)
            factory.from_pretrained.assert_called_with('new-checkpoint')
        pretrained_component.cache_clear()

    def test_partial_tokenizer_cache_downloads_missing_vocabulary(self):
        from inference import pretrained_component
        pretrained_component.cache_clear()
        empty = SimpleNamespace(vocab_size=0)
        complete = SimpleNamespace(vocab_size=50257)
        factory = Mock()
        factory.from_pretrained.side_effect = [empty, complete]
        with patch.dict('sys.modules', {'transformers': SimpleNamespace(AutoTokenizer=factory)}):
            self.assertIs(pretrained_component('AutoTokenizer', 'checkpoint'), complete)
            self.assertIs(pretrained_component('AutoTokenizer', 'checkpoint'), complete)
            self.assertEqual(factory.from_pretrained.call_args_list,
                             [unittest.mock.call('checkpoint', local_files_only=True),
                              unittest.mock.call('checkpoint')])
            factory.from_pretrained.side_effect = [empty, empty]
            with self.assertRaisesRegex(RuntimeError, 'vocabulary could not be loaded'):
                pretrained_component('AutoTokenizer', 'broken')
        pretrained_component.cache_clear()

    def test_bundle_roundtrip_and_prediction(self):
        rng = np.random.default_rng(42)
        labels = np.tile([0, 1], 30)
        names = set(b for blocks in BLOCKS.values() for b in blocks)
        features = {name: rng.normal(size=(60, 3)).astype(np.float32) for name in names}
        for value in features.values():
            value[:, 0] += labels*5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.joblib'
            bundle = fit_bundle(features, labels, {'seed': 42, 'lr_Cs': [.1, 1]},
                                ['one', 'two', 'three'], ['neu', 'hap', 'ang', 'sad'], list(map(str, range(60))), path)
            reloaded = Predictor(path)
            sample = {name: values[1:2] for name, values in features.items()}
            for model in BLOCKS:
                score = predict_features(bundle, sample, model)
                self.assertGreater(score, .5)
                self.assertLessEqual(score, 1)
                self.assertEqual(score, predict_features(reloaded.bundle, sample, model))
            with self.assertRaises(ValueError):
                predict_features(bundle, sample, 'Unknown model')
            sample['wavlm'] = np.full((1, 3), np.nan)
            with self.assertRaises(ValueError):
                predict_features(bundle, sample, 'Speech reference')
            with patch('inference.feature_signature', return_value='changed'):
                with self.assertRaisesRegex(RuntimeError, 'Feature code has changed'):
                    Predictor(path)
            with self.assertRaisesRegex(ValueError, 'unique IDs'):
                fit_bundle(features, labels, {'seed': 42, 'lr_Cs': [1]}, [], [], ['same']*60, path)

    def test_ui_builds_with_microphone_and_upload(self):
        from app import create_app
        app = create_app()
        audio = next(block for block in app.blocks.values() if block.__class__.__name__ == "Audio")
        self.assertEqual(audio.sources, ["microphone", "upload"])
        self.assertEqual(audio.type, 'filepath')
        self.assertIsNone(audio.format)

    def test_missing_artifact_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'prepare_live_model.py'):
                Predictor(Path(directory)/'missing.joblib')


if __name__ == '__main__':
    unittest.main()
