"""Emotion head must not consume transcript counts or private examples."""
import unittest
import numpy as np
from experiment.train_emotion import features

class EmotionTrainingTests(unittest.TestCase):
    def test_audio_features_exclude_transcript_counts(self):
        blocks={'wavlm':np.ones((2,3)), 'speech_emo':np.ones((2,4)), 'prosody':np.zeros((2,58))}
        expected=features(blocks)
        blocks['prosody'][:,24:27]=999
        np.testing.assert_array_equal(features(blocks),expected)
        self.assertEqual(expected.shape,(2,62))
        blocks['wavlm'][0,0]=np.nan
        with self.assertRaises(ValueError):features(blocks)

class ConnectedEmotionTests(unittest.TestCase):
    def test_candidate_integrity_and_nine_class_description(self):
        from inference import emotion_candidate, describe_voice
        head=emotion_candidate()
        self.assertEqual(len(head['model'].classes_),9)
        self.assertEqual(head['training_examples'],1202)
        p=np.full((1,9),.025);p[0,4]=.8
        text,scores=describe_voice(p,list(head['model'].classes_),{})
        self.assertEqual(len(scores),9)
        self.assertIn('Experimental MUStARD++',text)
        self.assertIn('Frustration-like',text)
