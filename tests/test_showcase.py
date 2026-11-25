import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import soundfile as sf
from scripts.prepare_demos import select_examples
from experiment.train_emotion import LABELS

class ShowcaseTests(unittest.TestCase):
    def test_selects_both_classes_for_nine_annotations_without_model_scores(self):
        with tempfile.TemporaryDirectory() as d:
            wav=Path(d)/'sample.wav';sf.write(wav,.1*np.sin(np.arange(48000)/10),16000)
            rows=[dict(id=f'{emotion}-{label}',Explicit_Emotion=emotion,label=label,utterance='This is a classroom example.',wav=str(wav),show='Test') for emotion in LABELS for label in (0,1)]
            rows.append(dict(rows[0],id='000-excluded',utterance='This has a sex reference.'))
            selected=select_examples(pd.DataFrame(rows))
            self.assertEqual(len(selected),18)
            self.assertEqual({(r['emotion'],r['label']) for r in selected},{(emotion,label) for emotion in LABELS for label in (0,1)})
            self.assertNotIn('000-excluded',[r['id'] for r in selected])
