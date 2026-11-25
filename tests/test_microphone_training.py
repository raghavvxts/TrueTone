import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from experiment.microphone_training import validate,augment,calibrated,temperature,metrics

class MicrophoneTrainingTests(unittest.TestCase):
    def fixtures(self,directory):
        rows=[]
        for i in range(12):
            file=f'{i}.wav'; x=.15*np.sin(2*np.pi*(140+i*11)*np.arange(32000)/16000)
            sf.write(directory/file,x,16000)
            rows.append(dict(id=str(i),audio=file,speaker=f's{i//2}',split=['train','calibration','test'][i//4],consent='yes',transcript='Test words',sarcasm_rater1=str(i%2),sarcasm_rater2=str(i%2),emotion_rater1='Neutral',emotion_rater2='Neutral'))
        return pd.DataFrame(rows)
    def test_split_consent_disagreement_and_duplicate_guards(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frame=self.fixtures(root);validate(frame,root)
            for column,value in [('consent','no'),('speaker','s0'),('emotion_rater2','Anger')]:
                changed=frame.copy();changed.loc[11,column]=value
                with self.assertRaises(ValueError):validate(changed,root)
            changed=frame.copy();changed.loc[11,'audio']=changed.loc[0,'audio']
            with self.assertRaisesRegex(ValueError,'Duplicate audio'):validate(changed,root)
    def test_augmentation_and_calibration_end_to_end_on_synthetic_features(self):
        x=np.sin(np.arange(16000)/10)*.1
        self.assertEqual(len(augment(x,42)),3)
        np.testing.assert_array_equal(augment(x,42)[2][1],augment(x,42)[2][1])
        rng=np.random.default_rng(42);features=rng.normal(size=(100,3));y=(features[:,0]>0).astype(int)
        model=LogisticRegression().fit(features[:50],y[:50])
        t=temperature(model.predict_proba(features[50:75]),y[50:75],[0,1])
        p=calibrated(model.predict_proba(features[75:]),t)
        np.testing.assert_allclose(p.sum(axis=1),1)
        result=metrics(y[75:],p,[0,1]);self.assertTrue(np.isfinite(result['log_loss']))
        self.assertEqual(np.shape(result['confusion_matrix']),(2,2))
