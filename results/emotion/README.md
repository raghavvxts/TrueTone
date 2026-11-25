# MUStARD++ surface-emotion training

25 November 2026. Experimental candidate; selectable explicitly in the app, not the default.

The candidate trains a nine-class logistic-regression head on frozen WavLM embeddings, frozen four-class emotion outputs, and acoustic prosody. The three transcript-count prosody columns are excluded. This trains a new head, not the pretrained encoder backbone. All 1,202 official targets are used for the final refit. Adil's and other private recordings are not training or evaluation examples.

The target is `Explicit_Emotion` (surface emotion), not `Implicit_Emotion` (hidden emotion), per the [official annotation definitions](https://github.com/cfiltnlp/MUStARD_Plus_Plus). These are multimodal human annotations; predicting them from audio alone has limitations.

Evaluation leaves each show out in turn. Normalized target-text duplicates of test utterances are removed from each training fold. Standardization is fitted only on training rows. C=0.1 is fixed; no hyperparameter search uses test scores. `evaluation.json` records counts, per-show accuracy, pooled accuracy, nine-class macro F1, log loss and the confusion matrix. `held_out_predictions.csv` contains public dataset IDs and labels only. The old detector cannot predict five of the nine labels; its nine-class score is a task-coverage comparison, not an equal-label-set comparison.

Decision: do not promote this candidate. Its macro F1 improves over the frozen four-class detector but accuracy is below the training-fold majority baseline. Neither these TV results nor a high softmax score establish reliability on live microphones or unseen speakers. Calibration and independent labelled microphone evaluation remain necessary. The final serving refit is not evaluated on its own training predictions.

Reproduce after preparing dataset features:

```sh
OPENBLAS_NUM_THREADS=4 .venv/bin/python experiment/train_emotion.py
```

Artifact: `models/truetone_emotion.joblib` (candidate). Existing sarcasm weights and default emotion predictions are unchanged. No fine-tuned encoder or proven improvement on Adil's clip is claimed.

Update: this candidate can now be selected explicitly in the app as **MUStARD++ nine-class (experimental)**. It is not the default. This exposes the trained head for inspection without claiming an accuracy improvement or replacing the evaluated sarcasm pipeline.
