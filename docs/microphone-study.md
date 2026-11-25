# Private microphone adaptation protocol

This workflow trains candidate sarcasm and nine-class surface-emotion heads on MUStARD++ plus consented microphone training recordings. Frozen encoders remain frozen. It does not overwrite the app or fine-tune an encoder backbone.

## Collect recordings

The local manifest is `truetone/microphone/manifest.csv`. All recordings, labels, candidates and results must stay under ignored `truetone/`. Do not commit friends' recordings or personal transcripts.

1. Ask each participant for permission to use their recordings for this local project, including model training/evaluation. Record `consent=yes` only after they agree. Use anonymous speaker IDs.
2. Assign speakers to train, calibration or test BEFORE recording or inspecting scores. A person must never cross splits. At least two speakers per split are enforced (six total). This minimum checks the workflow; aim for at least 10 speakers and 100–200 balanced clips before drawing conclusions.
3. Record 3–10 seconds of English speech in TrueTone using the built-in microphone with the lid open. Stop, download the WAV into `truetone/microphone/audio/`, and clear before the next recording. Do not record anyone without consent. Match likely showcase conditions, including ordinary laptop distance and quiet-room speech.
4. Include sincere and sarcastic versions of identical wording; neutral, happy, sad, angry, frustrated, excited, surprised, fearful and disgusted surface deliveries where natural. Do not ask people to exaggerate every example. Include different rooms and distances across all splits rather than assigning one room to each class.
5. Have two independent raters listen without seeing app predictions. Each supplies sarcasm `0` or `1`, and one exact emotion label: Anger, Disgust, Excitement, Fear, Frustration, Happiness, Neutral, Sadness, Surprise. This labels perceived surface delivery, not the person's actual feelings. Copy the human-checked words into `transcript`.
6. Keep disagreements recorded. Discuss/adjudicate them before entering the final matching rater labels used for training; retain the original annotations separately under the ignored study folder. Do not invent an agreed label for an ambiguous clip; collect clearer examples or document exclusion before running the experiment.

Manifest columns:

| Column | Meaning |
|---|---|
| id | Unique anonymous recording ID |
| audio | Relative WAV/MP3/M4A path, e.g. audio/train-s01-01.wav |
| speaker | Anonymous person ID, stable across every clip |
| split | train, calibration, or test |
| consent | yes, only with actual consent |
| transcript | Human-checked spoken words |
| sarcasm_rater1 / sarcasm_rater2 | Agreed 0 or 1 |
| emotion_rater1 / emotion_rater2 | Agreed exact surface-emotion label |

Both sarcasm classes are required in each split. Each test speaker should provide several examples of both classes. Do not label a new instructor recording from the app's prediction or add it to training during the evaluation.

## Train and evaluate once

An empty manifest has already been created on this Mac. On a fresh checkout:

```sh
.venv/bin/python experiment/microphone_training.py --init
```

After collecting and labelling:

```sh
OPENBLAS_NUM_THREADS=4 .venv/bin/python experiment/microphone_training.py
```

The script validates data before loading feature encoders. It uses original MUStARD++ features plus the microphone training split. Fixed C=0.1 prevents test-driven hyperparameter selection. Original, quarter-volume and synthetic 20 dB SNR versions are generated only for microphone training rows; calibration and test clips remain untouched. Reference transcripts are used so this experiment isolates the classifier from ASR errors.

Temperature scaling is fitted only on the calibration split. The final test report compares the existing model, candidate before calibration and candidate after calibration: accuracy, macro F1, log loss, ten-bin expected calibration error and confusion matrix. The old emotion model has only four supported labels; its nine-label result measures coverage as well as prediction quality.

Private outputs:

- `truetone/microphone/candidate.joblib`
- `truetone/microphone/evaluation.json`

No automatic app promotion occurs. A completed test run is locked against reuse by the script. Do not delete the report to tune repeatedly against those same test people; obtain a fresh independent test set after changes. The lock is a workflow safeguard, not a cryptographic study registry.

## Promotion decision and next iteration

Compare candidate macro F1, accuracy, per-class failures and calibration against the existing model. Do not promote a candidate simply because one friend gets the desired label. A small, imbalanced test set cannot establish general reliability. Summarize disagreement and class counts, and inspect the confusion matrices. Calibration can improve reliability while leaving classification accuracy unchanged.

After locking a classifier decision, run a separate end-to-end test with automatic transcripts on fresh speakers and compare them to human transcripts. Existing `experiment/check_transcripts.py` checks TV transcripts only; it is not a substitute for this microphone study. Any new ASR choice, augmentation setting, threshold or encoder fine-tuning requires validation speakers and a fresh final test. Missing conversational context remains an intrinsic limitation.

No microphone accuracy result, backbone fine-tuning, or production integration is claimed until these recordings exist and the evaluation is actually run.
