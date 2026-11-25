# TrueTone: final MUStARD++ migration

The active notebook, app, model manifest, final PPTX and paper describe the same MUStARD++ project: 1,202 balanced targets with frozen encoders and logistic-regression classifiers. MUStARD remains the historical baseline, never relabeled as MUStARD++.

## Evidence

The controlled [paired evaluation](results/mustardpp/report.md) compares identical shared test targets and feature/classifier settings. Text + speech F1 is 0.7608 for MUStARD++ versus 0.7435 for MUStARD. The paired group-bootstrap interval for the difference includes zero. This supports reporting an observed gain, not proven superiority or microphone accuracy.

The original [690-target notebook](research/baseline/TrueTone_MUStARD.ipynb), [model](models/baseline/manifest.json), and [original supplied and corrected local-run outputs](results/mustard-baseline/) are archived separately. Their gated-network and mismatch-ablation results are historical; they are not the final MUStARD++ experiment.

## Remaining limits

- Frozen models and scripted television speech can transfer poorly to live speech.
- Shared-target folds retain show/speaker overlap beyond exact-duplicate grouping.
- Scaling is fitted on outer training data and reused during inner C selection; inner preprocessing is not fully nested.
- The development comparison is not an untouched external test, and per-show classes are uneven.
- Default emotion estimates have four classes; an optional experimental MUStARD++ head has nine. Neither is validated on live microphone speakers. Scores are uncalibrated, and the cue-contrast flag is an unvalidated heuristic.
- Speech-reference speaking-rate features depend on transcript word counts.
- Final app weights are refitted on all targets; training predictions are not an accuracy estimate.

Read the [README](README.md) for running, reproducing and verifying the final project. Personal test recordings and dataset caches are ignored by Git.
