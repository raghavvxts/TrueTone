# App audit — 25 November 2026

## Dataset identity

The default serving bundle is MUStARD++ with 1,202 unique targets (601 per sarcasm class). Existing automated checks verify its manifest SHA-256, feature signature, classifier classes and scalers fitted on 1,202 examples. The website now rejects the archived 690-target MUStARD bundle instead of displaying a misleading MUStARD++ label. Baseline files remain intentionally archived for comparison.

Frozen RoBERTa, WavLM, sentiment and SUPERB encoders are generic pretrained feature extractors, not MUStARD-trained classifiers. Reusing them does not mean that the app is still serving MUStARD. The nine-label MUStARD++ surface-emotion head was trained separately but remains a candidate: see ../emotion/README.md. Replacing emotion features inside the sarcasm pipeline requires retraining and reevaluating that pipeline; it cannot be substituted silently.

## Demo recordings

Run `scripts/prepare_demos.py` after preparing data. It creates four local dataset audio examples, two per sarcasm class, selected by ID and duration without inspecting prediction scores. The app exposes these in a collapsed examples panel after restart. Reference transcripts are supplied. Audio stays in ignored `truetone/demos/`, not Git. Dataset terms still apply.

Run `scripts/check_demo_robustness.py` for original, quarter-volume and synthetic 20 dB SNR variants. Saved results contain scores and timings, no personal audio or transcripts. These examples were seen during final refit; results are playback/stability checks, not independent accuracy. They do not test automatic transcription because reference transcripts are supplied.

## What to improve next

1. Collect a consented, balanced microphone dataset spanning different speakers, microphones and rooms. Include sincere and sarcastic deliveries of identical sentences, plus neutral, angry and frustrated speech. Get independent human labels; retain disagreement instead of inventing certainty.
2. Reserve entire speakers for a final test set before experimentation. Never tune on the instructor's evaluation recording. Keep personal recordings local and excluded from Git.
3. First compare corrected transcripts with automatic transcripts to separate ASR errors from classifier errors. Tiny/base/small diagnostic checks currently favor Tiny for latency; nine checked TV clips do not establish microphone ASR accuracy.
4. Fit calibration on a separate validation set and report reliability, macro F1, per-class recall and confusion matrices. A softmax score near 100% is not a correctness guarantee. Do not cap or adjust scores to make a demo look right.
5. Evaluate audio augmentation (noise, gain, reverberation) using training folds only. Compare a fine-tuned speech encoder against the frozen baseline with identical speaker-disjoint evaluation. The small, imbalanced nine-emotion corpus makes overfitting likely; no backbone fine-tuning improvement is claimed here.
6. Context can be essential to sarcasm. A single-utterance microphone app cannot reliably infer missing conversational context. Any context input needs its own training and evaluation.

No new generalization-accuracy gain is claimed by this audit. The existing nine-emotion candidate reaches 31.0% held-out-show accuracy, below a 36.4% majority baseline, so it is not promoted. Existing sarcasm evaluation and paper/PPT metrics remain unchanged. No private recording was added to training or public results.

## Observed stability check

All 12 requests returned finite sarcasm scores and normalized emotion scores. Ten of twelve predictions matched the dataset sarcasm label, but this is training-set playback and must not be reported as test accuracy. One non-sarcastic example (`1_10004`) was misclassified in its original and noisy versions; lowering its volume changed its predicted class. The other three examples retained their class across the tested conditions. Examples were not removed to conceal this failure. Volume sensitivity is a concrete reason to evaluate train-fold audio augmentation before claiming reliable live recording performance.

Automated validation: 28 Python tests and two JavaScript theme tests pass. The new examples panel builds successfully; clicking those examples in the installed desktop app has not yet been visually reverified. Existing prior microphone checks do not validate new speakers' emotion labels. The earlier clear-during-analysis reset change still needs its dedicated UI replay.

## Connected experimental mode and generated audio

The nine-class MUStARD++ surface-emotion head is now available under **Adjust transcript or model → Emotion model → MUStARD++ nine-class (experimental)**. Existing four-class remains the default because no reliable live-microphone improvement has been established. The candidate artifact SHA-256 and feature signature are checked before loading. It changes the displayed emotion scores only; sarcasm features, predictions and cue-contrast calculations retain their evaluated pipeline.

`scripts/check_generated_audio.py` generated 16 macOS-synthesized clips (two voices × eight sentences), evaluated clean and synthetic 20 dB SNR versions in both modes: 64 complete finite inference results. Reference transcripts were supplied; no emotion/sarcasm ground truth is assigned. Top emotion changed under added noise on 10/16 pairs in the existing mode and 4/16 in the experimental mode. This is a limited synthetic stability observation, not evidence of improved human emotion accuracy. Sarcasm scores remained identical between emotion modes. Full numeric results are in generated_audio.json; audio stays local under ignored truetone/.

Both modes were also exercised through the actual Gradio app API on generated audio. The new model returns nine emotion scores and the existing mode returns four. Thirty-one Python tests and two JavaScript theme tests pass. The local microphone manifest remains empty, so no microphone-trained/calibrated candidate exists or has been connected. The installed desktop app needs a restart to load the new Python UI. Visual replay of the new controls and clear-during-analysis is still outstanding.

## Labelled showcase section

The app now has a full-width collapsed **Showcase demos · sarcasm and emotions** section, with a selector and explicit Load selected demo action. The local library contains 18 source-dataset recordings: one for each combination of two sarcasm classes and nine surface-emotion annotations. Selection uses annotation, ID, 2–10 second duration and a classroom-content filter, not whether model predictions are correct. Source audio and transcripts remain local under truetone/demos/.

All 18 load actions were checked through the actual app API: audio, exact reference transcript, cleared prior results and separate annotation text were returned. An invalid ID was rejected by Gradio's dropdown validation. Thirty-six inference requests (18 clips × two emotion modes) returned valid outputs. Sarcasm predictions matched the source annotation on 14/18 examples in each mode; four mismatches are preserved. This is training-set playback, not held-out accuracy. Surface labels are dataset annotations and are not independently relabelled by us. Reference transcripts were used; automatic transcription is not validated by this check. See showcase_verification.json and scripts/verify_showcase.py.

Demo loading updates audio and transcript together. User audio input resets stale reference labels; the programmatic demo load no longer triggers a transcript-erasing reset. Clear resets the selected reference text and outputs. Neither emotion mode nor sarcasm models are silently changed when a demo is loaded.

Desktop follow-up: restarted the installed native app, expanded the showcase section, selected/loaded 1_1484, confirmed the exact reference transcript remained in the input, played the four-second clip, selected the experimental nine-class head and analyzed. The displayed transcript, non-sarcastic prediction and Anger-like emotion estimate matched that request's results. Progress hid the old result during processing. This was a UI/playback verification, not independent human relabelling of all 18 emotions. Final automated checks: 32 Python tests and two JavaScript theme tests pass.
