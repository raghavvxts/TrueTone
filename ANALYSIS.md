# Project analysis

## Contents and architecture

The source is one 16-cell notebook. It downloads MUStARD annotations and utterance videos, extracts 16 kHz mono audio, computes frozen RoBERTa/WavLM/sentiment/emotion representations plus 58 prosody features, then evaluates logistic regression and gated fusion networks. Evaluation uses repeated stratified five-fold CV and held-out BBT/FRIENDS shows. The final cell generates Markdown, LaTeX, figures, PDF and ZIP outputs.

The supplied archive includes 14 model variants, fold metrics, OOF predictions and reports. The standalone PDF duplicates the archived report. No raw data, feature caches, checkpoints, web application or inference entry point were supplied.

## Findings and changes

1. Feature caches previously used only sample count: reordered or changed data could silently reuse unrelated rows. Cache identity now includes ordered IDs/transcripts/context, audio path/size/mtime, configuration and feature version. This is metadata-based audio invalidation; same-size, same-mtime edits still require deleting caches. Emotion labels and prosody use the same identity.
2. Articulation rate subtracted inter-segment pauses from already active speech duration, double-counting silence. It now uses active speech duration directly.
3. The doubled bootstrap tail proportion returned 2 for identical predictions. It is capped at 1. This remains a descriptive bootstrap tail statistic, not a calibrated null-hypothesis test.
4. Bootstrap significance uses the first CV repeat; reported mean fold F1 uses all three. Their deltas are different estimands. Interpret confidence intervals with caution: utterances can share speakers and shows, while the bootstrap treats them independently.
5. LR preprocessing is fit on the outer training fold before inner hyperparameter CV. Outer test folds remain isolated, but inner validation influences scaler statistics. A fully nested preprocessing pipeline would be needed for strict inner-CV isolation.
6. Speech-only and prosody-only features use transcript word counts for speaking rate. They are therefore not strictly independent of textual input. Names are retained to match historical artifacts.
7. Early-fusion LR and the LR no-mismatch ablation are identical configurations and yield duplicate results. They are retained as named comparisons.
8. Model identifiers are not pinned to immutable Hugging Face revisions; dependency versions in the supplied report describe its original environment. Fresh installations may differ. The tested local environment is captured separately in `requirements-verified.txt`.
9. Default parallel classifier workers are capped at eight and can be overridden. Torch was missing from notebook installation dependencies and is now explicit.
10. Full pipeline automation saves the executed notebook even when a cell fails. The source notebook's historical outputs were cleared; extracted results are preserved unchanged.

## Interpretation of supplied results

Mean random-split F1 peaks at 0.762 for gated fusion without mismatch, versus 0.758 with mismatch. LR text-only scores 0.668; LR speech-only 0.733. On pooled held-out shows, LR speech-only leads at 0.652; LR multimodal with mismatch scores 0.628. These results do not establish that explicit mismatch improves over speech-only or simple fusion. Show/speaker overlap and sitcom laugh tracks can inflate random-split estimates.

## Verification scope

Local tests execute notebook functions on synthetic audio/features and validate saved artifact structure and fold-summary consistency. Full encoder downloads, all 690 clips, complete repeated-CV training and report regeneration are not covered by these tests. Historical metrics must not be presented as newly reproduced results after these corrections.

Local verification: Python 3.12.14 on macOS arm64; five unittest cases passed (16.346 seconds). Notebook schema/code compilation, runner help and script compilation also passed. The original five-page PDF was text-extracted and its first page visually inspected; no PDF content was changed.

## Complete local run: 20 November 2026

Completed the full corrected notebook on macOS arm64 CPU, using four classifier workers, in 10.8 minutes. Downloaded all 690 MUStARD clips; extracted audio, all encoder blocks and 58 prosody features. All 15 code cells executed without errors. Both protocols completed: 210 repeated-CV fold/model rows and 28 held-out-show rows, covering 14 model variants. Generated fold means match their saved summaries. The generated five-page PDF was rendered and visually inspected.

Fresh outputs and the executed notebook are in `results/local-run-2026-11-20/`; original artifacts remain in the parent `results/` directory. The source notebook still has clean outputs. Dataset/model downloads and feature caches remain local under ignored `sarcastone/`.

Best mean CV F1: gated multimodal with mismatch, 0.7655 (fold standard deviation 0.0476). Best pooled held-out-show F1: LR speech-only, 0.6584. This run does not establish that explicit mismatch improves cross-show generalisation. Statistical and protocol caveats above still apply.

The first execution exposed missing pip in the uv environment; pip is now an explicit setup dependency. The runner now logs cell starts/completions and saves progress after each completed cell. Its callbacks were separately checked by executing and saving a small notebook. No cloud machine was provisioned; the report's Vast cost is a configured estimate, not actual local spend.
