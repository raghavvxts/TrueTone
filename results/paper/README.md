# Paper figures and evidence

Updated 26 November 2026. The final five-page PDF is in paper/TrueTone_IEEE_Paper.pdf. It uses an IEEE-style two-column layout; it is not a claim of publication or acceptance by IEEE.

Rebuild from repository root:

```sh
.venv/bin/python scripts/build_paper.py
```

The builder calls scripts/paper_figures.py. Four figures derive from saved held-out predictions and metric files: pooled F1 comparison, two sarcasm confusion matrices, paired F1 intervals and the nine-class emotion confusion matrix. Six tables cover targets, feature blocks, F1, all five classification metrics, held-out shows and the separate emotion experiment. No training curves or epoch loss histories are fabricated.

The primary comparison retains original MUStARD results and the size-matched control, while identifying MUStARD++ as the final serving dataset. Final-refit predictions are not presented as held-out evaluation. The observed F1 gain's interval includes zero. The weak nine-emotion results and unrun human-microphone study are reported separately from final sarcasm performance. Generated upload checks are functional/ASR diagnostics, not emotion or sarcasm accuracy.

provenance.json records the source and final PDF hashes at verification time. Regenerate that record if the source results or PDF change. figure_data.json contains exact confusion counts and emotion class counts. Rendered QA images stay under ignored truetone/paper-build/. Only the final PDF is kept inside paper/.

## Reviewer audit

A subsequent audit clarified that the main benchmark uses dataset transcripts, whereas application uploads may use ASR. It added metric definitions, exact checkpoint identifiers and recorded software versions, the distinction between repeated out-of-fold probability averaging and the single final refit, and rare-class emotion failures (zero correct disgust/fear predictions, 1/48 frustration predictions). Whisper and scikit-learn references were added from their original publications. The five revised pages were rendered and inspected.

Remaining research limits cannot be repaired by wording: inner preprocessing is not fully nested, the F1-gain interval crosses zero, no untouched human-speaker benchmark or calibrated microphone study exists, upstream checkpoint revisions were not pinned, and pretraining overlap was not audited. A MUStARD++ mismatch-feature ablation has not been run. The document supports a transparent project comparison; it does not establish state-of-the-art or dependable real-world emotion recognition.
