# Paper figures and evidence

Updated 26 November 2026. The final five-page PDF is in paper/TrueTone_IEEE_Paper.pdf. It uses an IEEE-style two-column layout; it is not a claim of publication or acceptance by IEEE.

Rebuild from repository root:

```sh
.venv/bin/python scripts/build_paper.py
```

The builder calls scripts/paper_figures.py. Four figures derive from saved held-out predictions and metric files: pooled F1 comparison, two sarcasm confusion matrices, paired F1 intervals and the nine-class emotion confusion matrix. Six tables cover targets, feature blocks, F1, all five classification metrics, held-out shows and the separate emotion experiment. No training curves or epoch loss histories are fabricated.

The primary comparison retains original MUStARD results and the size-matched control, while identifying MUStARD++ as the final serving dataset. Final-refit predictions are not presented as held-out evaluation. The observed F1 gain's interval includes zero. The weak nine-emotion results and unrun human-microphone study are reported separately from final sarcasm performance. Generated upload checks are functional/ASR diagnostics, not emotion or sarcasm accuracy.

provenance.json records the source and final PDF hashes at verification time. Regenerate that record if the source results or PDF change. figure_data.json contains exact confusion counts and emotion class counts. Rendered QA images stay under ignored truetone/paper-build/. Only the final PDF is kept inside paper/.
