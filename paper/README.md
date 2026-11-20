# TrueTone IEEE paper

This is an editable IEEEtran conference-style manuscript, prepared from the official IEEE conference template on Overleaf:
https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn

## Open in Overleaf

Upload the project ZIP as a new project. Set `main.tex` as the main document and select pdfLaTeX. Overleaf runs BibTeX automatically. The included IEEEtran.cls is the unmodified class distributed through CTAN; its original license is retained in that file. Standard packages and IEEEtran.bst are provided by Overleaf.

## Files

- main.tex: complete manuscript, with the four authors and Bennett University affiliation.
- references.bib: cited papers, exact model cards and repository.
- table-main.tex and table-ablations.tex: tables generated directly from the completed local run's CSVs.
- figures/model-comparison.pdf: vector research figure.
- IEEEtran.cls: original IEEE conference/journal class.

## Results and provenance

All experiment results refer to `results/local-run-2026-11-20/` in the TrueTone repository, commit dc4a74b. The original November 19 report contains a different run and has not been mixed into these results. Evaluation used original MUStARD, 690 examples. MUStARD++ appears as related work and future evaluation, not as completed experimental data.

Main numbers: mean random-split CV F1 0.765474 for gated fusion with mismatch; best pooled held-out-show F1 0.658368 for speech LR. CV means and pooled cross-show metrics use different aggregation rules. Bootstrap intervals are exploratory, and speech-only configurations use transcript word rates.

The prose uses ordinary ASCII spaces. Source files were checked for unusual Unicode whitespace and unwanted zero-width characters. TeX commands control mathematical and typographic spacing in the PDF.

Local build: `tectonic main.tex` (also compatible with pdfLaTeX/BibTeX on Overleaf).
