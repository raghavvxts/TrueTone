# TrueTone

Research pipeline for multimodal sarcasm detection using utterance text and speech prosody on MUStARD.

## Project files

- [Research notebook](TrueTone.ipynb)
- [IEEE paper](paper/TrueTone_IEEE_Paper.pdf)
- [Presentation](presentations/TrueTone_Final_Presentation.pptx)
- [Completed experiment](results/local-run-2026-11-20/)

## Run

Python 3.12 recommended. The notebook installs its own dependencies; use an isolated environment.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name truetone
.venv/bin/python scripts/run_notebook.py
```

Alternatively, open `TrueTone.ipynb` in Jupyter and run all cells. A full run downloads MUStARD clips and four pretrained encoders. CUDA is used when available, otherwise CPU. The original notebook targets a Linux GPU host with 32 GB RAM and about 40 GB disk; CPU execution is substantially slower. ffmpeg is installed via apt on Linux or provided by imageio-ffmpeg as a fallback.

- `TRUETONE_DIR`: data/cache/output root; defaults to writable `/workspace/truetone` or `./truetone`.
- `TRUETONE_JOBS`: parallel classifier workers, default at most 8.
- `VAST_PRICE_PER_HOUR`: cost estimate only; does not provision a machine.
- Full execution writes report, figures, CSVs, OOF predictions and a ZIP under the data root. The executed notebook defaults to `truetone/executed.ipynb`.

## Local verification

```sh
.venv/bin/python -m unittest discover -s tests -v
```

This executes actual notebook prosody extraction on generated tone/silence audio, LR and gated-network training on synthetic features, train/test scaling isolation, bootstrap edge cases, notebook validation and supplied result consistency. It does not download or execute the pretrained encoders.

## Archived results

- The supplied experiment is archived in `results/`: report, result tables, figures, and predictions. Report branding has been standardised to TrueTone; numerical results are unchanged.
- Duplicate root-level ZIP/PDF files have been removed. Saved source-notebook outputs are cleared to distinguish the source from executed runs.
- `ANALYSIS.md`: code review, findings and execution limitations.

The top-level `results/` files are historical, not the output of the corrected notebook. No dataset, feature caches, or trained classifier weights were supplied. This is an experiment pipeline, not an inference service or web app.

Dataset and pretrained models retain their upstream terms; this repository does not bundle videos or model weights.

## Completed full run

On 20 November 2026, the complete notebook ran successfully on local CPU in 10.8 minutes: 690 MUStARD examples, 14 models, repeated CV and held-out-show evaluation, all 15 code cells without errors. Fresh results, report and executed notebook are in [`results/local-run-2026-11-20/`](results/local-run-2026-11-20/). Original supplied results remain in `results/`. Best mean CV F1 was 0.7655; best pooled cross-show F1 was 0.6584. See `ANALYSIS.md` for interpretation and limitations.

## Paper and presentation

- [IEEE paper PDF](paper/TrueTone_IEEE_Paper.pdf)
- [Final presentation](presentations/TrueTone_Final_Presentation.pptx)

The `paper/` folder contains only the final PDF, based on the completed local MUStARD run. The editable manuscript is maintained separately in Overleaf.
