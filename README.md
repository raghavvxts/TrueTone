# TrueTone / SarcasTone

Research pipeline for multimodal sarcasm detection using utterance text and speech prosody on MUStARD. The implementation retains the original SarcasTone name.

## Run

Python 3.12 recommended. The notebook installs its own dependencies; use an isolated environment.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name sarcastone
.venv/bin/python scripts/run_notebook.py
```

Alternatively, open `SarcasTone_vastaiClaude-2.ipynb` in Jupyter and run all cells. A full run downloads MUStARD clips and four pretrained encoders. CUDA is used when available, otherwise CPU. The original notebook targets a Linux GPU host with 32 GB RAM and about 40 GB disk; CPU execution is substantially slower. ffmpeg is installed via apt on Linux or provided by imageio-ffmpeg as a fallback.

- `SARCASTONE_DIR`: data/cache/output root; defaults to writable `/workspace/sarcastone` or `./sarcastone`.
- `SARCASTONE_JOBS`: parallel classifier workers, default at most 8.
- `VAST_PRICE_PER_HOUR`: cost estimate only; does not provision a machine.
- Full execution writes report, figures, CSVs, OOF predictions and a ZIP under the data root. The executed notebook defaults to `sarcastone/executed.ipynb`.

## Local verification

```sh
.venv/bin/python -m unittest discover -s tests -v
```

This executes actual notebook prosody extraction on generated tone/silence audio, LR and gated-network training on synthetic features, train/test scaling isolation, bootstrap edge cases, notebook validation and supplied result consistency. It does not download or execute the pretrained encoders.

## Supplied artifacts

- `results/`: unmodified extraction of `sarcastone_outputs.zip`, including the original report, result tables and figures.
- `Unknown-3 (1).pdf`: byte-identical copy of `results/SarcasTone_Report.pdf`.
- Original ZIP/PDF remain for provenance. Saved notebook outputs have been cleared to avoid confusing earlier execution with current verification.
- `ANALYSIS.md`: code review, findings and execution limitations.

Saved results are historical, not the output of the corrected notebook. No dataset, feature caches, or trained classifier weights were supplied. This is an experiment pipeline, not an inference service or web app.

Dataset and pretrained models retain their upstream terms; this repository does not bundle videos or model weights.

## Completed full run

On 20 November 2026, the complete notebook ran successfully on local CPU in 10.8 minutes: 690 MUStARD examples, 14 models, repeated CV and held-out-show evaluation, all 15 code cells without errors. Fresh results, report and executed notebook are in [`results/local-run-2026-11-20/`](results/local-run-2026-11-20/). Original supplied results remain in `results/`. Best mean CV F1 was 0.7655; best pooled cross-show F1 was 0.6584. See `ANALYSIS.md` for interpretation and limitations.

## Paper and presentation

- [IEEE paper PDF](paper/TrueTone_IEEE_Paper.pdf)
- [Editable LaTeX source](paper/main.tex), bibliography and generated result tables in `paper/`
- [Corrected presentation](presentations/TrueTone_Final_Presentation.pptx)

The paper uses the completed local MUStARD run. To regenerate its tables and vector figure, run `.venv/bin/python paper/generate_results.py`. Compile `paper/main.tex` using IEEEtran and BibTeX. Source text uses ordinary ASCII spaces.
