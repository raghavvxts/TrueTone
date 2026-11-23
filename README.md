# TrueTone

<img src="assets/logo-192.png" width="88" height="88" alt="TrueTone speech and waveform logo">

Research pipeline for multimodal sarcasm detection using utterance text and speech prosody on MUStARD.

## Project files

- [Research notebook](TrueTone.ipynb)
- [IEEE paper](paper/TrueTone_IEEE_Paper.pdf)
- [Presentation](presentations/TrueTone-PID-02.pptx)
- [Completed experiment](results/local-run-2026-11-20/)

## Microphone demo

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-ui.txt
.venv/bin/python app.py
```

Open **http://127.0.0.1:7860** in Chrome, Edge or Safari. Allow microphone access, record 1–30 seconds of English speech, stop, then click **Analyze recording**. You can also upload audio. Leave the optional transcript blank for local Whisper transcription, or type the exact spoken words to correct it. Selecting a new recording clears the previous transcript and result.

The optional transcript and model selector are under **Adjust transcript or model**.

The demo reports **Sarcastic / Not sarcastic**, both model scores and the transcript used. It runs after recording stops, rather than continuously streaming predictions. If an embedded browser cannot detect your microphone, open the URL in your normal browser. Silence and recordings outside the duration limit are rejected.

The included `models/truetone_lr.joblib` contains final multimodal and speech-reference logistic-regression classifiers fitted on MUStARD. These are final refits, distinct from the fold-trained research evaluation models. The UI does not use the gated network. Speech-reference features also need transcript word counts. Model scores are uncalibrated estimates; live speech and automatic transcription were not part of the reported CV benchmark.

The first inference downloads the frozen Hugging Face encoders and `openai/whisper-tiny.en` (several GB in total). After downloads, inference is local and audio is not sent to a speech API. Gradio temporary uploads are cleaned periodically. Browser microphone access requires localhost or HTTPS. This app binds to localhost and does not create a public sharing link.

To retrain the final classifiers from the actual notebook data/feature pipeline:

```sh
.venv/bin/python scripts/prepare_live_model.py
```

This downloads MUStARD if needed, reuses valid feature caches, and fits the final LR models without rerunning the evaluation protocols. Only load trusted local joblib bundles. `TRUETONE_MODEL` overrides the model path; `TRUETONE_PORT` overrides the default 7860 port. Pretrained caches use `HF_HOME` or the local `truetone/hf` directory.

## macOS Desktop app

Double-click **TrueTone.app** on your Desktop. The app starts the local Python server in the background, shows elapsed startup time, and opens the website when it is ready. No Terminal commands are needed for normal use. The app runs its own server on port 7861 only while open. Closing its window or choosing Quit stops that server, including an analysis in progress. A parent-process watchdog also shuts it down if the app is force-quit. Nothing starts at login or stays running after exit.

Choose **TrueTone > Enable Microphone** if the input device is missing, and grant microphone access. **TrueTone > Open in Browser** is available as a fallback. The first analysis may still take time while encoders load; its progress appears in the website.

The app uses this project folder and its `.venv`; keep both in place. If you move the folder, rebuild the launcher:

```sh
.venv/bin/python scripts/build_desktop_app.py
```

To update an app moved into Applications, run `.venv/bin/python scripts/build_desktop_app.py /Applications/TrueTone.app`. Choose **TrueTone > Reload Workspace** (Cmd+R) if the page fails to render. The wrapper uses a fresh web session on launch and attempts one automatic recovery if the renderer stops.

Building requires macOS with Xcode Command Line Tools. Server startup errors are logged in `~/Library/Logs/TrueTone/server.log`. The app stays local; it needs no Vercel deployment or public tunnel.

## Run the research experiment

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

The top-level `results/` files are historical, not the output of the corrected notebook. No dataset, feature caches, or trained classifier weights were supplied. The original notebook is an experiment pipeline. A local microphone demo is available in the section above.

Dataset and pretrained models retain their upstream terms; this repository does not bundle videos or pretrained encoder weights. The final LR classifier weights are included for the local demo.

## Completed full run

On 20 November 2026, the complete notebook ran successfully on local CPU in 10.8 minutes: 690 MUStARD examples, 14 models, repeated CV and held-out-show evaluation, all 15 code cells without errors. Fresh results, report and executed notebook are in [`results/local-run-2026-11-20/`](results/local-run-2026-11-20/). Original supplied results remain in `results/`. Best mean CV F1 was 0.7655; best pooled cross-show F1 was 0.6584. See `ANALYSIS.md` for interpretation and limitations.

## Paper and presentation

- [IEEE paper PDF](paper/TrueTone_IEEE_Paper.pdf)
- [Final presentation](presentations/TrueTone-PID-02.pptx)

The `paper/` folder contains only the final PDF, based on the completed local MUStARD run. The editable manuscript is maintained separately in Overleaf.

UI colors, spacing and typography are defined in `ui/style.css`; design conventions are documented in `ui/STYLE_GUIDE.md`. Both light and dark modes use the same shared design tokens.

Each analyzed clip also reports estimated voice delivery (neutral, happy, angry, or sad), all four emotion scores, mean pitch when measurable, and transcript-based speaking pace. Ambiguous scores are marked mixed/uncertain. These are uncalibrated model estimates, not evidence of a speaker’s actual feelings or identity. Changing audio, transcript, or model clears the previous result.

## Live prediction limitations and cue contrast

The fullscreen UI uses the available window width. Detailed emotion scores are available under a disclosure so the main result remains readable.

A clip can contain positive wording and negative vocal delivery yet still receive a low sarcasm score from the learned classifier. The interface now flags strong opposing polarities and shows **Conflicting cues — review context** when the classifier is below its 0.5 sarcasm threshold. This is a separate, unvalidated review heuristic: text valence must exceed 0.5 and voice valence be below -0.5 (or the reverse). Text valence is P(positive) - P(negative); voice valence is P(happy) - P(angry) - P(sad). It does not override or recalibrate classifier scores, and is available only in Text + speech mode. No accuracy improvement is claimed from adding this flag.

The emotion encoder has only four categories. A sad-like match can overlap frustrated or subdued delivery; frustration is not classified separately. Reliable improvement on everyday recordings requires diverse labeled recordings and held-out evaluation, rather than a phrase-specific rule or threshold tuned to one recording. The classifier weights and research results are unchanged.

The header’s sun/moon **Dark mode** switch switches themes without reloading or clearing the current recording and result. OLED dark mode uses a pure black page, neutral charcoal surfaces, and light text. Browsers remember the selected theme locally; the native app starts a fresh web session and follows the system appearance initially. Theme behavior can be checked with `node --test tests/test_theme.cjs`.
