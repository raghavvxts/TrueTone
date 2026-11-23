# TrueTone

<img src="assets/logo-192.png" width="88" height="88" alt="TrueTone speech and waveform logo">

**PID-02 · Multimodal sarcasm detection on MUStARD++ using text and speech prosody.**

The final app uses logistic-regression classifiers fitted on **1,202 MUStARD++ target utterances**, balanced at 601 sarcastic and 601 non-sarcastic. Text, sentiment, speech and emotion encoders remain frozen. The original 690-utterance MUStARD experiment is retained as the baseline, with its original weights and results clearly separated.

| Team member | Enrolment |
|---|---|
| Manas Sharma | S24CSEU2420 |
| Raghav Vats | S24CSEU1665 |
| Vaibhav Singh | S24CSEU1668 |
| Sumit Bhati | S24CSEU2870 |

Bennett University, Greater Noida, India · Instructor: Dr. Saurabh Kumar Srivastava

## Evaluation files

- [Final presentation](presentations/TrueTone-PID-02.pptx)
- [IEEE-format paper PDF](paper/TrueTone_IEEE_Paper.pdf) — the only file in `paper/`
- [MUStARD++ research notebook](TrueTone.ipynb)
- [Full paired comparison and limitations](results/mustardpp/report.md)
- [Verification record](experiment/VERIFICATION.md)
- [Dataset audit](experiment/data_audit.json) and [active model manifest](models/manifest.json)
- [MUStARD baseline notebook](research/baseline/TrueTone_MUStARD.ipynb), [weights](models/baseline/), and [historical results](results/mustard-baseline/)

## Use the local app

On this Mac, open **TrueTone.app** in Applications. It starts its own local server on port 7861 and shows elapsed startup time. Closing the window or quitting stops the server; a watchdog also handles force-quit. Nothing runs at login or remains running after exit.

1. Record or upload **1–30 seconds of English speech**.
2. Stop recording, then choose **Analyze recording**.
3. Read the sarcasm estimate, transcript and voice-delivery estimate. Expand **Voice emotion scores** for details.

Leave the optional transcript blank for local Whisper transcription, or enter the exact spoken words under **Adjust transcript or model**. New audio clears the previous transcript and results. Choose **TrueTone > Enable Microphone** if needed, or **Open in Browser** for a browser fallback. Microphone access requires permission and localhost or HTTPS.

Fresh launches use light mode. The sun/moon switch enables OLED dark mode without clearing the recording. Previous output is hidden while analysis runs.

To launch from source on another machine:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-ui.txt
.venv/bin/python app.py
```

Open **http://127.0.0.1:7860**. Initial use downloads several GB of frozen encoder weights and `openai/whisper-tiny.en`. Afterwards audio processing is local; recordings are not sent to a speech API. Models stay in RAM while the process is open, speeding up later clips, and are released on exit. Timing depends on hardware and clip length. The app binds to localhost and does not create a public sharing link.

The macOS wrapper uses this folder and its `.venv`; keep both in place. Rebuild after moving the folder:

```sh
.venv/bin/python scripts/build_desktop_app.py /Applications/TrueTone.app
```

Building requires Xcode Command Line Tools. Startup errors appear in `~/Library/Logs/TrueTone/server.log`. **Reload Workspace** (Cmd+R) reloads the page. The loading screen waits for the interface to render and retries once if needed.

After rebuilding the local app, macOS may require renewed project access. Use **TrueTone > Grant Project Folder Access**, select this TrueTone folder, then reopen the app. **Enable Microphone** requests both macOS and WebKit audio permission and immediately releases the permission-check stream; recording starts only with the Record button.

## What is actually trained

`models/truetone_lr.joblib` is the active **MUStARD++** bundle. It contains **Text + speech** and **Speech reference** classifiers. `models/baseline/truetone_lr.joblib` preserves the original MUStARD bundle; neither artifact is an out-of-fold evaluation model.

| Feature block | Dimensions |
|---|---:|
| RoBERTa-large target-text embedding | 1,024 |
| CardiffNLP text sentiment | 3 |
| WavLM-large speech embedding | 2,048 |
| Acoustic/prosodic statistics | 58 |
| SUPERB speech emotion probabilities | 4 |
| Text–voice valence mismatch | 6 |

Blocks are standardized, clipped to [-5, 5] and divided by the square root of their dimension before concatenation. Classifiers use class-balanced logistic regression, a 0.5 prediction threshold and three-fold selection of C from `[0.1, 0.5, 2, 10, 50, 300, 1000]` by log loss. Encoders are not fine-tuned. **The final app does not use a gated network.** MUStARD++ emotion/valence/arousal labels are not fed into the classifier; emotion features come from the frozen audio encoder. No personal recordings enter training or evaluation.

Speech reference includes transcript-based speaking-rate features, so it is not completely independent of text. The emotion encoder supports neutral, happy, angry and sad only. Frustration is not a separate class; uncertain outputs are marked mixed. Scores are uncalibrated estimates, not evidence of a person's feelings or a verified sarcasm label.

The optional conflicting-cues flag is a separate, unvalidated polarity heuristic. It does not change the classifier score. It triggers for text valence above 0.5 and voice valence below -0.5, or the reverse. Text valence is P(positive) − P(negative); voice valence is P(happy) − P(angry) − P(sad).

## Does MUStARD++ perform better?

The paired experiment tests the **same 688 shared targets** using three repetitions of five stratified grouped folds, seeds 42–44. Shared IDs, identical normalized target text and identical decoded audio are grouped across both corpora and excluded from each outer test fold's training sets. Each clip's three out-of-fold probabilities are averaged before computing these metrics:

| Model | MUStARD baseline F1 | MUStARD++ F1 |
|---|---:|---:|
| Text only | 0.660 | 0.686 |
| Speech reference | 0.724 | 0.724 |
| Text + speech | 0.744 | **0.761** |

For MUStARD++ Text + speech, accuracy is **0.751**, macro-F1 **0.751**, precision **0.733** and recall **0.791**. F1 is not accuracy.

The observed multimodal F1 gain is **+0.017**, with a 95% paired group-bootstrap interval of **[-0.007, +0.042]** (2,000 samples). The interval includes zero: this experiment favors MUStARD++ but does not conclusively establish superiority. A size-matched MUStARD++ training control reaches F1 **0.743**, suggesting that additional training volume matters. We select MUStARD++ as the final dataset to satisfy the project requirement, while reporting this uncertainty.

Shared test features use MUStARD++ transcripts, including six corrected shared transcripts. The expanded release has 514 added and two removed IDs relative to the baseline. Outer-fold scalers see only outer training data, but are reused during inner C selection, so inner preprocessing is not fully nested. Grouping exact duplicates cannot remove all related dialogue, episode or speaker effects. This is a development-set comparison, not an untouched external benchmark.

A separate evaluation holds out each of the five shows across all 1,202 targets. Results vary by show. Golden Girls has 39 sarcastic targets out of 40; Sarcasmoholics has 14 out of 14. Their high or unstable F1 scores need caution. See [per-show scores](results/mustardpp/held_out_show.csv). Scripted TV results do not establish accuracy on live microphones or unseen speakers.

## Reproduce training and evaluation

Install `requirements-ui.txt` as above. Run from this repository root:

```sh
.venv/bin/python experiment/download.py
.venv/bin/python experiment/prepare_baseline.py
.venv/bin/python experiment/prepare.py
.venv/bin/python experiment/compare.py
```

The downloader checks official annotations against the video mirror and verifies archive SHA-256 values. Preparation extracts target audio at 16 kHz mono and caches frozen features. Comparison writes CSVs, out-of-fold predictions, bootstrap intervals and the report to `results/mustardpp/`, then refits the final app bundle on all 1,202 targets. It leaves the archived baseline model intact. Downloads and private caches stay under ignored `data/` and `truetone/`; they are not committed.

For a final refit from already prepared MUStARD++ features, without rerunning evaluation:

```sh
.venv/bin/python scripts/prepare_live_model.py
```

Alternatively, run all cells of `TrueTone.ipynb`. For headless execution, register the kernel with `.venv/bin/python -m ipykernel install --user --name truetone`, then run `.venv/bin/python scripts/run_notebook.py`. The executed notebook stays under ignored `truetone/`.

`TRUETONE_PORT` overrides the standalone port. `TRUETONE_MODEL` can select a trusted local model bundle. `HF_HOME` controls the pretrained-model cache; default is `truetone/hf`. Only load trusted joblib files.

## Verification and document maintenance

```sh
.venv/bin/python -m unittest discover -s tests -v
node --test tests/test_theme.cjs
```

Tests cover audio validation/resampling, model serialization, feature compatibility, voice estimates, microphone/upload configuration, desktop lifecycle, notebook validity, baseline result consistency and the final MUStARD++ artifact/results. Encoder inference is checked separately using local dataset audio; no private recording is committed. Uploaded-audio inference is verified in the installed app. A physical microphone recording still requires the macOS/WebKit permission check documented in the verification record.

The final deck retains the supplied design, cover, team details and public repository link. Its charts derive from the saved evaluation CSVs. Maintenance scripts `scripts/update_presentation.py` and `scripts/build_paper.py` regenerate the final documents; slide maintenance additionally needs `defusedxml` (`uv pip install --python .venv/bin/python defusedxml`). The paper source is in the build script, outside `paper/`.

## Sources

- [Official MUStARD++ annotations](https://github.com/cfiltnlp/MUStARD_Plus_Plus), associated with [Ray et al., LREC 2022](https://aclanthology.org/2022.lrec-1.756/)
- [MUStARD baseline](https://github.com/soujanyaporia/MUStARD), introduced by [Castro et al., ACL 2019](https://aclanthology.org/P19-1455/)
- [MUStARD++ video mirror](https://huggingface.co/datasets/sifan077/MUStARD_plus_plus)

Dataset and pretrained models retain upstream terms. The repository contains final LR weights, code, PPTX and paper PDF; it does not bundle source videos or frozen encoder checkpoints.
