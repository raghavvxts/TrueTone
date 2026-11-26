# TrueTone

<img src="assets/logo-192.png" width="80" height="80" alt="TrueTone speech and waveform logo">

**PID-02 · Speech and text sarcasm detection using MUStARD++**

Record or upload English speech to receive a sarcasm estimate, a transcript and a voice-delivery estimate. Processing runs locally after pretrained models download. This is a research demo; scores are not verified emotion or sarcasm labels.

## For our professor

| Material | Open |
|---|---|
| Final presentation | [TrueTone-PID-02.pptx](presentations/TrueTone-PID-02.pptx) |
| Research paper | [TrueTone_IEEE_Paper.pdf](paper/TrueTone_IEEE_Paper.pdf) |
| Main research notebook | [TrueTone.ipynb](TrueTone.ipynb) |
| Evaluation and limitations | [MUStARD++ report](results/mustardpp/report.md) |
| Model identity | [Serving manifest](models/manifest.json) |
| Showcase walkthrough | [Professor demo guide](docs/SHOWCASE.md) |

**Public repository:** [github.com/raghavvxts/TrueTone](https://github.com/raghavvxts/TrueTone)

## Run the app

Use **Python 3.12**. From the repository folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ui.txt
python app.py
```

Open **http://127.0.0.1:7860**. Initial use downloads several GB of pretrained weights; later processing is local. The server binds only to localhost. Stop it with Ctrl+C.

On Raghav's configured Mac, open **TrueTone.app** from Applications. It starts its owned server on port 7861 and stops it on exit. The macOS app is built locally, not included as a portable preinstalled application. To build it on another Mac after installing dependencies:

```sh
.venv/bin/python scripts/build_desktop_app.py /Applications/TrueTone.app
```

Keep the repository and `.venv` in place. Building requires Xcode Command Line Tools. Windows/Linux users can use the browser app.

1. Record or upload 1–30 seconds of English speech. Allow microphone access and keep the MacBook lid open.
2. Stop, then click **Analyze recording**.
3. Check **Transcript used**. Correct misheard words under **Adjust transcript or model** and analyze again.
4. Read the model estimates. **Clear** before the next speaker.

Light mode is the default; the sun/moon switch enables OLED dark mode. WAV, MP3 and M4A decoding uses SoundFile or bundled imageio-ffmpeg. No system ffprobe installation is needed.

## What is final, and what is experimental?

| Component | Status |
|---|---|
| Sarcasm classifier | **Final:** frozen encoders + logistic regression, refitted on all 1,202 MUStARD++ targets (601 per class) |
| Existing four-class voice detector | Default pretrained, frozen detector; uncalibrated |
| MUStARD++ nine-emotion head | **Experimental opt-in:** trained on surface-emotion labels; not proven reliable on live speakers |
| Microphone adaptation/calibration | Workflow implemented; awaiting consented, human-labelled recordings |
| Original MUStARD | Archived baseline only; rejected by the final app's dataset guard |

The final classifier uses target audio and transcripts, not video frames or preceding dialogue. Generic pretrained encoders are not an old MUStARD classifier. Selecting the experimental emotion head changes displayed emotion scores, not the evaluated sarcasm pipeline.

The paired shared-target experiment reports **F1 0.761** for MUStARD++ versus **0.744** for MUStARD; corresponding MUStARD++ accuracy is **0.751**. The difference's 95% interval includes zero. These results do not establish accuracy on your microphone or a new speaker. Training-set demos and generated speech are not independent accuracy tests.

The PPT describes the final sarcasm study. The updated paper includes that study, both dataset baselines, and separate sections for the experimental emotion head and application checks. It clearly identifies the unrun microphone-adaptation study and does not present synthetic demos as accuracy evidence.

## For the team

- [Repository map and maintenance tasks](docs/TEAM_GUIDE.md)
- [Full setup, model details and research reproduction](docs/PROJECT_GUIDE.md)
- [Private microphone collection, training and calibration](docs/microphone-study.md)
- [App audit, generated-audio checks and next improvements](results/app-audit/README.md)
- [Nine-emotion training results](results/emotion/README.md)
- [Historical verification record](experiment/VERIFICATION.md)

Run automated checks:

```sh
.venv/bin/python -m unittest discover -s tests -v
node --test tests/test_theme.cjs
```

Optional local dataset demos require preparing the dataset first; follow the full project guide, then run `.venv/bin/python scripts/prepare_demos.py` and restart the app. Expand **Showcase demos · sarcasm and emotions**. These 18 examples were seen during final training and are labelled accordingly.

## Team

| Member | Enrolment |
|---|---|
| Manas Sharma | S24CSEU2420 |
| Raghav Vats | S24CSEU1665 |
| Vaibhav Singh | S24CSEU1668 |
| Sumit Bhati | S24CSEU2870 |

Bennett University, Greater Noida · Instructor: Dr. Saurabh Kumar Srivastava

## Data and distribution

Dataset downloads, pretrained encoder caches, environments, generated speech and personal recordings stay outside Git. The repository includes code, final LR weights, experimental emotion-head weights, results, final PPTX and the paper PDF. The `paper/` folder contains only that PDF.

Sources: [MUStARD++ annotations](https://github.com/cfiltnlp/MUStARD_Plus_Plus), [MUStARD++ paper](https://aclanthology.org/2022.lrec-1.756/), [MUStARD baseline](https://github.com/soujanyaporia/MUStARD), [video mirror](https://huggingface.co/datasets/sifan077/MUStARD_plus_plus). Dataset and pretrained models retain upstream terms; inclusion of project code does not relicense them. Local demo audio is not bundled for redistribution.
