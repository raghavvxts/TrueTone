# Team guide

Run commands from the repository root. Start with the root README; use PROJECT_GUIDE.md for the full research workflow.

## Repository map

| Folder/file | Purpose |
|---|---|
| app.py | Gradio interface and model selection |
| inference.py | Validation, ASR, frozen feature extraction and model inference |
| ui/ | Layout, OLED/light theme and style guide |
| desktop/ | Native macOS launcher and microphone entitlements |
| TrueTone.ipynb | Primary MUStARD++ research notebook |
| models/truetone_lr.joblib + manifest.json | Final serving bundle and integrity/provenance record |
| models/truetone_emotion.joblib | Optional experimental nine-class head |
| models/baseline/ and research/baseline/ | Original MUStARD comparison artifacts |
| experiment/ | Dataset preparation, evaluation and candidate-training workflows |
| scripts/ | Build, demo, document and robustness utilities |
| results/mustardpp/ | Final sarcasm evaluation |
| results/emotion/ | Experimental emotion evaluation |
| results/app-audit/ | API/synthetic stability checks, limitations and next steps |
| results/mustard-baseline/ | Historical results; not final model performance |
| presentations/ | Final PPTX only |
| paper/ | Final PDF only |
| tests/ | Automated checks |
| assets/ | App/web icons; previous logo retained in archive/ |
| truetone/, data/, .venv/ | Local ignored runtime/data/environment folders |

## Common tasks

| Task | Command / location |
|---|---|
| Run browser app | `.venv/bin/python app.py` |
| Build Mac app | `.venv/bin/python scripts/build_desktop_app.py /Applications/TrueTone.app` |
| Prepare local dataset demos | `.venv/bin/python scripts/prepare_demos.py` (requires prepared dataset) |
| Check generated speech | `.venv/bin/python scripts/check_generated_audio.py` (macOS say required) |
| Check dataset volume/noise variants | `.venv/bin/python scripts/check_demo_robustness.py` |
| Refit final sarcasm weights | `.venv/bin/python scripts/prepare_live_model.py` (requires prepared features) |
| Train experimental emotion head | `.venv/bin/python experiment/train_emotion.py` |
| Collect/train microphone candidates | Follow [microphone protocol](microphone-study.md) |
| Maintain final documents | scripts/update_presentation.py and scripts/build_paper.py; render and inspect after changes |

The requirements-ui.txt file is the normal installation entry point. requirements-verified.txt records an older local dependency snapshot, not a universal cross-platform lockfile.

## Keep final results reproducible

Do not edit weights, label names or feature functions independently. A changed feature signature invalidates existing artifacts and requires extraction/refitting plus evaluation. Experimental voice-mode selection deliberately does not alter sarcasm features. Preserve baseline provenance rather than renaming MUStARD artifacts to MUStARD++.

Before committing: run tests, inspect `git diff --check` and `git status`, confirm dataset/recordings/caches are ignored, and check changed documents visually. Never publish private recordings, rater annotations, transcripts or credentials. Do not treat demo predictions as independent test accuracy.

## Source bundle

A Git archive is a clean source/evaluation bundle: it excludes ignored recordings, caches, the installed application and the virtual environment. Teammates install dependencies after extracting it. It includes the PPTX and PDF. The Desktop ZIP supplied with this cleanup is a snapshot of the committed repository, not a complete offline installation.
