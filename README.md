# TrueTone

**PID-02 · MUStARD++ sarcasm detection using text and speech**

## Start here

### [Open TrueTone.ipynb](TrueTone.ipynb)

This is the main notebook for the project discussion. It explains the dataset, frozen feature extractors, logistic-regression training, grouped evaluation and final MUStARD++ model. Saved result tables are included so they can be reviewed without downloading the dataset or retraining.

**Run All defaults to review mode.** It reads committed results and shows the training implementation. Set `RETRAIN = True` only to download/prepare the datasets and rerun the complete experiment; this can require several GB and substantial time.

## Measured results

| Shared-target Text + speech | MUStARD baseline | Final MUStARD++ |
|---|---:|---:|
| Sarcastic-class F1 | 0.744 | 0.761 |
| Accuracy | 0.741 | 0.751 |

Both conditions are evaluated on the same 688 shared targets using repeated grouped folds. The observed F1 gain's confidence interval includes zero. The final app weights are refitted on all 1,202 MUStARD++ targets (601 per class); their own training predictions are not a test-accuracy estimate.

The classifier uses audio and target transcripts. Pretrained encoders remain frozen; there is no gated network in the final study. The separate nine-emotion head is experimental and has not demonstrated reliable live-speaker accuracy.

## Submission files

- [Research notebook](TrueTone.ipynb)
- [Final presentation](presentations/TrueTone-PID-02.pptx)
- [Research paper](paper/TrueTone_IEEE_Paper.pdf)
- [Evaluation evidence](results/mustardpp/report.md)

## Supporting code

The notebook calls `experiment/` for dataset preparation and evaluation. `inference.py`, `models/`, `app.py`, `ui/`, `desktop/` and `scripts/` support the working local app and reproduction. `research/baseline/` and `models/baseline/` preserve the original MUStARD comparison. These are dependencies/evidence, not extra presentation notebooks to demonstrate.

- [Team setup and repository map](docs/TEAM_GUIDE.md)
- [Full technical guide](docs/PROJECT_GUIDE.md)
- [Showcase walkthrough](docs/SHOWCASE.md)

To run the browser app with Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ui.txt
python app.py
```

Open http://127.0.0.1:7860. First use downloads pretrained models. On the configured Mac, TrueTone.app starts its server on opening and stops it on exit. Personal recordings, datasets, caches and environments stay ignored by Git.

## Team

| Member | Enrolment |
|---|---|
| Manas Sharma | S24CSEU2420 |
| Raghav Vats | S24CSEU1665 |
| Vaibhav Singh | S24CSEU1668 |
| Sumit Bhati | S24CSEU2870 |

Bennett University, Greater Noida · Instructor: Dr. Saurabh Kumar Srivastava

[Public repository: raghavvxts/TrueTone](https://github.com/raghavvxts/TrueTone). Dataset and pretrained-model terms remain with their upstream owners; see the technical guide for sources.
