# Professor showcase

## Before the session

- Open TrueTone once with internet available to finish model downloads. Run a short analysis, then close it. First-run downloads and cold loading are slower than subsequent analyses.
- Keep the MacBook lid open, choose its built-in microphone and allow microphone permission. Do not rely on AirPods being selected automatically.
- Have the final PPTX, paper PDF and public GitHub README ready. The app is local; the GitHub URL is not a hosted inference website.

## Suggested walkthrough

1. Show the README's final/experimental table and the model manifest: 1,202 MUStARD++ targets, balanced sarcasm classes, frozen encoders plus logistic regression.
2. Expand **Showcase demos · sarcasm and emotions**, choose one of 18 recordings and click **Load selected demo**. Play it in the audio player, then analyze. Explain that its reference transcript is supplied and it was seen during the final refit. This demonstrates the pipeline, not generalization.
3. Clear. Invite a fresh 3–10 second English recording. Keep ordinary volume and laptop distance, stop, and analyze.
4. Inspect the transcript before interpreting the prediction. If misheard, enter the exact spoken words and rerun. State that this corrected-transcript run is different from fully automatic inference.
5. Show the sarcasm estimate and voice category separately. A high probability can be wrong; emotion labels describe model matches, not a person's verified feelings.
6. If requested, select the experimental MUStARD++ nine-class head. Explain its weak held-out-show results and that it has not been validated on new microphone speakers. Switching it does not change sarcasm predictions.
7. Show held-out evaluation and limitations. The final serving model was refitted on all targets; its own training predictions are not the reported test results.
8. Quit the Mac app to stop its owned server. For the source/browser launch, use Ctrl+C in its terminal.

## If the instructor's result is wrong

Do not relabel their recording using the model output or retrain on it during evaluation. Check the transcript, recording quality and missing conversation context. Discuss the TV-to-microphone domain gap and the speaker-separated training/calibration/test protocol. A candid error is more useful than a falsely guaranteed demo.

## What is currently verified

The latest recorded checks include 32 Python tests, two JavaScript theme tests, both emotion modes through the app API, and 64 generated-audio inference requests. Synthetic checks are not human accuracy measurements. Earlier built-in microphone capture was verified; the showcase selector, loading, playback, experimental mode and completed analysis have now been visually checked. Clear-during-analysis still requires its dedicated replay. See the app audit for details and dates.
