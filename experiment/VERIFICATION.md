# Final MUStARD++ verification

Verified locally on 23 November 2026.

- The primary MUStARD++ notebook executed all seven code cells without errors: official/mirror download checks, baseline feature preparation, expanded feature preparation, paired evaluation and final refit.
- Active weights identify MUStARD++ with 1,202 unique targets. SHA-256: `b6f75922fbbd3f5324522d93a16c3f30c1ebeec66498f99fd7fcd661e58683a5`.
- The original MUStARD artifact remains separately archived with 690 targets. The isolated Desktop experiment clone retains no remote and was not modified by migration.
- All 20 Python tests and both JavaScript theme tests pass. Tests include recomputing all nine shared-target metric rows from the saved out-of-fold predictions.
- The local app API passed manual-transcript and automatic-ASR requests, both serving models, an added MUStARD++ clip, voice estimates, and invalid-input rejection. Warm requests took approximately 0.4-0.8 seconds on this Mac.
- Live feature predictions match cached dataset features within 1e-5 for the checked target in both serving modes. These checks use training clips and do not estimate accuracy.
- A dataset WAV uploaded through the installed Mac app produced its transcript, sarcasm score and voice estimate. Previous empty-state text disappeared during loading. Clearing restored the empty state.
- Closing the Mac app stopped its owned server and closed port 7861. The final Swift wrapper compiled and was installed in Applications. Project-folder recovery was exercised after rebuilding.
- Physical built-in microphone capture and analysis were subsequently verified on 24 November 2026 with the lid open; see the completed check below. The Enable Microphone command requests WebKit capture directly and stops its permission-check stream immediately. Browser/upload fallback remains available.
- All 12 final slides were rendered and visually checked. PPTX schema validation passed against the original. Cover, unaffected slides, layouts, relationships, themes and presentation settings were preserved; result images were regenerated from CSVs. The Desktop PPTX matches the repository copy.
- The three-page paper was rendered and checked. Authors, enrolments, PID-02, dataset counts, metrics and uncertainty match the experiment. The paper folder contains only the final PDF; extracted PDF text contains no nonbreaking, narrow or zero-width spaces.
- Dataset downloads, caches, personal recordings and document staging files are ignored and excluded from the commit.

The observed Text + speech F1 improvement is 0.7608 versus 0.7435. Its 95% paired interval includes zero; no conclusive superiority or live-microphone accuracy claim is made.

## Microphone follow-up: 24 November 2026

- macOS showed TrueTone microphone permission enabled. The launcher was rebuilt with explicit microphone/audio-input entitlements; tests verify that signing includes these and excludes camera access.
- Chrome successfully requested microphone access and captured a 15.18-second, 44.1 kHz mono WAV through the local interface. Every sample was zero; silence validation correctly rejected it.
- The hardware registry reported `AppleClamshellState = Yes`: the MacBook lid was closed. Apple disconnects the built-in microphone in that state. The README and silence error now explain opening the lid or choosing an external microphone.
- The closed-lid test could not verify non-silent inference. This was subsequently completed with the lid open, as recorded below. No captured audio is included in Git.

## Completed built-in microphone check: 24 November 2026

- macOS reported the lid open. Only the built-in MacBook microphone was used; AirPods were not selected.
- The installed desktop app recorded a non-silent 4.63-second, 44.1 kHz mono clip. Its UI returned a transcript, sarcasm estimate, four-class voice estimate, pitch and speaking-rate information. Both serving models also returned all expected outputs through the local API on that same clip. Warm requests were under one second on this Mac.
- Automatic transcription misheard part of the spoken test sentence. This verifies the recording/inference path, not speech-recognition, sarcasm or emotion accuracy. Users can enter the exact words before analysis.
- An earlier noisy capture caused Whisper to repeat a short phrase and report an implausible speaking rate. New validation rejects obvious automatic-transcript repetition or excessive output before feature extraction/classification, with instructions to rerecord or correct the transcript. The observed failure was replayed through the local API and correctly rejected. This guard is a heuristic; manually supplied transcripts are not subjected to its repetition check.
- All 21 Python tests pass, including the observed ASR failure and confirmation that rejected output never reaches the feature encoders. Training weights, dataset and evaluation metrics are unchanged.
- No personal recording, transcript or live prediction scores were added to the repository. The app remains open with the result for review; its server runs only while the app is open. Closing/restarting the app during this check again closed port 7861.

## Paper revision: 26 November 2026

The paper was expanded to five IEEE-style two-column pages with six tables and four figures derived from saved results. Both MUStARD and MUStARD++ use the same shared-target evaluation; full metric rows, final training C values, confusion matrices, paired intervals, held-out-show scores and the separate nine-emotion study are included. Generated uploads and physical microphone checks are distinguished from accuracy evaluation. No epoch-loss curves, calibrated microphone results or unsupported superiority claims were added. All five pages were rendered and inspected; PDF text contains no nonbreaking/narrow/zero-width spaces. Only the final PDF remains in paper/. Source hashes and figure provenance are recorded under results/paper/.
