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
- Physical microphone capture remains pending macOS/WebKit authorization in this session; no successful live recording is claimed. The Enable Microphone command now requests WebKit capture directly and stops its permission-check stream immediately. Browser/upload fallback remains available.
- All 12 final slides were rendered and visually checked. PPTX schema validation passed against the original. Cover, unaffected slides, layouts, relationships, themes and presentation settings were preserved; result images were regenerated from CSVs. The Desktop PPTX matches the repository copy.
- The three-page paper was rendered and checked. Authors, enrolments, PID-02, dataset counts, metrics and uncertainty match the experiment. The paper folder contains only the final PDF; extracted PDF text contains no nonbreaking, narrow or zero-width spaces.
- Dataset downloads, caches, personal recordings and document staging files are ignored and excluded from the commit.

The observed Text + speech F1 improvement is 0.7608 versus 0.7435. Its 95% paired interval includes zero; no conclusive superiority or live-microphone accuracy claim is made.

## Microphone follow-up: 24 November 2026

- macOS showed TrueTone microphone permission enabled. The launcher was rebuilt with explicit microphone/audio-input entitlements; tests verify that signing includes these and excludes camera access.
- Chrome successfully requested microphone access and captured a 15.18-second, 44.1 kHz mono WAV through the local interface. Every sample was zero; silence validation correctly rejected it.
- The hardware registry reported `AppleClamshellState = Yes`: the MacBook lid was closed. Apple disconnects the built-in microphone in that state. The README and silence error now explain opening the lid or choosing an external microphone.
- A non-silent physical microphone inference still requires opening the lid or selecting a working external input. It has not yet been verified. No captured audio is included in Git.
