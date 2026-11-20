# Method

## 1. Task and data
Binary sarcasm detection on **MUStARD** (Castro et al., ACL 2019): 690 audiovisual utterances from *Friends*, *The Big Bang Theory*, *The Golden Girls* and
*Sarcasmaholics Anonymous* (345 sarcastic / 345 non-sarcastic; shows: FRIENDS (356), BBT (280), GOLDENGIRLS (40), SARCASMOHOLICS (14)).
Each item has the transcript of the target utterance, the preceding dialogue context, speaker, show, and the utterance video clip. We use the **utterance transcript and the utterance audio only**
(video frames are not used because the project studies text + speech prosody; context text is used only in one ablation).
Audio is extracted with ffmpeg to 16 kHz mono PCM (mean clip length 5.2 s).

## 2. Feature extraction (all encoders frozen; extracted once and cached)
| Block | Dim | Description |
|---|---|---|
| `text_emb` | 1024 | `roberta-large` mean-pooled last hidden layer of the utterance |
| `text_sent` | 3 | `cardiffnlp/twitter-roberta-base-sentiment-latest` posteriors [neg, neu, pos] ("what is said") |
| `textctx_emb` | 1024 | same encoder on the (utterance, context) sentence pair (ablation only) |
| `wavlm` | 2048 | `microsoft/wavlm-large`: mean+std over time of the average of the middle third of transformer layers |
| `prosody` | 58 | librosa: pYIN F0 (mean, P10/P90, std & range in semitones, slope, voiced fraction, jitter proxy), RMS energy (mean/std/max/dynamic range/slope, shimmer proxy), speaking rate (words/s from the transcript, articulation rate excluding pauses, onset rate), pauses (count, total, ratio, mean, max; silence gaps ≥150 ms at 30 dB), spectral centroid/flatness/rolloff/ZCR, 13 MFCC mean+std |
| `speech_emo` | 4 | `superb/wav2vec2-base-superb-er` posteriors over ['neu', 'hap', 'ang', 'sad'] ("how it is said") |
| `mismatch` | 6 | explicit incongruity features: text valence = P(pos)−P(neg); speech valence = Σ P(emotion)·v(emotion) with v(happy)=+1, v(angry/sad)=−1, v(neutral)=0; then valence difference, |difference|, product, and a polarity-flip flag (opposite signs, both |·|>0.1) |

## 3. Models (text-only vs speech-only vs multimodal)
* **Logistic regression (LR):** blocks standardised on the training fold only, clipped to ±5, each block scaled by 1/√dim so no block dominates, concatenated (early fusion); L2 strength chosen by inner 3-fold CV on the training fold (log-loss), balanced class weights.
* **Gated-fusion network (NET):** one projection (Dropout→Linear(128)→LayerNorm→GELU→Dropout(0.4)) per block; a sample-wise softmax gate over the modality blocks weights the projections; MLP head. AdamW (lr 0.001, wd 0.01), one-cycle schedule, 60 epochs, batch 64, label smoothing 0.1, average of 3 seeds. **Hyper-parameters were fixed a priori and not tuned on test folds.**
* **Text-only** = `text_emb`+`text_sent`; **Speech-only** = `wavlm`+`prosody`+`speech_emo`; **Multimodal** = all of them + `mismatch`.
* **Ablations:** remove mismatch / prosody / WavLM; prosody-only; mismatch-only (sentiment+emotion posteriors+mismatch); text with dialogue context.

## 4. Evaluation protocol
* **A. Repeated stratified 5-fold CV × 3 repeats** (random split; speakers/shows overlap between train and test — optimistic).
* **B. Cross-show test** (train on all other shows, test on each of BBT, FRIENDS — the only shows containing both classes; GOLDENGIRLS and SARCASMOHOLICS are ~100 % sarcastic and stay in training). The held-out show's speakers and acoustic conditions are unseen — the harder, more realistic setting.
* Metrics at threshold 0.5: accuracy, precision, recall, F1 (sarcastic = positive class), macro-F1; mean ± std over folds.
* **Significance:** paired bootstrap (2000 resamples) of ΔF1 on pooled out-of-fold predictions (first repeat).
* **Research-question analysis:** utterances are split by tertiles of |text valence − speech valence| (and by polarity flip) and models are compared per subset.

## 5. Reproducibility
Seed 42; GPU: NVIDIA GeForce RTX 3090 Ti; torch 2.14.1+cu132, transformers 5.23.0, librosa 1.0.1, scikit-learn 1.9.1; Python 3.12.15.
Total wall-clock of this notebook run: 10.4 min (≈ $0.04 at $0.25/h).
Data: https://github.com/soujanyaporia/MUStARD (annotations); video clips from the HuggingFace mirror `MichiganNLP/MUStARD`.
