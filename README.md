# A6: Speech Processing — Results & Findings

This repository contains the completed notebook and resources for **Assignment 6 (Speech Processing)**. The assignment investigates the core components of speech systems, including speech tokenization, Connectionist Temporal Classification (CTC) alignment, self-supervised representations with wav2vec 2.0, and voice cloning using OpenVoice V2.

---

## 1. Key Results & Findings

### Part 1: Speech Tokenization
We explored text-to-phoneme and emotion/accent tokenization. In this model, emotion/accent tags (e.g., `[HAPPY]`, `[SAD]`) are mapped to single token IDs instead of being broken into individual letters.
* **Findings**: Injecting a single discrete token ID (e.g., `[HAPPY]` = ID 37) functions similarly to the `[CLS]` token in BERT. This single token can condition the entire output prosody through cross-attention, allowing for direct control over synthesis style.

---

### Part 2: Mel Spectrogram & CTC Alignment
We implemented the CTC collapsing function and log-space forward algorithm from scratch:
* **CTC Collapsing**: Verified how adjacent duplicates are merged and blanks (`_`) are removed (e.g., `HHEELLLLOO` collapses to `HELO`, whereas `H_EE_LL_LO` collapses to `HELLO`). Blanks are critical to prevent separate identical characters from merging.
* **Forward Algorithm**: Evaluated the probability of the label `"HEL"` over a 6-frame logit grid. The dynamic programming algorithm successfully summed over all valid alignment paths (e.g., `H_EL__`, `HH_ELL`, `_HEELL`) without explicit path enumeration, yielding:
  $$\log P_{\text{CTC}}(\text{"HEL"}) \approx -5.5957 \quad \left(P_{\text{CTC}} \approx 0.003714\right)$$
* **Synthetic ASR Task**: Trained a `TinyCTCModel` (BiLSTM + Linear projection) on a synthetic frame-to-character speech recognition task. Over 300 steps of training, the CTC loss decreased from **`2.0042`** down to **`0.0175`**, demonstrating rapid convergence.

---

### Part 3: wav2vec 2.0 Representation Learning
We evaluated a pretrained, frozen `wav2vec2-base` encoder (94M parameters) on a 4-way word classification task using the `SpeechCommands` dataset (`yes`, `no`, `stop`, `go`).
* **Linear Probe Accuracy**:
  * **Random Baseline**: $25.0\%$
  * **Mel-spectrogram Baseline (mean-pooled)**: **$72.9\%$**
  * **wav2vec 2.0 Linear Probe**: **$85.4\%$**
* **Findings**: The high classification accuracy ($85.4\%$) of wav2vec2 compared to the raw mel-spectrogram baseline ($72.9\%$) confirms that self-supervised pretraining (masking and contrastive learning) successfully extracts rich, task-invariant acoustic-phonetic representations that are highly linearly separable.

---

### Part 4: OpenVoice V2 Voice Cloning
We implemented voice cloning by separating speaker identity (tone color) from linguistic/prosodic style (base speaker):
* **Tone Color Extraction**: Extracted a 256-dimensional speaker embedding vector of shape `[1, 256, 1]` from a $13.52$-second reference audio recording.
* **Cloned Voice Accent Metrics**:
  * **US**: Duration $1.67$s | RMS Energy $0.0700$ | Mel Spectral Centroid $1.8314$ | Cosine Similarity $0.7870$
  * **BR**: Duration $1.18$s | RMS Energy $0.0979$ | Mel Spectral Centroid $3.5854$
  * **INDIA**: Duration $1.40$s | RMS Energy $0.0428$ | Mel Spectral Centroid $0.6852$
  * **AU**: Duration $1.64$s | RMS Energy $0.0920$ | Mel Spectral Centroid $3.1663$
* **Findings**: OpenVoice successfully converted base TTS speech (in English, Spanish, and French) into the cloned voice's timbre while maintaining the speed, rhythm, and emotion of the base speaker. This confirms that tone color represents a purely language-independent acoustic signature.

---

## 2. Conclusion
The experiments highlight the evolution of speech models:
1. **Alignment-free Training**: CTC resolves variable-duration sequence alignment using dynamic programming.
2. **Self-Supervised Learning**: wav2vec 2.0 learns robust acoustic features without textual annotations.
3. **Identity Disentanglement**: OpenVoice V2 demonstrates that voice identity (timbre) can be decoupled from style and language, enabling zero-shot, cross-lingual voice cloning from a very short clip.
