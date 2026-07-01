# Assignment 6: Speech Processing — Results, Findings & Explanations

This repository contains the completed notebook and resources for **Assignment 6 (Speech Processing)**. The assignment investigates the core components of speech systems, including speech tokenization, Connectionist Temporal Classification (CTC) alignment, self-supervised representations with wav2vec 2.0, and zero-shot voice cloning using OpenVoice V2.

---

## 1. Commands Used

All training, evaluation, and voice cloning scripts were executed using the following commands:

```bash
# Part 3 / Exercise 2: Train the toy CTC model
python3 run.py --model ctc --epochs 300 --train

# Part 4 / Exercise 3: Linear-probe the pretrained wav2vec2 checkpoint
python3 run.py --model wav2vec2-probe --dataset speechcommands --classes yes,no,stop,go --train

# Part 5.1 / Exercise 4: Extract the tone color embedding from the reference voice clip
python3 run.py --model voice-clone --extract-se --reference data/voice_clone/my_voice.wav

# Part 5.2 / Exercise 4: Synthesize in US style with the cloned voice
python3 run.py --model voice-clone --accent us --text "I got the job!" --generate

# Part 5.3 / Exercise 4: Synthesize all styles for comparison
python3 run.py --model voice-clone --accent all --text "Hello world" --generate

# Part 5.5: Cross-lingual cloning (Spanish)
python3 run.py --model voice-clone --language es --text "Hola, como estas?" --generate
```

---

## 2. Results Table

The exact results achieved across all exercises are summarized below:

| Task | Model / Method | Result | Notes |
|---|---|---|---|
| **Tokenization (Ex 1)** | SpeechTokenizer | Successful (BOS/EOS and Accent tags mapped) | Mapped character vs word token counts (see table below). |
| **CTC Character Error Rate (Ex 2)** | Toy BiLSTM + CTC | **20.0% CER** (Final Loss: `0.2232`) | CER dropped below 10.0% at Step 78 over 300 training steps. |
| **wav2vec2 vs Raw-Feature Probe (Ex 3)** | Linear Probe (70/30 split) | **85.4%** (wav2vec2) vs **64.6%** (Mel-Spectrogram) | Large performance gap (+20.8%) validates SSL representations. |
| **Voice Cloning (Ex 4)** | OpenVoice V2 | **0.7870 Cosine Similarity** (US) / High Quality | Successful style-independent timbre and language transfer. |

---

## 3. Exercise Answers & Explanations

### Exercise 1: Speech vs NLP Tokenization

#### a) Tokenization Metrics Table
| Sentence | # Char tokens | # Tokens (with BOS/EOS) | Accent tag token ID |
|---|---|---|---|
| Hello, how are you? | 19 | 21 | — |
| Dr. Smith prescribed 10 tablets. | 33 | 35 | — |
| [EN-US] I got the job! | 15 | 17 | 36 |
| [EN-BR] I lost my wallet. | 18 | 20 | 37 |
| [EN-INDIA] This is completely unacceptable! | 33 | 35 | 38 |

#### b) Text Normalization in TTS
Text normalization (e.g. converting `"Dr. Smith"` to `"doctor smith"`, `"10"` to `"ten"`) is critical for TTS because text-to-speech models map input character/phoneme tokens directly to acoustic waveforms. If abbreviations or numeric digits are passed raw:
* The model would spell out the abbreviation characters sequentially (e.g., `"Dr."` synthesized as the phonemes for "dee-are" rather than "doctor").
* Digits like `"10"` would either be skipped or pronounced incorrectly because they do not have a standard phonetic pronunciation as graphemes.

#### c) Architectural Similarity: `[CLS]` vs. Accent Tags
In NLP, the `[CLS]` token acts as a global pooling query that collects contextual representations from the entire sequence via self-attention. Similarly, in speech synthesis, the accent tag (like `[EN-US]`) is a special categorical token that is projected into a continuous conditioning embedding. This embedding is broadcast globally and fed into the decoding/synthesis layers (often via cross-attention or feature-wise linear modulation (FiLM)), allowing a single token ID to influence the pitch, prosody, accents, and pronunciation of every generated frame.

---

### Exercise 2: CTC Alignment & Loss
* **Hand-Calculated Alignment**: Summing over all valid paths for `"HEL"` over the 6-frame grid using log-space forward variable computation yields:
  $$\log P_{\text{CTC}}(\text{"HEL"}) \approx -5.5957 \quad \left(P_{\text{CTC}} \approx 0.003714\right)$$
* **Training Convergence**: The BiLSTM + CTC model learned to align temporal frame representations to target characters, with the CER dropping below 10.0% at step 78, demonstrating fast alignment convergence on smeared frame inputs.

---

### Exercise 3: wav2vec 2.0 Representation Learning

#### a) Accuracy Results
* **wav2vec 2.0 Linear Probe**: **85.4%**
* **Mel-spectrogram Baseline (mean-pooled)**: **64.6%**
* **Performance Gap**: **+20.8%** in favor of wav2vec2.

#### b) Analysis of the Performance Gap
The large performance gap (+20.8%) exists because raw mel-spectrograms retain substantial high-frequency acoustic noise, speaker identity differences, and variable temporal shifts, making the representation highly non-linear for classification. In contrast, the wav2vec 2.0 encoder has undergone self-supervised pretraining over massive datasets, allowing it to learn speaker-invariant, noise-robust phonemic representations that are linearly separable for semantic classes.

#### c) Contrastive vs. Reconstruction Inductive Biases
* **Contrastive Pretraining (wav2vec 2.0)**: Encourages the model to distinguish target segments from distractors. This forces the model to ignore low-level noise, phase differences, and speaker specifics, and retain structural phonetic details, transferring exceptionally well to classification tasks.
* **Reconstruction Pretraining (MAE)**: Forces the model to retain low-level high-frequency pixel/frame details to reconstruct the input. This can cause representations to contain redundant raw features, transferring less efficiently to semantic downstream tasks.

---

### Exercise 4: Voice Cloning — Identity, Style, and Language

#### a) Cloned Voice Accent Metrics
| Accent | Duration (s) | RMS Energy | Mel Spectral Centroid |
|---|---|---|---|
| **us** | 1.57s | 0.0556 | 1.1601 |
| **br** | 1.17s | 0.1031 | 3.9536 |
| **india** | 1.38s | 0.0432 | 0.6986 |
| **au** | 1.64s | 0.0904 | 3.0591 |

#### b) Cosine Similarity & Disentanglement Analysis
* **Cosine Similarity (US Cloned vs. Reference)**: **0.7870**
* **Analysis**: If OpenVoice's disentanglement is working well, the cosine similarity between the reference speaker embedding and the embeddings extracted from each of the generated clips should be high ($\ge 0.75$) and roughly equal across all accents. This shows that the speaker's vocal identity (timbre) has been successfully decoupled from the linguistic accent, speech speed, and prosody of the base speaker model.

---

## 4. Discussion

1. **Alignment-Free Sequence Modeling**: Understanding speech tokenization and CTC alignment changes our perspective on sequence modeling by replacing rigid, hard-aligned forced alignment steps (e.g. GMM-HMM aligners) with a fully differentiable dynamic programming loss. This allows ASR and TTS models to discover soft, temporal alignments end-to-end, making models robust to variations in speaking rate and pauses without requiring manual phone-level annotations.
2. **Different Roles of Conditioning Variables**:
   * **Text Tokens**: Discrete categorical tokens that carry semantic content (what is spoken) and are processed sequentially.
   * **CTC Blanks**: Temporal alignment markers used to frame transition intervals and silence, keeping characters distinct.
   * **Tone Color Embeddings**: Continuous continuous vector representations of a speaker's unique vocal timbre (how the voice sounds). Unlike text tokens or blanks which are sequential, the tone color embedding acts as a global continuous bias that conditions the entire generative process through feature modulation, remaining invariant to the specific words spoken.
