import os
import sys
import argparse

# Pre-parse the --gpu argument from sys.argv to set environment variables
# BEFORE any PyTorch context initialization happens.
gpu_val = None
for idx, arg in enumerate(sys.argv):
    if arg == '--gpu' and idx + 1 < len(sys.argv):
        gpu_val = sys.argv[idx + 1]
        break
if gpu_val is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_val

# Now import PyTorch and other libraries
import json
import numpy as np
import torch
import torch.nn as nn

# Hardware Diagnostics & Device Selection
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

def print_hardware_diagnostics():
    print("=" * 60)
    print("HARDWARE DIAGNOSTICS")
    print("=" * 60)
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"  Detected {n_gpus} available GPU(s):")
        for i in range(n_gpus):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Apple Silicon MPS backend detected for hardware acceleration.")
    else:
        print("  CPU-only mode (training will be slow)")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  Device selected: {device}")
    print("=" * 60)

# ── 1. CTC Model Implementation & Training (Exercise 2) ─────────────────────
def run_ctc_training(epochs=300):
    import torch.nn.functional as F
    import random
    import distance  # Used to compute Character Error Rate (CER)

    ALPHABET = list('helo wrd')
    CHAR2IDX = {c: i+1 for i, c in enumerate(ALPHABET)}
    IDX2CHAR = {i+1: c for i, c in enumerate(ALPHABET)}
    VOCAB_SIZE = len(ALPHABET) + 1
    N_MELS = 20
    WORDS = ['hello', 'world', 'hero', 'red', 'led', 'doer']

    def synthesize_frames(word):
        frames = []
        for ch in word:
            n = random.randint(3, 8)
            base = np.zeros(N_MELS)
            base[CHAR2IDX[ch] % N_MELS] = 3.0
            for _ in range(n):
                frames.append(base + np.random.randn(N_MELS) * 0.5)
        return np.stack(frames)

    def ctc_greedy_decode(log_probs_seq):
        argmax_seq = np.argmax(log_probs_seq, axis=-1)
        # Collapse repeats
        collapsed = []
        prev = -1
        for val in argmax_seq:
            if val != prev:
                collapsed.append(val)
                prev = val
        # Strip blanks (0)
        decoded_chars = [IDX2CHAR[val] for val in collapsed if val != 0]
        return "".join(decoded_chars)

    class TinyCTCModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(N_MELS, 64, batch_first=True, bidirectional=True)
            self.fc   = nn.Linear(64 * 2, VOCAB_SIZE)

        def forward(self, x):
            h, _ = self.lstm(x)
            return F.log_softmax(self.fc(h), dim=-1)

    print(f"Training TinyCTCModel for {epochs} steps on device: {device}...")
    model = TinyCTCModel().to(device)
    
    # Use all available GPUs for training if multi-GPU is active
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel to train across {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    losses = []
    cers = []
    dropped_below_10 = -1

    for step in range(epochs):
        word = random.choice(WORDS)
        frames = synthesize_frames(word)
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
        targets = torch.tensor([CHAR2IDX[c] for c in word], dtype=torch.long).to(device)

        log_probs = model(x).transpose(0, 1)  # (T, B, V)
        input_lengths  = torch.tensor([log_probs.size(0)]).to(device)
        target_lengths = torch.tensor([len(targets)]).to(device)

        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Greedy decoding to track CER
        model.eval()
        with torch.no_grad():
            lp = model(x).squeeze(0).cpu().numpy()  # (T, V)
            decoded = ctc_greedy_decode(lp)
            dist = distance.levenshtein(decoded, word)
            cer = dist / len(word)
            cers.append(cer)
            
            if cer < 0.10 and dropped_below_10 == -1:
                dropped_below_10 = step + 1
        model.train()

        if (step + 1) % 50 == 0 or step == 0:
            mean_loss = np.mean(losses[-50:])
            mean_cer = np.mean(cers[-50:])
            print(f'Step {step+1:3d} | CTC loss: {mean_loss:.4f} | CER: {mean_cer*100:.1f}%')

    print(f"\nFinal training performance: Loss = {losses[-1]:.4f} | CER = {cers[-1]*100:.1f}%")
    if dropped_below_10 != -1:
        print(f"Character Error Rate (CER) dropped below 10% at training step: {dropped_below_10}")
    else:
        print("Character Error Rate (CER) did not drop below 10% during training.")

    # Save Results
    os.makedirs('results', exist_ok=True)
    results_file = 'results/a6_results.json'
    results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception:
            pass

    results['ctc_toy_asr_final_loss'] = float(losses[-1])
    results['ctc_toy_asr_mean_loss_last_50'] = float(np.mean(losses[-50:]))
    results['ctc_toy_asr_final_cer'] = float(cers[-1])
    results['ctc_toy_asr_cer_dropped_below_10_step'] = int(dropped_below_10)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"CTC training complete. Saved results to {results_file}.")


# ── 2. wav2vec 2.0 vs Raw Features Linear Probing (Exercise 3) ──────────────
def run_wav2vec_probe(dataset_name='speechcommands', classes='yes,no,stop,go'):
    import torchaudio
    import torchaudio.transforms as T
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    probe_words = [w.strip() for w in classes.split(',')]
    n_per_class = 40

    print("Loading pretrained wav2vec 2.0 model from Hugging Face...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    w2v_model.eval()

    # Use all available GPUs for feature extraction if multi-GPU is active
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        w2v_model = nn.DataParallel(w2v_model)

    print(f"Loading SpeechCommands dataset subset for classes: {probe_words}...")
    os.makedirs('data/speechcommands', exist_ok=True)
    
    sc_extracted_path = 'data/speechcommands/SpeechCommands/speech_commands_v0.02'
    download_flag = not os.path.exists(sc_extracted_path)
    sc_dataset = torchaudio.datasets.SPEECHCOMMANDS(root='data/speechcommands', download=download_flag)

    by_label = {w: [] for w in probe_words}
    for i in range(len(sc_dataset)):
        wvf, sr, label, *_ = sc_dataset[i]
        if label in by_label and len(by_label[label]) < n_per_class:
            by_label[label].append(wvf)
        if all(len(v) >= n_per_class for v in by_label.values()):
            break

    print("Extracting features (wav2vec2 representations AND raw Mel-Spectrogram)...")
    mel_transformer = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80)
    
    w2v_feats, mel_feats, labels_list = [], [], []
    with torch.no_grad():
        for label, clips in by_label.items():
            for wvf in clips:
                # 1. wav2vec2 features
                inputs = processor(wvf.squeeze(0).numpy(), sampling_rate=16000, return_tensors='pt').to(device)
                if isinstance(w2v_model, nn.DataParallel):
                    out = w2v_model.module(**inputs).last_hidden_state
                else:
                    out = w2v_model(**inputs).last_hidden_state
                pooled_w2v = out.mean(dim=1).squeeze(0).cpu()
                w2v_feats.append(pooled_w2v)
                
                # 2. Raw mel-spectrogram features
                mel_spec = mel_transformer(wvf)
                pooled_mel = mel_spec.mean(dim=-1).squeeze(0).cpu()
                mel_feats.append(pooled_mel)

                labels_list.append(probe_words.index(label))

    X_w2v = torch.stack(w2v_feats).numpy()
    X_mel = torch.stack(mel_feats).numpy()
    y = np.array(labels_list)

    # 1. Train Linear Probe on wav2vec2
    print("Training linear probe on wav2vec2 features...")
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_w2v, y, test_size=0.25, random_state=42, stratify=y)
    clf_w2v = LogisticRegression(max_iter=1000)
    clf_w2v.fit(X_train_w, y_train_w)
    acc_w2v = clf_w2v.score(X_test_w, y_test_w)

    # 2. Train Linear Probe on Mel-Spectrogram baseline
    print("Training linear probe on raw Mel-Spectrogram features...")
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_mel, y, test_size=0.25, random_state=42, stratify=y)
    clf_mel = LogisticRegression(max_iter=1000)
    clf_mel.fit(X_train_m, y_train_m)
    acc_mel = clf_mel.score(X_test_m, y_test_m)

    print("\n" + "="*50)
    print("LINEAR PROBE PROFILING REPORT")
    print("="*50)
    print(f"Raw Mel-Spectrogram baseline test accuracy: {acc_mel*100:.1f}%")
    print(f"Frozen wav2vec2 representation test accuracy: {acc_w2v*100:.1f}%")
    print(f"Performance Gap: +{(acc_w2v - acc_mel)*100:.1f}%")
    print("="*50)

    # Save Results
    os.makedirs('results', exist_ok=True)
    results_file = 'results/a6_results.json'
    results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception:
            pass

    results['wav2vec2_linear_probe_accuracy'] = float(acc_w2v)
    results['mel_spectrogram_linear_probe_accuracy'] = float(acc_mel)
    results['feature_extractor_performance_gap'] = float(acc_w2v - acc_mel)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"Linear probing complete. Saved results to {results_file}.")


# ── 3. OpenVoice V2 Speaker Identity & Similarity Eval (Exercise 4) ────────
def get_openvoice_converter():
    from huggingface_hub import snapshot_download
    from openvoice.api import ToneColorConverter

    print("Loading OpenVoice V2 checkpoints...")
    ckpt_dir = snapshot_download(repo_id='myshell-ai/OpenVoiceV2', local_dir='data/voice_clone/checkpoint', local_dir_use_symlinks=False)
    
    dev_str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    tone_color_converter = ToneColorConverter(f'{ckpt_dir}/converter/config.json', device=dev_str)
    tone_color_converter.load_ckpt(f'{ckpt_dir}/converter/checkpoint.pth')
    
    # Enable DataParallel if multiple GPUs are available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
         tone_color_converter.model = nn.DataParallel(tone_color_converter.model)
         
    return tone_color_converter, ckpt_dir

def extract_tone_color(reference_file='my_voice.wav'):
    from openvoice import se_extractor

    ref_path = reference_file if os.path.exists(reference_file) else os.path.join('data/voice_clone', reference_file)
    if not os.path.exists(ref_path):
        print(f"ERROR: Reference voice file '{ref_path}' not found! Check your path.")
        sys.exit(1)

    print(f"Extracting tone color embedding from: {ref_path}...")
    tone_color_converter, _ = get_openvoice_converter()
    
    os.makedirs('data/voice_clone/processed', exist_ok=True)
    target_se, audio_name = se_extractor.get_se(
        ref_path, tone_color_converter, target_dir='data/voice_clone/processed', vad=True
    )
    print(f"Extracted tone color embedding shape: {target_se.shape}")
    
    # Save Results
    os.makedirs('results', exist_ok=True)
    results_file = 'results/a6_results.json'
    results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception:
            pass

    results['openvoice_speaker_embedding_shape'] = list(target_se.shape)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print("Speaker embedding saved to 'data/voice_clone/processed/' and JSON updated.")

def compute_cosine_similarity(tgt_se_path, generated_wav_path, tone_color_converter):
    from openvoice import se_extractor
    
    try:
        tgt_se = torch.load(tgt_se_path, map_location=device)
        temp_eval_dir = 'data/voice_clone/processed_eval'
        os.makedirs(temp_eval_dir, exist_ok=True)
        gen_se, _ = se_extractor.get_se(
            generated_wav_path, tone_color_converter, target_dir=temp_eval_dir, vad=True
        )
        
        tgt_flat = tgt_se.view(-1)
        gen_flat = torch.tensor(gen_se, device=device).view(-1)
        
        similarity = torch.nn.functional.cosine_similarity(tgt_flat, gen_flat, dim=0).item()
        return similarity
    except Exception as e:
        print(f"Warning: Could not compute similarity for {generated_wav_path}: {e}")
        return None

def generate_voice_clone(accent='us', text="I got the job!", language=None):
    from melotts import MeloTTS

    os.makedirs('data/voice_clone', exist_ok=True)
    se_path = 'data/voice_clone/processed/se.pth'
    if not os.path.exists(se_path):
        print(f"ERROR: Speaker embedding '{se_path}' not found! Run with '--extract-se' first.")
        sys.exit(1)

    tone_color_converter, ckpt_dir = get_openvoice_converter()
    target_se = torch.load(se_path, map_location=device)

    style_to_se = {
        'us':    ('en-us.pth',    'EN-US'),
        'br':    ('en-br.pth',    'EN-BR'),
        'india': ('en-india.pth', 'EN_INDIA'),
        'au':    ('en-au.pth',    'EN-AU'),
    }

    os.makedirs('results', exist_ok=True)
    results_file = 'results/a6_results.json'
    results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception:
            pass

    if language is not None:
        lang_upper = language.upper()
        print(f"Generating cross-lingual base speech in language: {lang_upper}...")
        base_tts = MeloTTS(language=lang_upper, device=str(device))
        speaker_ids = base_tts.hps.data.spk2id
        spk_id = list(speaker_ids.values())[0]
        temp_wav = 'data/voice_clone/temp_base.wav'
        base_tts.tts_to_file(text, spk_id, temp_wav, speed=1.0)

        output_wav = f'data/voice_clone/cloned_cross_lingual_{language.lower()}.wav'
        
        lang_se_file = f"{ckpt_dir}/base_speakers/ses/{language.lower()}.pth"
        if not os.path.exists(lang_se_file):
            lang_se_file = f"{ckpt_dir}/base_speakers/ses/en-default.pth"
        
        source_se = torch.load(lang_se_file, map_location=device)
        
        # Unwrap DataParallel to call custom method convert if wrapped
        converter_model = tone_color_converter
        if hasattr(tone_color_converter, "module"):
            converter_model = tone_color_converter.module
            
        converter_model.convert(
            model=temp_wav,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_wav
        )
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        print(f"Generated cross-lingual cloned voice: {output_wav}")
        
        sim = compute_cosine_similarity(se_path, output_wav, tone_color_converter)
        if sim is not None:
            print(f"Cosine Similarity (Identity preservation): {sim:.4f}")
            results[f'openvoice_similarity_cross_lingual_{language.lower()}'] = sim
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
        return

    accents_to_process = []
    if accent == 'all':
        accents_to_process = list(style_to_se.keys())
    else:
        if accent not in style_to_se:
            print(f"ERROR: Accent '{accent}' not supported. Choose from: {list(style_to_se.keys())} or 'all'")
            sys.exit(1)
        accents_to_process = [accent]

    for acc in accents_to_process:
        pth_name, spk_key = style_to_se[acc]
        print(f"Generating base speech in style: {spk_key}...")
        
        base_tts = MeloTTS(language='EN', device=str(device))
        speaker_ids = base_tts.hps.data.spk2id
        spk_id = speaker_ids[spk_key]
        temp_wav = 'data/voice_clone/temp_base.wav'
        base_tts.tts_to_file(text, spk_id, temp_wav, speed=1.0)

        output_wav = f'data/voice_clone/cloned_{acc}.wav'
        source_se = torch.load(f'{ckpt_dir}/base_speakers/ses/en-default.pth', map_location=device)
        
        converter_model = tone_color_converter
        if hasattr(tone_color_converter, "module"):
            converter_model = tone_color_converter.module
            
        converter_model.convert(
            model=temp_wav,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_wav
        )
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        print(f"Generated expressive cloned voice: {output_wav}")

        sim = compute_cosine_similarity(se_path, output_wav, tone_color_converter)
        if sim is not None:
            print(f"Cosine Similarity (Identity preservation): {sim:.4f}")
            results[f'openvoice_similarity_{acc}'] = sim
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)


# ── 4. Main Argument Parser ────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speech Processing Assignment 6 CLI tool")
    parser.add_argument('--model', type=str, required=True, choices=['ctc', 'wav2vec2-probe', 'voice-clone'],
                        help="Select the model to run: ctc, wav2vec2-probe, or voice-clone")
    parser.add_argument('--epochs', type=int, default=300,
                        help="Number of epochs/steps for training (default: 300)")
    parser.add_argument('--train', action='store_true',
                        help="Flag to train the ctc or wav2vec2-probe model")
    parser.add_argument('--dataset', type=str, default='speechcommands',
                        help="Dataset name for wav2vec2 linear-probe (default: speechcommands)")
    parser.add_argument('--classes', type=str, default='yes,no,stop,go',
                        help="Comma-separated class names for linear-probe (default: yes,no,stop,go)")
    parser.add_argument('--extract-se', action='store_true',
                        help="Extract tone color speaker embedding from the reference audio clip")
    parser.add_argument('--reference', type=str, default='my_voice.wav',
                        help="Path to reference voice recording (default: my_voice.wav)")
    parser.add_argument('--accent', type=str, default='us',
                        help="Accent/style to synthesize: us, br, india, au, or all (default: us)")
    parser.add_argument('--text', type=str, default="I got the job!",
                        help="Text string to synthesize for voice cloning")
    parser.add_argument('--language', type=str, default=None,
                        help="Language for cross-lingual voice cloning (e.g. es, fr, zh, jp, kr)")
    parser.add_argument('--generate', action='store_true',
                        help="Synthesize cloned voice for selected accent/language")
    parser.add_argument('--gpu', type=str, default=None,
                        help="GPU device ID to select (e.g., --gpu 1) for multi-GPU hardware configurations")

    args = parser.parse_args()

    # Run diagnostics printout
    print_hardware_diagnostics()

    if args.model == 'ctc':
        if args.train:
            run_ctc_training(epochs=args.epochs)
        else:
            print("ERROR: Please pass '--train' to train the ctc model.")

    elif args.model == 'wav2vec2-probe':
        if args.train:
            run_wav2vec_probe(dataset_name=args.dataset, classes=args.classes)
        else:
            print("ERROR: Please pass '--train' to run the wav2vec2 linear-probe.")

    elif args.model == 'voice-clone':
        if args.extract_se:
            extract_tone_color(reference_file=args.reference)
        elif args.generate:
            generate_voice_clone(accent=args.accent, text=args.text, language=args.language)
        else:
            print("ERROR: Please pass '--extract-se' or '--generate' for voice cloning operations.")
