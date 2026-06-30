import os
import sys
import argparse
import json
import numpy as np
import torch

# General Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

# ── 1. CTC Model Implementation & Training ──────────────────────────────────
def run_ctc_training(epochs=300):
    import torch.nn as nn
    import torch.nn.functional as F
    import random

    ALPHABET = list('helo wrd')
    CHAR2IDX = {c: i+1 for i, c in enumerate(ALPHABET)}
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    losses = []
    for step in range(epochs):
        word = random.choice(WORDS)
        frames = synthesize_frames(word)
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
        targets = torch.tensor([CHAR2IDX[c] for c in word], dtype=torch.long).to(device)

        log_probs = model(x).transpose(0, 1)
        input_lengths  = torch.tensor([log_probs.size(0)]).to(device)
        target_lengths = torch.tensor([len(targets)]).to(device)

        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 50 == 0 or step == 0:
            print(f'Step {step+1:3d} | CTC loss: {np.mean(losses[-50:]):.4f}')

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
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"CTC training complete. Saved results to {results_file}.")


# ── 2. wav2vec 2.0 Feature Extraction & Probing ─────────────────────────────
def run_wav2vec_probe(dataset_name='speechcommands', classes='yes,no,stop,go'):
    import torchaudio
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    probe_words = [w.strip() for w in classes.split(',')]
    n_per_class = 40

    print("Loading pretrained wav2vec 2.0 model from Hugging Face...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    w2v_model.eval()

    print(f"Loading SpeechCommands dataset subset for classes: {probe_words}...")
    os.makedirs('data/speechcommands', exist_ok=True)
    
    # Check if local cache folder exists
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

    # Extract Features
    print("Extracting frozen wav2vec2 representations...")
    feats, labels_list = [], []
    with torch.no_grad():
        for label, clips in by_label.items():
            for wvf in clips:
                # Resample if not 16kHz
                inputs = processor(wvf.squeeze(0).numpy(), sampling_rate=16000, return_tensors='pt').to(device)
                out = w2v_model(**inputs).last_hidden_state
                pooled = out.mean(dim=1).squeeze(0).cpu()
                feats.append(pooled)
                labels_list.append(probe_words.index(label))

    X = torch.stack(feats).numpy()
    y = np.array(labels_list)

    # Train Linear Probe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f'Linear probe test accuracy: {acc*100:.1f}% (random baseline: {100/len(probe_words):.1f}%)')

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

    results['wav2vec2_linear_probe_accuracy'] = float(acc)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"Linear probing complete. Saved results to {results_file}.")


# ── 3. OpenVoice V2 Speaker Identity & Voice Cloning ───────────────────────
def get_openvoice_converter():
    from huggingface_hub import snapshot_download
    from openvoice.api import ToneColorConverter

    print("Loading OpenVoice V2 checkpoints...")
    ckpt_dir = snapshot_download(repo_id='myshell-ai/OpenVoiceV2', local_dir='data/voice_clone/checkpoint', local_dir_use_symlinks=False)
    
    # We pass the device parameter as string ('cuda', 'mps', or 'cpu')
    dev_str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    tone_color_converter = ToneColorConverter(f'{ckpt_dir}/converter/config.json', device=dev_str)
    tone_color_converter.load_ckpt(f'{ckpt_dir}/converter/checkpoint.pth')
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

def generate_voice_clone(accent='us', text="I got the job!", language=None):
    from melotts import MeloTTS

    os.makedirs('data/voice_clone', exist_ok=True)
    se_path = 'data/voice_clone/processed/se.pth'
    if not os.path.exists(se_path):
        print(f"ERROR: Speaker embedding '{se_path}' not found! Run with '--extract-se' first.")
        sys.exit(1)

    tone_color_converter, ckpt_dir = get_openvoice_converter()
    target_se = torch.load(se_path, map_location=device)

    # Dictionary mapping accents/styles to check files
    style_to_se = {
        'us':    ('en-us.pth',    'EN-US'),
        'br':    ('en-br.pth',    'EN-BR'),
        'india': ('en-india.pth', 'EN_INDIA'),
        'au':    ('en-au.pth',    'EN-AU'),
    }

    accents_to_process = []
    if language is not None:
        # Cross-lingual mode
        lang_upper = language.upper()
        # MeloTTS supports EN, ES, FR, ZH, JP, KR
        print(f"Generating cross-lingual base speech in language: {lang_upper}...")
        base_tts = MeloTTS(language=lang_upper, device=str(device))
        speaker_ids = base_tts.hps.data.spk2id
        
        # Select first speaker ID
        spk_id = list(speaker_ids.values())[0]
        temp_wav = 'data/voice_clone/temp_base.wav'
        base_tts.tts_to_file(text, spk_id, temp_wav, speed=1.0)

        # Apply tone color conversion to match reference
        output_wav = f'data/voice_clone/cloned_cross_lingual_{language.lower()}.wav'
        
        # Load language-specific speaker embedding for cross-lingual mapping
        lang_se_file = f"{ckpt_dir}/base_speakers/ses/{language.lower()}.pth"
        if not os.path.exists(lang_se_file):
            # Fallback to default EN speaker
            lang_se_file = f"{ckpt_dir}/base_speakers/ses/en-default.pth"
        
        source_se = torch.load(lang_se_file, map_location=device)
        tone_color_converter.convert(
            model=temp_wav,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_wav
        )
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        print(f"Generated cross-lingual cloned voice: {output_wav}")
        return

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
        
        # 1. Initialize MeloTTS for English
        base_tts = MeloTTS(language='EN', device=str(device))
        speaker_ids = base_tts.hps.data.spk2id
        
        # Select matching speaker ID
        spk_id = speaker_ids[spk_key]
        temp_wav = 'data/voice_clone/temp_base.wav'
        base_tts.tts_to_file(text, spk_id, temp_wav, speed=1.0)

        # 2. Convert tone color to match reference
        output_wav = f'data/voice_clone/cloned_{acc}.wav'
        source_se = torch.load(f'{ckpt_dir}/base_speakers/ses/en-default.pth', map_location=device)
        
        tone_color_converter.convert(
            model=temp_wav,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_wav
        )
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        print(f"Generated expressive cloned voice: {output_wav}")


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

    args = parser.parse_args()

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
