# A6: Speech Processing Assignment

This repository contains the files and requirements for **Assignment 6 (Speech Processing)** of the Deep Learning course. You can push this folder to your GitHub and clone it directly on the puffer server to run the training and voice cloning models.

## Repository Contents
* **`A6-Speech-Processing.ipynb`**: The main Jupyter Notebook for the assignment.
* **`A6.md`**: Instruction guide detailing the parts, topics, and exercises of the lab.
* **`VOICE_RECORDING_GUIDE.md`**: Detailed instructions on how to record and convert your own voice clip for Part 5 (Voice Cloning).
* **`requirements.txt`**: List of Python dependencies required to run the speech models (wav2vec2, CTC, OpenVoice, MeloTTS, etc.).
* **`figures/`**: Folder containing images rendered inside the notebook.

---

## Getting Started

### Step 1: Push This Repo to Your GitHub
To clone this assignment on the puffer server, you should first push it to your personal GitHub account:
1. Go to [GitHub](https://github.com) and create a new repository (e.g. `A6-Speech`). Do *not* initialize it with a README or .gitignore.
2. In your local terminal, navigate to this `A6` folder and link it to your remote repository:
   ```bash
   git remote add origin https://github.com/<your-username>/A6-Speech.git
   ```
3. Push the main branch:
   ```bash
   git push -u origin main
   ```

### Step 2: Clone and Run on Puffer Server
1. SSH into the puffer server.
2. Clone your repository:
   ```bash
   git clone https://github.com/<your-username>/A6-Speech.git
   cd A6-Speech
   ```
3. Activate your virtual/conda environment (e.g. `ai_env`):
   ```bash
   conda activate ai_env
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Install OpenVoice dependencies (inside the notebook, these are downloaded via `!git clone` and pip):
   ```bash
   python3 -m unidic download
   ```
6. Start your Jupyter environment and open `A6-Speech-Processing.ipynb`.

---

## Voice Recording Requirements
For the **Voice Cloning** exercise in Part 5:
* Record **10 to 30 seconds** of your own voice.
* Convert it to a **16 kHz Mono WAV** file (recommended) or a high-quality **MP3** file.
* Save the file to `data/voice_clone/my_voice.wav` (or `.mp3`) and ensure the path matches `reference_path` in **Cell 36** of the notebook.
* Refer to [VOICE_RECORDING_GUIDE.md](file:///Users/htutkoko/Library/CloudStorage/GoogleDrive-htutkoko1994@gmail.com/My%20Drive/Job%20in%20progress/Deep%20Learning/A6/VOICE_RECORDING_GUIDE.md) for step-by-step recording and conversion instructions.
