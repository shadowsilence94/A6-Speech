# Voice Recording Guide for A6 Speech Assignment

To perform the voice cloning exercise in Part 5 of the assignment successfully, you must provide a short audio recording of your own voice. Below are the required specifications and instructions on how to format, record, and place the file.

---

## 1. File Path & Naming
* Place your recording in the workspace under the folder:
  `data/voice_clone/`
* In **Cell 36** of the notebook, the reference path is set to:
  `reference_path = 'data/voice_clone/my_voice.mp3'`
* You can use either **`.mp3`** or **`.wav`** format. Ensure that the filename and extension on disk match exactly with the `reference_path` variable defined in the cell (e.g., if you use a WAV file, update the variable to `'data/voice_clone/my_voice.wav'`).

---

## 2. Technical Specifications (Recommended)
For optimal compatibility with OpenVoice and PyTorch's audio processing backends (`torchaudio`/`librosa`):
* **File Format**: **WAV** (strongly recommended for lossless compatibility) or **MP3**.
* **Sampling Rate**: **16,000 Hz (16 kHz)** or **22,050 Hz (22.05 kHz)**.
* **Audio Channels**: **Mono** (1 channel). Stereo (2 channels) can cause channel mismatch warnings.
* **Bit Depth**: **16-bit PCM**.

---

## 3. Recording Guidelines
* **Duration**: **10 to 30 seconds** of continuous, clear speech.
  * *Too short* (< 5s) will fail to extract a representative tone color embedding.
  * *Too long* (> 30s) increases VRAM consumption and processing time without improving quality.
* **Content**: Speak naturally, using your normal pitch, pacing, and volume. We have provided a phonetically balanced script in [recording_script.txt](file:///Users/htutkoko/Library/CloudStorage/GoogleDrive-htutkoko1994@gmail.com/My%20Drive/Job%20in%20progress/Deep%20Learning/A6/recording_script.txt) for you to read.
* **Environment**: Record in a quiet room with minimal background noise, echo, or wind interference. Background hums or noise will be captured by the encoder as part of your "tone color," degrading the quality of the cloned voice.

---

## 4. How to Convert Your Recording

### Method A: Using `ffmpeg` (Command Line)
If you have recorded your voice on a phone or computer (e.g., as `.m4a`, `.aac`, or `.mp3`) and have `ffmpeg` installed on your machine or the puffer server, you can convert it to the standard mono WAV format using the following command:
```bash
ffmpeg -i input_recording.m4a -ac 1 -ar 16000 data/voice_clone/my_voice.wav
```
*(Replace `input_recording.m4a` with your raw recorded file).*

### Method B: Using Audacity (GUI)
1. Open your recording in Audacity.
2. Go to the bottom left and set the **Project Rate (Hz)** to `16000`.
3. Select **Tracks** $\rightarrow$ **Mix** $\rightarrow$ **Mix Stereo down to Mono** (if recorded in stereo).
4. Go to **File** $\rightarrow$ **Export** $\rightarrow$ **Export as WAV**.
5. Set encoding to **Signed 16-bit PCM** and save it as `my_voice.wav`.

### Method C: Online Converters
If you do not have command-line tools installed, you can use any free online audio converter to convert your file to:
* **Format**: WAV
* **Change resolution (sample rate)**: 16000 Hz
* **Audio channels**: Mono
