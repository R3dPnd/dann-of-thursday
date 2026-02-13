## Wake Word: "ok Dann"

### Goal
Reliable local wake-word detection that flips the agent into listening state when it hears **"ok Dann"**.

### Engine Choice (recommended)
- Use **Picovoice Porcupine** (cross-platform, Windows-friendly, easy custom wake word training via web console).

### Prereqs (Windows/macOS/Linux)
1) Install Python 3.9+ and ensure `python`/`pip` on PATH.  
2) In repo root:
   
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Setup Porcupine

1. **Get AccessKey** (free):
   - Sign up at https://console.picovoice.ai/
   - Create a free account and get your AccessKey

2. **Create Custom Wake Word**:
   - Log into Picovoice Console
   - Go to "Porcupine" → "Create Custom Wake Word"
   - Type "ok Dann" as your wake word
   - Select your platform:
     - **Windows x86_64** for most Windows systems (Intel/AMD processors)
     - **Windows arm64** only if you're on a Windows on ARM device (e.g., Surface Pro X)
     - Check your architecture: `python -c "import platform; print(platform.machine())"` → AMD64 = x86_64, ARM64 = arm64
   - Download the `.ppn` model file

3. **Place Model File**:
   - Save the downloaded `.ppn` file as `models/ok_dann.ppn`
   - Ensure the file is in the correct location

4. **Configure AccessKey**:
   - Edit `config.yaml` and set `wake_word.access_key` to your Picovoice AccessKey

### Models

- **Custom wake word**: Train "ok Dann" via Picovoice Console (web-based, no local training needed).
- **Model format**: `.ppn` files (platform-specific, download for your OS).
- **Training**: Done entirely in browser - no Linux dependencies or complex setup required.

### Runtime Config (suggested defaults)

- Sample rate: 16 kHz mono
- Frame size: 512 samples (32 ms) - fixed requirement for Porcupine
- Sensitivity: start at `0.5`, tune to reduce false positives
- Trigger debounce: require 2 consecutive detections before firing `wake_detected`
- Cooldown: ignore detections for 2 s after a trigger to avoid echo re-trigger

### Minimal Detector Loop (Python)

```python
import numpy as np
import pvporcupine
import sounddevice as sd

# Initialize Porcupine
porcupine = pvporcupine.create(
    access_key="YOUR_ACCESS_KEY",  # From Picovoice Console
    keyword_paths=["models/ok_dann.ppn"],
    sensitivities=[0.5],
)

samplerate = 16000
block = 512  # Fixed requirement for Porcupine

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    
    # Convert float32 to int16 PCM for Porcupine
    audio_float = np.frombuffer(indata, dtype=np.float32)
    audio_float = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = (audio_float * 32767).astype(np.int16)
    
    # Process audio frame
    keyword_index = porcupine.process(audio_int16)
    if keyword_index >= 0:  # 0 = first keyword detected
        print("wake_detected")
        # TODO: signal orchestrator to enter LISTENING state

with sd.InputStream(channels=1, samplerate=samplerate, blocksize=block,
                    dtype="float32", callback=callback):
    print("Listening for 'ok Dann'...")
    sd.sleep(10_000_000)

# Cleanup
porcupine.delete()
```

### Integration Notes
- Run wake detector in its own thread/task; post `wake_detected` to the orchestrator.
- Mute/suspend wake detection while TTS audio plays to avoid echo triggers.
- Expose a config block:
  - `wake_word.engine: porcupine`
  - `wake_word.access_key: YOUR_ACCESS_KEY`
  - `wake_word.model_path: models/ok_dann.ppn`
  - `wake_word.sensitivity: 0.5`
  - `wake_word.cooldown_ms: 2000`
  - `wake_word.debounce: 2`

### Tuning Checklist
- If false positives: lower sensitivity (e.g., 0.35) and add debounce.
- If misses wake: increase sensitivity (e.g., 0.65) and ensure mic gain is adequate.
- Test in quiet room, noisy room, and with playback audio.
- Ensure you're using the correct platform-specific `.ppn` file (Windows x86_64 vs macOS arm64, etc.)

### Quick Test
```bash
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python wakeword_test.py --model models/ok_dann.ppn --access-key YOUR_ACCESS_KEY
```
- Confirm "wake_detected" prints reliably; adjust sensitivity/debounce, then integrate.

### Advantages of Porcupine
- **Windows-friendly**: No Linux-only dependencies
- **Easy training**: Web-based console, no local setup
- **Cross-platform**: Works on Windows, macOS, Linux
- **Fast training**: Custom wake words trained in seconds
- **Free tier**: Free AccessKey for development
