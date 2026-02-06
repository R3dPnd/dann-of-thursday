## Wake Word: "ok Dann"

### Goal
Reliable local wake-word detection that flips the agent into listening state when it hears **“ok Dann”**.

### Engine Choice (recommended)
- Use **openWakeWord** (MIT, local, customizable). Porcupine is solid but licensed; openWakeWord keeps everything open.

### Prereqs (macOS)
1) Install Python 3.10+ (e.g. `brew install python@3.11`) and ensure `python3`/`pip3` on PATH.  
2) In repo root:
   
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Models

- Default English keyword model: `ok_nabu` or `hey_mycroft` ships with openWakeWord.  
- For a closer fit to **“ok Dann”**, train or fine-tune a custom model:
  - Record 50–150 positive samples of “ok Dann” (different mic distances, noise).
  - Record 1–2 minutes of negative/background audio.
  - Use the openWakeWord training guide: https://github.com/dscripka/openWakeWord#training
  - Export the resulting `.tflite` model and place it in `models/ok_dann.tflite`.

### Runtime Config (suggested defaults)

- Sample rate: 16 kHz mono
- Frame size/hop: 512 samples (32 ms) with 50% overlap
- Sensitivity: start at `0.5`, tune to reduce false positives
- Trigger debounce: require 2 consecutive detections before firing `wake_detected`
- Cooldown: ignore detections for 2 s after a trigger to avoid echo re-trigger

### Minimal Detector Loop (Python)

```
import sounddevice as sd
import numpy as np
from openwakeword.model import Model

model = Model(wakeword_paths=["models/ok_dann.tflite"], sensitivity=[0.5])
samplerate = 16000
block = 512

def callback(indata, frames, time, status):
    if status:  # log over/underruns
        print(status, flush=True)
    audio = np.frombuffer(indata, dtype=np.float32)
    scores = model.predict(audio)
    if any(s > 0.5 for s in scores):
        print("wake_detected")
        # TODO: signal orchestrator to enter LISTENING state

with sd.InputStream(channels=1, samplerate=samplerate, blocksize=block,
                    dtype="float32", callback=callback):
    print("Listening for 'ok Dann'...")
    sd.sleep(10_000_000)
```

### Integration Notes
- Run wake detector in its own thread/task; post `wake_detected` to the orchestrator.
- Mute/suspend wake detection while TTS audio plays to avoid echo triggers.
- Expose a config block:
  - `wake_word.engine: openwakeword`
  - `wake_word.model_path: models/ok_dann.tflite`
  - `wake_word.sensitivity: 0.5`
  - `wake_word.cooldown_ms: 2000`
  - `wake_word.debounce: 2`

### Tuning Checklist
- If false positives: lower sensitivity (e.g., 0.35) and add debounce.
- If misses wake: increase sensitivity (e.g., 0.65) and ensure mic gain is adequate.
- Test in quiet room, noisy room, and with playback audio.

### Quick Test
```bash
source .venv/bin/activate
python wakeword_test.py  # script wrapping the loop above
```
- Confirm “wake_detected” prints reliably; adjust sensitivity/debounce, then integrate.
