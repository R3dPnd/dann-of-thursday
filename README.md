# Dann of Thursday

![img](https://images6.fanpop.com/image/photos/43800000/Dann-of-Thursday-gun-x-sword-43866941-720-480.jpg)

Voice AI agent: say **"ok Dann"** to ask questions. Uses wake word → STT (faster-whisper) → Ollama → Piper TTS.

## Setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

1. **Wake word model**: 
   - Get a free AccessKey from https://console.picovoice.ai/
   - Create custom wake word "ok Dann" in Picovoice Console
   - Download the `.ppn` model file to `models/ok_dann.ppn`
   - Set `wake_word.access_key` in `config.yaml` (see [wake-word.md](wake-word.md))
2. **Ollama**: Install and run `ollama serve`, pull a model (e.g. `ollama pull llama3.2`).
3. **Piper TTS**: Included via `pip install piper-tts` (works on M1). Download a voice from [Piper voices](https://github.com/rhasspy/piper/releases) (e.g. `en_US-lessac-medium.onnx`), place in `models/piper/`, set `tts.voice_model` in config.

Edit `config.yaml` for your paths and preferences.

## Run

```bash
source .venv/bin/activate
python -m src.main
```

Say "ok Dann" then ask your question.
