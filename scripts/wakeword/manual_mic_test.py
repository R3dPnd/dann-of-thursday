"""Interactive wake word test — run this directly in your terminal."""

import numpy as np
import pvporcupine
import sounddevice as sd
import yaml
import time
import sys

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

ak = cfg["wake_word"]["access_key"]
device_id = cfg["audio"].get("input_device")
model_path = cfg["wake_word"].get("model_path", "models/ok_dann.ppn")
sensitivity = cfg["wake_word"].get("sensitivity", 0.8)

porcupine = pvporcupine.create(
    access_key=ak,
    keyword_paths=[model_path],
    sensitivities=[sensitivity],
)

dev_info = sd.query_devices(device_id) if device_id is not None else sd.query_devices(kind="input")
print(f"Using mic:    {dev_info['name']} (device {device_id})")
print(f"Wake model:   {model_path}")
print(f"Sensitivity:  {sensitivity}")
print(f"Say 'ok Dann' clearly. Ctrl+C to stop.\n")

frame_count = 0
detected = 0

def callback(indata, frames, time_info, status):
    global frame_count, detected
    frame_count += 1
    if status:
        print(f"  audio warning: {status}")

    pcm = indata[:, 0]
    rms = np.sqrt(np.mean(pcm ** 2))
    pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    result = porcupine.process(pcm_int16)

    bar = "#" * min(int(rms * 200), 50)
    if frame_count % 30 == 0:
        sys.stdout.write(f"\r  level: [{bar:<50}] rms={rms:.4f}")
        sys.stdout.flush()

    if result >= 0:
        detected += 1
        print(f"\n  *** WAKE WORD DETECTED! (#{detected}) ***")

try:
    with sd.InputStream(
        channels=1,
        samplerate=porcupine.sample_rate,
        blocksize=porcupine.frame_length,
        dtype="float32",
        device=device_id,
        callback=callback,
    ):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print(f"\n\nStopped. {detected} detections over {frame_count} frames.")
finally:
    porcupine.delete()
