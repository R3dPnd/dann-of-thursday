"""
Minimal wake word detector for "ok Dann" using openWakeWord.

Prereqs (macOS):
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

Run:
    source .venv/bin/activate
    python wakeword_test.py --model models/ok_dann.tflite

Press Ctrl+C to stop.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from openwakeword.model import Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listen for the wake word 'ok Dann'.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/ok_dann.tflite"),
        help="Path to wake word model (.tflite).",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.5,
        help="Detection sensitivity (0.0–1.0). Higher is more sensitive.",
    )
    parser.add_argument(
        "--debounce",
        type=int,
        default=2,
        help="Consecutive detections required before triggering.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Seconds to ignore detections after a trigger.",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16_000,
        help="Microphone sample rate (Hz).",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=512,
        help="Block size (samples) for the audio callback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    model = Model(
        wakeword_paths=[str(args.model)],
        sensitivity=[args.sensitivity],
    )

    last_trigger = 0.0
    consecutive = 0

    def callback(indata, frames, time_info, status):
        nonlocal last_trigger, consecutive
        if status:
            print(f"[audio-status] {status}", flush=True)

        audio = np.frombuffer(indata, dtype=np.float32)
        scores = model.predict(audio)

        hit = any(score >= args.sensitivity for score in scores)
        consecutive = consecutive + 1 if hit else 0

        now = time.monotonic()
        if consecutive >= args.debounce and (now - last_trigger) >= args.cooldown:
            print("wake_detected", flush=True)
            last_trigger = now
            consecutive = 0

    print("Listening for 'ok Dann'... (Ctrl+C to stop)")
    with sd.InputStream(
        channels=1,
        samplerate=args.samplerate,
        blocksize=args.block,
        dtype="float32",
        callback=callback,
    ):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
