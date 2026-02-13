"""
Minimal wake word detector for "ok Dann" using Picovoice Porcupine.

Prereqs:
    python3 -m venv .venv
    source .venv/bin/activate  # or: .venv\Scripts\activate on Windows
    pip install --upgrade pip
    pip install -r requirements.txt

Setup:
    1. Get a free AccessKey from https://console.picovoice.ai/
    2. Create a custom wake word "ok Dann" in Picovoice Console
    3. Download the .ppn model file to models/ok_dann.ppn

Run:
    source .venv/bin/activate  # or: .venv\Scripts\activate on Windows
    python wakeword_test.py --model models/ok_dann.ppn --access-key YOUR_ACCESS_KEY

Press Ctrl+C to stop.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pvporcupine
import sounddevice as sd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listen for the wake word 'ok Dann'.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/ok_dann.ppn"),
        help="Path to wake word model (.ppn).",
    )
    parser.add_argument(
        "--access-key",
        type=str,
        required=True,
        help="Picovoice AccessKey from https://console.picovoice.ai/",
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
        help="Block size (samples) for the audio callback. Must be 512 for Porcupine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    if args.block != 512:
        raise ValueError("Porcupine requires block_size=512 for 16kHz audio")

    porcupine = pvporcupine.create(
        access_key=args.access_key,
        keyword_paths=[str(args.model)],
        sensitivities=[args.sensitivity],
    )

    try:
        last_trigger = 0.0
        consecutive = 0

        def callback(indata, frames, time_info, status):
            nonlocal last_trigger, consecutive
            if status:
                print(f"[audio-status] {status}", flush=True)

            # Convert float32 audio to int16 PCM for Porcupine
            audio_float = np.frombuffer(indata, dtype=np.float32)
            audio_float = np.clip(audio_float, -1.0, 1.0)
            audio_int16 = (audio_float * 32767).astype(np.int16)

            # Porcupine.process() returns keyword index (0 for first keyword, -1 if no match)
            keyword_index = porcupine.process(audio_int16)
            hit = keyword_index >= 0

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
    finally:
        porcupine.delete()


if __name__ == "__main__":
    main()
