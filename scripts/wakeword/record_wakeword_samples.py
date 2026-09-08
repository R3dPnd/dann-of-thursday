#!/usr/bin/env python3
"""Interactively record real "ok Dann" clips from your own voice.

The existing training data (training/positive/) is 3000 clips synthesized
from a single Piper TTS voice — enough acoustic diversity to validate the
training pipeline, but not enough for the model to recognize a real human
speaker. This script fixes the actual gap: it records YOU saying "ok Dann"
repeatedly, so scripts/train_wakeword.py can mix real recordings in.

Two modes:

  Interactive (run this yourself, in your own terminal):
    .venv/bin/python scripts/record_wakeword_samples.py
    .venv/bin/python scripts/record_wakeword_samples.py --count 60
  Press Enter, then say "ok Dann" within ~2 seconds. Too quiet/mistimed
  takes are rejected on the spot and redone.

  Auto (fixed cadence, no keypress — for when someone else, e.g. an
  assistant, is running the command and you're talking along live):
    .venv/bin/python scripts/record_wakeword_samples.py --auto
  Counts down out loud in the terminal, then records, on a steady beat.
  Say "ok Dann" once per beat. Still auto-rejects silent/too-quiet takes
  (up to a few retries per slot, then moves on so it can't hang forever).

Output goes to training/positive_real/, separate from the synthetic clips
so train_wakeword.py can weight them differently.

After recording, retrain:
    .venv/bin/python scripts/train_wakeword.py
"""
import argparse
import random
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "training" / "positive_real"

SAMPLE_RATE = 16000
CLIP_SECONDS = 3
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS  # matches train_wakeword.py's clip shape
RECORD_SECONDS = 2.5  # window you actually have to say the phrase in

SILENCE_PEAK_THRESHOLD = 0.02  # below this, treat the clip as a mistimed/silent take


def record_clip() -> np.ndarray:
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    return audio[:, 0]


def embed_in_window(audio: np.ndarray) -> np.ndarray:
    """Place the recorded utterance at a random offset in a 3-second window,
    matching the clip shape train_wakeword.py expects."""
    clip = np.zeros(CLIP_SAMPLES, dtype=np.int16)
    n = min(len(audio), CLIP_SAMPLES)
    offset = random.randint(0, CLIP_SAMPLES - n)
    clip[offset:offset + n] = audio[:n]
    return clip


def save_wav(path: Path, clip: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(clip.tobytes())


MAX_AUTO_RETRIES = 3  # per slot, before giving up on it and moving on


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40, help="Number of clips to record (default: 40)")
    parser.add_argument(
        "--auto", action="store_true",
        help="Fixed-cadence countdown instead of press-Enter — no keyboard interaction needed",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(OUT_DIR.glob("ok_dann_real_*.wav"))
    start_idx = len(existing)

    print(f"Recording {args.count} real 'ok Dann' clips into {OUT_DIR}")
    print(f"({start_idx} already there from a previous run — continuing from #{start_idx})")
    print("Vary it naturally: normal pace, a bit fast, a bit slow, from a few feet away, etc.\n")

    if args.auto:
        print("Auto mode: say \"ok Dann\" once per countdown, starting in 3 seconds...\n", flush=True)
        time.sleep(3)

    saved = 0
    slot_retries = 0
    max_attempts = args.count * (MAX_AUTO_RETRIES + 1)  # hard ceiling so a stuck mic can't hang forever
    attempts = 0

    while saved < args.count and attempts < max_attempts:
        attempts += 1
        idx = start_idx + saved

        if args.auto:
            print(f"[{saved + 1}/{args.count}] 3...", flush=True)
            time.sleep(0.6)
            print("  2...", flush=True)
            time.sleep(0.6)
            print("  1...", flush=True)
            time.sleep(0.6)
            print("  say it now!", end="", flush=True)
        else:
            input(f"[{saved + 1}/{args.count}] Press Enter, then say \"ok Dann\"...")
            print("  recording...", end="", flush=True)

        audio = record_clip()
        peak = float(np.max(np.abs(audio)) / 32768.0)
        rms = float(np.sqrt(np.mean((audio.astype(np.float32) / 32768.0) ** 2)))
        print(f" peak={peak:.3f} rms={rms:.4f}")

        if peak < SILENCE_PEAK_THRESHOLD:
            if args.auto:
                slot_retries += 1
                if slot_retries >= MAX_AUTO_RETRIES:
                    print(f"  too quiet {MAX_AUTO_RETRIES}x in a row for this slot — giving up on it.\n")
                    slot_retries = 0
                else:
                    print("  too quiet / mistimed — trying that slot again.\n")
                time.sleep(0.5)
            else:
                print("  too quiet / mistimed — let's redo that one.\n")
            continue

        slot_retries = 0
        clip = embed_in_window(audio)
        out_path = OUT_DIR / f"ok_dann_real_{idx:04d}.wav"
        save_wav(out_path, clip)
        saved += 1
        print(f"  saved {out_path.name}\n")
        if args.auto:
            time.sleep(0.7)

    total = len(list(OUT_DIR.glob('*.wav')))
    if saved < args.count:
        print(f"Stopped early: only {saved}/{args.count} clips came through clean (mic issue?).")
    print(f"Done. {total} real clips total in {OUT_DIR}")
    print("Next: .venv/bin/python scripts/train_wakeword.py")


if __name__ == "__main__":
    main()
