"""
Train a custom "ok Dann" openWakeWord model from pre-generated samples.

Prerequisites:
    Run scripts/generate_wakeword_samples.py first.

What this script does:
    1. Downloads openWakeWord feature models on first run (~5 MB total)
    2. Generates additional negative speech + noise clips with piper-tts
    3. Extracts audio embeddings for all clips (positives + negatives)
    4. Trains a small DNN classifier (~10–20 min on CPU)
    5. Exports model to models/ok_dann.onnx

Run:
    .venv/Scripts/python.exe scripts/train_wakeword.py

Switch to the trained model in config.yaml:
    wake_word:
      model: models/ok_dann.onnx
"""

import collections
import copy
import random
import sys
import wave
from pathlib import Path

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "openwakeword"))

from piper import PiperVoice, SynthesisConfig
from openwakeword.utils import AudioFeatures
import openwakeword.utils as oww_utils

# ── Minimal OWW trainer (avoids openwakeword.train → openwakeword.data → pronouncing
#    → pkg_resources which is missing in setuptools 82+) ──────────────────────

class _FCNBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU())
    def forward(self, x):
        return self.net(x)

class _Net(nn.Module):
    def __init__(self, input_shape: tuple, layer_dim: int = 128, n_blocks: int = 1):
        super().__init__()
        flat = input_shape[0] * input_shape[1]
        self.flatten   = nn.Flatten()
        self.entry     = nn.Sequential(nn.Linear(flat, layer_dim), nn.LayerNorm(layer_dim), nn.ReLU())
        self.blocks    = nn.ModuleList([_FCNBlock(layer_dim) for _ in range(n_blocks)])
        self.out       = nn.Sequential(nn.Linear(layer_dim, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.entry(self.flatten(x))
        for b in self.blocks:
            x = b(x)
        return self.out(x)

class OWWTrainer:
    """Minimal trainer that replicates openwakeword.train.Model for single-class wake words."""

    def __init__(self, input_shape: tuple = (16, 96), layer_dim: int = 128):
        self.input_shape = input_shape
        self.model       = _Net(input_shape, layer_dim)
        self.optimizer   = optim.Adam(self.model.parameters(), lr=0.0001)
        self.history     = collections.defaultdict(list)

    @staticmethod
    def _cosine_lr(step: int, warmup: int, hold: int, total: int, target_lr: float) -> float:
        if step < warmup:
            return target_lr * step / max(warmup, 1)
        if step < warmup + hold:
            return target_lr
        progress = (step - warmup - hold) / max(total - warmup - hold, 1)
        return 0.5 * target_lr * (1 + np.cos(np.pi * progress))

    def train_model(self, X, max_steps: int, warmup_steps: int, hold_steps: int,
                    X_val=None, val_steps=None, lr: float = 0.0001,
                    negative_weight_schedule=None) -> None:
        device = torch.device("cpu")
        self.model.to(device)
        val_steps = set(val_steps or [])
        neg_sched = negative_weight_schedule or [1]

        for step, (x, y) in tqdm(enumerate(X), total=max_steps, desc="Training"):
            x, y = x.to(device), y.to(device)

            # LR schedule
            new_lr = self._cosine_lr(step, warmup_steps, hold_steps, max_steps, lr)
            for g in self.optimizer.param_groups:
                g["lr"] = new_lr

            # Weighted BCE — upweight negatives to reduce false positives
            w_val = neg_sched[min(step, len(neg_sched) - 1)]
            weight = torch.where(y == 0, torch.tensor(float(w_val)), torch.ones_like(y))

            self.optimizer.zero_grad()
            pred = self.model(x).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy(pred, y, weight=weight)
            loss.backward()
            self.optimizer.step()

            self.history["loss"].append(loss.item())

            if step in val_steps and X_val is not None:
                self.model.eval()
                val_losses, correct, total = [], 0, 0
                with torch.no_grad():
                    for xv, yv in X_val:
                        pv = self.model(xv.to(device)).squeeze(1)
                        val_losses.append(
                            torch.nn.functional.binary_cross_entropy(pv, yv.to(device)).item()
                        )
                        correct += ((pv > 0.5) == yv.to(device).bool()).sum().item()
                        total += len(yv)
                acc = correct / max(total, 1)
                tqdm.write(f"  step {step:>5}  val_loss={np.mean(val_losses):.4f}  acc={acc:.3f}")
                self.history["val_loss"].append(np.mean(val_losses))
                self.history["val_acc"].append(acc)
                self.model.train()

    def export_to_onnx(self, output_path: str) -> None:
        self.model.eval().cpu()
        dummy = torch.rand((1,) + self.input_shape)
        torch.onnx.export(
            self.model, dummy, output_path,
            input_names=["input"],
            output_names=["ok_dann"],
            opset_version=13,
        )
        print(f"ONNX model exported → {output_path}", flush=True)

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).resolve().parents[1]
VOICE_MODEL  = REPO_ROOT / "models" / "pieper" / "en_GB-northern_english_male-medium.onnx"
POSITIVE_DIR     = REPO_ROOT / "training" / "positive"
ADVERSARIAL_DIR  = REPO_ROOT / "training" / "adversarial"
OUT_MODEL    = REPO_ROOT / "models" / "ok_dann.onnx"
OWW_MODELS_DIR = REPO_ROOT / "models" / "openwakeword" / "openwakeword" / "resources" / "models"

PIPER_RATE    = 22050
TARGET_RATE   = 16000
CLIP_SAMPLES  = TARGET_RATE * 3   # 3-second clips = 48 000 samples

WINDOW_FRAMES = 16    # DNN input: 16 frames × 96 embedding dims
BATCH_SIZE    = 128
TRAIN_STEPS   = 10000

N_NEG_SPEECH  = 2000
N_NEG_NOISE   = 500

# Varied negative phrases — none contain "ok dann" or close variants
NEGATIVE_PHRASES = [
    "what time is it", "open the door", "turn on the lights", "play some music",
    "set a timer for five minutes", "what is the weather like today",
    "remind me to call the office", "add milk to the shopping list",
    "how far is the nearest station", "call my wife",
    "navigate home", "send a message to john", "cancel that",
    "volume up", "volume down", "pause the music", "skip this song",
    "take a photo", "start recording", "stop recording",
    "good morning", "good night", "see you later", "talk to you soon",
    "one two three four five six seven eight nine ten",
    "the quick brown fox jumps over the lazy dog",
    "to be or not to be that is the question",
    "i would like a cup of tea please", "thank you very much",
    "i need some help with this", "what day is it today",
    "how do i get to the airport", "book a table for two",
    "turn off the television", "dim the bedroom lights",
    "is it going to rain tomorrow", "find a nearby restaurant",
    "read me the news", "what is on television tonight",
    "hey there how are you doing", "could you help me please",
    "i am going to the supermarket", "do you want some coffee",
    "the meeting starts at three o clock", "please call me back later",
]

LENGTH_SCALES  = [0.75, 0.90, 1.00, 1.15, 1.30]
NOISE_SCALES   = [0.333, 0.667, 1.000]
NOISE_W_SCALES = [0.60, 0.80, 1.00]

# ── Audio helpers ─────────────────────────────────────────────────────────────

def synthesize(voice: PiperVoice, text: str) -> np.ndarray:
    cfg = SynthesisConfig(
        length_scale=random.choice(LENGTH_SCALES),
        noise_scale=random.choice(NOISE_SCALES),
        noise_w_scale=random.choice(NOISE_W_SCALES),
    )
    chunks = list(voice.synthesize(text.strip(), cfg))
    audio  = np.frombuffer(b"".join(c.audio_int16_bytes for c in chunks), dtype=np.int16)
    f = audio.astype(np.float32) / 32767.0
    r = scipy.signal.resample_poly(f, 320, 441)   # 22050 → 16000
    return np.clip(r * 32767, -32768, 32767).astype(np.int16)


def make_noise_clip() -> np.ndarray:
    """Gaussian noise at a random amplitude."""
    amp = random.uniform(0.01, 0.25)
    return (np.random.randn(CLIP_SAMPLES) * 32767 * amp).astype(np.int16)


def embed_in_clip(audio: np.ndarray) -> np.ndarray:
    clip = np.zeros(CLIP_SAMPLES, dtype=np.int16)
    if len(audio) >= CLIP_SAMPLES:
        return audio[:CLIP_SAMPLES]
    offset = random.randint(0, CLIP_SAMPLES - len(audio))
    clip[offset:offset + len(audio)] = audio
    return clip


def load_wavs_to_array(directory: Path) -> np.ndarray:
    files = sorted(directory.glob("*.wav"))
    if not files:
        return np.empty((0, CLIP_SAMPLES), dtype=np.int16)
    clips = []
    for f in files:
        with wave.open(str(f)) as wf:
            raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16)
        if len(audio) >= CLIP_SAMPLES:
            clips.append(audio[:CLIP_SAMPLES])
        else:
            padded = np.zeros(CLIP_SAMPLES, dtype=np.int16)
            padded[:len(audio)] = audio
            clips.append(padded)
    return np.vstack(clips)


def extract_features(clips: np.ndarray, F: AudioFeatures, batch: int = 200) -> np.ndarray:
    """Run embed_clips in batches to avoid memory spikes."""
    parts = []
    for i in range(0, len(clips), batch):
        parts.append(F.embed_clips(clips[i:i + batch]))
        print(f"  embedded {min(i + batch, len(clips))}/{len(clips)}", flush=True)
    return np.concatenate(parts, axis=0)


def make_windows(features: np.ndarray, label: int) -> tuple[np.ndarray, np.ndarray]:
    """Slide a WINDOW_FRAMES-wide window over each clip's feature sequence."""
    X, y = [], []
    for clip_feat in features:
        n = clip_feat.shape[0]
        for j in range(0, n - WINDOW_FRAMES):
            X.append(clip_feat[j:j + WINDOW_FRAMES])
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def repeating_loader(loader: DataLoader, max_steps: int):
    """Yield batches from loader, repeating until max_steps batches served."""
    step = 0
    while step < max_steps:
        for batch in loader:
            yield batch
            step += 1
            if step >= max_steps:
                return


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not POSITIVE_DIR.exists() or not list(POSITIVE_DIR.glob("*.wav")):
        print("ERROR: No positive clips found.")
        print("Run:  .venv/Scripts/python.exe scripts/generate_wakeword_samples.py  first.")
        sys.exit(1)

    # 1. Ensure feature models are downloaded
    print("Checking openWakeWord feature models…", flush=True)
    oww_utils.download_models(model_names=[], target_directory=str(OWW_MODELS_DIR))

    # 2. Load generated clips
    print("Loading positive clips…", flush=True)
    pos_clips = load_wavs_to_array(POSITIVE_DIR)
    print(f"  {len(pos_clips)} positive clips loaded", flush=True)

    adv_clips = load_wavs_to_array(ADVERSARIAL_DIR)
    print(f"  {len(adv_clips)} adversarial clips loaded", flush=True)

    # 3. Generate negative speech + noise with piper-tts
    print(f"Generating {N_NEG_SPEECH} negative speech clips…", flush=True)
    voice = PiperVoice.load(str(VOICE_MODEL), use_cuda=False)
    neg_speech = []
    for i in range(N_NEG_SPEECH):
        phrase = NEGATIVE_PHRASES[i % len(NEGATIVE_PHRASES)]
        neg_speech.append(embed_in_clip(synthesize(voice, phrase)))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{N_NEG_SPEECH}", flush=True)
    neg_speech_arr = np.vstack(neg_speech)

    print(f"Generating {N_NEG_NOISE} noise clips…", flush=True)
    neg_noise_arr = np.vstack([make_noise_clip() for _ in range(N_NEG_NOISE)])

    neg_clips = np.concatenate([adv_clips, neg_speech_arr, neg_noise_arr], axis=0)
    print(f"  Total negatives: {len(neg_clips)}", flush=True)

    # 4. Extract embeddings
    F = AudioFeatures(inference_framework="onnx")

    print("Extracting positive features…", flush=True)
    pos_feat = extract_features(pos_clips, F)

    print("Extracting negative features…", flush=True)
    neg_feat = extract_features(neg_clips, F)

    print(f"Feature shapes — pos: {pos_feat.shape}  neg: {neg_feat.shape}", flush=True)

    # 5. Create sliding-window training examples
    print("Creating training windows…", flush=True)
    X_pos, y_pos = make_windows(pos_feat, label=1)
    X_neg, y_neg = make_windows(neg_feat, label=0)

    X = np.concatenate([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])
    print(f"  {len(X)} total windows  ({y.sum():.0f} pos / {(1-y).sum():.0f} neg)", flush=True)

    # Shuffle and split 90/10 train/val
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(len(X) * 0.9)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # 6. Train
    print(f"\nTraining for {TRAIN_STEPS} steps (batch_size={BATCH_SIZE})…", flush=True)
    trainer = OWWTrainer(input_shape=(WINDOW_FRAMES, 96))
    val_at  = list(range(1000, TRAIN_STEPS, 500))

    trainer.train_model(
        X=repeating_loader(train_loader, TRAIN_STEPS),
        X_val=val_loader,
        max_steps=TRAIN_STEPS,
        warmup_steps=1000,
        hold_steps=3000,
        val_steps=val_at,
        lr=0.0001,
        negative_weight_schedule=[5] * TRAIN_STEPS,  # upweight negatives to reduce false positives
    )

    # 7. Export ONNX
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    trainer.export_to_onnx(str(OUT_MODEL))

    print(f"\nModel saved → {OUT_MODEL}", flush=True)
    print("Switch to it in config.yaml:", flush=True)
    print("  wake_word:", flush=True)
    print("    model: models/ok_dann.onnx", flush=True)


if __name__ == "__main__":
    main()
