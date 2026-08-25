#!/usr/bin/env bash
# Guided setup for Dann of Thursday — the local voice AI agent.
#
# This walks you through each piece of the pipeline (wake word -> STT ->
# LLM -> TTS), explains what it does and why it's needed, and only makes
# changes you confirm. Safe to re-run: every step first checks whether
# it's already done and skips it.
#
#   bash scripts/setup.sh
#
# Pass -y / --yes to accept the recommended default at every prompt
# (useful for a fast re-run once you already know your answers).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

AUTO_YES=0
for arg in "$@"; do
  case "$arg" in
    -y|--yes) AUTO_YES=1 ;;
  esac
done

BOLD=$(tput bold 2>/dev/null || true)
DIM=$(tput dim 2>/dev/null || true)
RESET=$(tput sgr0 2>/dev/null || true)

step()    { printf "\n%s== %s ==%s\n" "$BOLD" "$1" "$RESET"; }
explain() { printf "%s%s%s\n" "$DIM" "$1" "$RESET"; }
ok()      { printf "  \xE2\x9C\x93 %s\n" "$1"; }   # checkmark
todo()    { printf "  -> %s\n" "$1"; }

confirm() {
  # confirm "question" [default: Y/n]  -> returns 0 for yes.
  # With -y/--yes, takes the recommended default instead of prompting.
  local prompt="$1" default="${2:-Y}" reply
  if [ "$AUTO_YES" = "1" ]; then [[ "$default" =~ ^[Yy] ]]; return; fi
  local suffix="[Y/n]"; [ "$default" = "n" ] && suffix="[y/N]"
  read -r -p "  $prompt $suffix " reply || true
  reply="${reply:-$default}"
  [[ "$reply" =~ ^[Yy] ]]
}

get_config_value() {
  # get_config_value <section> <key>  — prints the value of "<key>:" inside
  # the top-level "<section>:" block of config.yaml (empty if not found).
  # Section-scoped so e.g. wake_word.model and a same-named key elsewhere
  # can't collide.
  python3 - "$1" "$2" <<'PYEOF'
import re, sys, pathlib
section, key = sys.argv[1], sys.argv[2]
text = pathlib.Path("config.yaml").read_text()
m = re.search(rf"^{re.escape(section)}:\n((?:[ \t].*\n?)*)", text, re.MULTILINE)
block = m.group(1) if m else ""
m2 = re.search(rf"^\s*{re.escape(key)}:\s*(\S+)", block, re.MULTILINE)
print(m2.group(1) if m2 else "", end="")
PYEOF
}

set_config_value() {
  # set_config_value <section> <key> <value>  — replaces "<key>: ..." inside
  # the top-level "<section>:" block with "<key>: <value>", leaving
  # everything else (including comments) untouched.
  python3 - "$1" "$2" "$3" <<'PYEOF'
import re, sys, pathlib
section, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path("config.yaml")
text = p.read_text()
sec_match = re.search(rf"^{re.escape(section)}:\n((?:[ \t].*\n?)*)", text, re.MULTILINE)
if not sec_match:
    print(f"  warning: no '{section}:' section in config.yaml — set {key} manually")
else:
    block = sec_match.group(1)
    pattern = re.compile(rf"^(\s*{re.escape(key)}:).*$", re.MULTILINE)
    new_block, n = pattern.subn(lambda m: f"{m.group(1)} {value}", block, count=1)
    if n:
        text = text[:sec_match.start(1)] + new_block + text[sec_match.end(1):]
        p.write_text(text)
        print(f"  set {section}.{key} = {value}")
    else:
        print(f"  warning: couldn't find '{key}:' under '{section}:' — set it manually")
PYEOF
}

cat <<EOF
${BOLD}Dann of Thursday — guided setup${RESET}

Dann is a local voice pipeline:

  "ok Dann" (wake word) -> record -> STT -> local LLM -> TTS -> speak
       Porcupine/OWW      mic      Whisper    Ollama     Piper   speaker

Everything runs on this machine — no audio or transcript leaves it unless
you explicitly route a question to Claude Code. This script sets up each
stage in order and explains what it's for before touching anything.
EOF

# ── 1. Python environment ─────────────────────────────────────────────────
step "1/5  Python environment"
explain "Dann's voice pipeline and its FastAPI backend are Python. This
creates an isolated virtualenv (.venv) so its dependencies (faster-whisper,
piper-tts, pvporcupine, fastapi, ...) don't collide with anything else on
your system, then installs everything from requirements.txt into it."

if [ -d .venv ]; then
  ok ".venv already exists"
else
  if confirm "Create .venv and install dependencies now?"; then
    python3 -m venv .venv
    ok "created .venv"
  else
    todo "skipped — nothing else in this script will work until you run:"
    todo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 0
  fi
fi

if [ -f .venv/bin/pip ]; then
  if confirm "Install/update Python dependencies from requirements.txt?"; then
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -r requirements.txt
    ok "dependencies installed"
  fi
fi

# ── 2. Config file ─────────────────────────────────────────────────────────
step "2/5  Config file"
explain "config.yaml controls every stage below: which wake word engine and
model to use, which Whisper size, which Ollama model, which Piper voice,
audio device selection, and the list of coding projects Dann knows about.
It's gitignored so your personal keys never get committed; config.example.yaml
is the checked-in template with sane defaults and comments."

if [ -f config.yaml ]; then
  ok "config.yaml already exists — leaving it as-is"
else
  cp config.example.yaml config.yaml
  ok "created config.yaml from config.example.yaml"
fi

# ── 3. Wake word ────────────────────────────────────────────────────────────
step "3/5  Wake word (\"ok Dann\")"
explain "The wake word engine listens to a live mic stream for one specific
phrase and only wakes the rest of the pipeline (which is heavier — STT,
an LLM call, TTS) once it hears it. config.yaml's wake_word.engine picks
which of the two supported engines is active; this step adapts to
whichever one is already set (see wake-word.md for a full comparison)."

wake_engine=$(get_config_value wake_word engine)
wake_engine="${wake_engine:-porcupine}"

if [ "$wake_engine" = "openwakeword" ]; then
  explain "Engine: openWakeWord — fully offline, no account or access key.
  Built-in phrases (e.g. hey_jarvis) are auto-downloaded by the library on
  first run; a custom \"ok Dann\" model needs training (see wake-word.md)."
  wake_model=$(get_config_value wake_word model)
  wake_model="${wake_model:-hey_jarvis}"
  if [[ "$wake_model" == *.onnx ]]; then
    if [ -f "$wake_model" ]; then
      ok "wake word model already at $wake_model"
    else
      todo "wake_word.model is set to $wake_model but that file doesn't exist"
      todo "  train a custom model (wake-word.md) or point wake_word.model at a built-in name like hey_jarvis"
    fi
  else
    ok "using built-in openWakeWord phrase '$wake_model' — auto-downloads on first run"
  fi
else
  explain "Engine: Picovoice Porcupine — needs (a) a free AccessKey tied to
  your account and (b) a custom .ppn acoustic model trained on the phrase
  \"ok Dann\", both obtained from the Picovoice web console."
  model_path=$(get_config_value wake_word model_path)
  model_path="${model_path:-models/ok_dann.ppn}"
  if [ -f "$model_path" ]; then
    ok "$model_path already present"
  else
    todo "no wake word model found at $model_path"
    if confirm "Open the Picovoice console in your browser to create one?" n; then
      open "https://console.picovoice.ai/" 2>/dev/null || echo "  -> https://console.picovoice.ai/"
    fi
    explain "  In the console: sign up (free) -> Porcupine -> Create Custom Wake
    Word -> type \"ok Dann\" -> pick macOS as the platform -> download the
    .ppn file -> save it as $model_path in this repo."
  fi

  access_key=$(get_config_value wake_word access_key)
  if [ -z "$access_key" ] || [ "$access_key" = "YOUR_PICOVOICE_ACCESS_KEY" ]; then
    if [ "$AUTO_YES" != "1" ]; then
      read -r -p "  Paste your Picovoice AccessKey (or press Enter to skip): " entered_key || true
      if [ -n "${entered_key:-}" ]; then
        set_config_value wake_word access_key "$entered_key"
      else
        todo "skipped — set wake_word.access_key in config.yaml before running Dann"
      fi
    fi
  else
    ok "wake_word.access_key already set in config.yaml"
  fi
fi

# ── 4. Ollama (LLM) ─────────────────────────────────────────────────────────
step "4/5  Ollama (local LLM)"
explain "Ollama runs the actual language model locally and serves it over
an HTTP API on localhost:11434 — Dann's src/llm/ollama.py just calls that
API. Nothing here talks to a cloud model unless you explicitly route a
question to Claude Code."

if command -v ollama >/dev/null 2>&1; then
  ok "ollama is installed"
else
  if confirm "Install Ollama via Homebrew?"; then
    command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }
    brew install ollama
    ok "installed ollama"
  else
    todo "skipped — install from https://ollama.com and re-run this script"
  fi
fi

if command -v ollama >/dev/null 2>&1; then
  if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    todo "ollama isn't running"
    if confirm "Start it now (brew services start ollama)?"; then
      brew services start ollama
      sleep 2
      ok "ollama service started"
    fi
  else
    ok "ollama is running"
  fi

  model=$(get_config_value ollama model)
  model="${model:-llama3.2}"
  if curl -sf http://localhost:11434/api/tags 2>/dev/null | grep -q "\"name\":\"${model}"; then
    ok "model '$model' already pulled"
  else
    explain "config.yaml has ollama.model = $model. This is the model Dann
    calls on every turn, so pull it now (a few GB download, one-time)."
    if confirm "Run 'ollama pull $model' now?"; then
      ollama pull "$model"
      ok "pulled $model"
    else
      todo "skipped — run 'ollama pull $model' yourself, or change ollama.model in config.yaml"
    fi
  fi
fi

# ── 5. Piper (TTS) ──────────────────────────────────────────────────────────
step "5/5  Piper (text-to-speech)"
explain "Piper turns the LLM's text reply into speech. The piper-tts Python
package (already in requirements.txt) is the engine; it still needs a
voice model — a small neural net trained on one speaker/accent — which
config.yaml's tts.voice_model points at."

voice_model=$(get_config_value tts voice_model)
voice_model="${voice_model:-models/piper/en_US-lessac-medium}"
if [ -f "${voice_model}.onnx" ]; then
  ok "voice model already at ${voice_model}.onnx"
else
  todo "no voice model at ${voice_model}.onnx"
  if confirm "Download the default voice (en_US-lessac-medium, ~60MB) now?"; then
    mkdir -p models/piper
    base="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
    curl -L -o models/piper/en_US-lessac-medium.onnx "${base}/en_US-lessac-medium.onnx"
    curl -L -o models/piper/en_US-lessac-medium.onnx.json "${base}/en_US-lessac-medium.onnx.json"
    ok "downloaded voice model to models/piper/"
  else
    todo "skipped — see https://github.com/rhasspy/piper/releases for other voices"
  fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
step "Done"
explain "Recap of what this pipeline needs to actually work, so you can see
at a glance if anything above was skipped:"

check() { [ -e "$2" ] && ok "$1" || todo "$1 — MISSING"; }
[ -d .venv ] && ok "Python venv" || todo "Python venv — MISSING"
[ -f config.yaml ] && ok "config.yaml" || todo "config.yaml — MISSING"

if [ "$wake_engine" = "openwakeword" ]; then
  if [[ "$wake_model" == *.onnx ]]; then
    check "wake word model ($wake_model)" "$wake_model"
  else
    ok "wake word: built-in openWakeWord phrase '$wake_model'"
  fi
else
  check "wake word model ($model_path)" "$model_path"
  access_key=$(get_config_value wake_word access_key)
  [ -n "$access_key" ] && [ "$access_key" != "YOUR_PICOVOICE_ACCESS_KEY" ] \
    && ok "Picovoice access_key set" \
    || todo "Picovoice access_key — MISSING (edit config.yaml)"
fi
command -v ollama >/dev/null 2>&1 && ok "Ollama installed" || todo "Ollama — MISSING"
check "Piper voice model" "${voice_model}.onnx"

cat <<EOF

${BOLD}Run it:${RESET}
  source .venv/bin/activate
  python -m src.main

Then say "ok Dann" and ask a question.

${BOLD}Dashboard (optional):${RESET} the FastAPI backend + React UI let you watch
the pipeline live and browse/run coding projects.
  NO_VOICE=1 .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
  cd ui && npm install && npm run dev   # http://localhost:3000

${BOLD}Remote access from another machine's browser:${RESET} see the
"Local AI Workstation" section in README.md and scripts/setup_workstation.sh.
EOF
