#!/usr/bin/env bash
# Installs everything Pipo needs and downloads the models. Safe to re-run: what is there is skipped.
#   scripts/setup.sh          install
#   scripts/setup.sh --check  only report what is missing (exit 1 if anything is)
set -uo pipefail
cd "$(dirname "$0")/.."

CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1
MODELS=models
QWEN_URL="https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
QWEN_FILE="qwen2.5-7b-instruct-q4_k_m.gguf"
KOKORO_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_FILES=(kokoro-v1.0.onnx voices-v1.0.bin)

failed=0
ok() { echo "  ok   $1"; }
fail() { echo "  FAIL $1"; failed=1; }
todo() { echo "  todo $1"; failed=1; }

command -v uv >/dev/null && ok "uv" || fail "install uv (curl -LsSf https://astral.sh/uv/install.sh | sh)"
pactl info >/dev/null 2>&1 && ok "sound server" || fail "no PulseAudio/PipeWire sound server"

if [[ -x .venv/bin/python && .venv -nt uv.lock ]]; then ok ".venv"
elif ((CHECK)); then todo ".venv missing or older than uv.lock (run make setup)"
else { [[ -f uv.lock ]] || uv lock --python 3.12; } && uv sync --frozen --python 3.12 && touch .venv && ok ".venv installed" || fail "uv sync failed"; fi

mkdir -p "$MODELS"
fetch() { # url file
  if [[ -s "$MODELS/$2" ]]; then ok "$2"
  elif ((CHECK)); then todo "$2 not downloaded"
  else echo "  ...  downloading $2"; curl -fL --retry 3 -o "$MODELS/$2.part" "$1" && mv "$MODELS/$2.part" "$MODELS/$2" && ok "$2" || fail "could not download $2"; fi
}
fetch "$QWEN_URL" "$QWEN_FILE"
for f in "${KOKORO_FILES[@]}"; do fetch "$KOKORO_URL/$f" "$f"; done
# Whisper large-v3-turbo comes from Hugging Face on first use and is cached in ~/.cache/huggingface.

exit $failed
