#!/usr/bin/env bash
source /home/shivam/va/env.sh
PM="$VA_ROOT/GPT-SoVITS/GPT_SoVITS/pretrained_models"
BASE="https://huggingface.co/lj1995/GPT-SoVITS/resolve/main"
mkdir -p "$PM/chinese-hubert-base" "$PM/chinese-roberta-wwm-ext-large" "$PM/v2Pro" "$PM/sv"

FILES=(
  "chinese-hubert-base/config.json"
  "chinese-hubert-base/preprocessor_config.json"
  "chinese-hubert-base/pytorch_model.bin"
  "chinese-roberta-wwm-ext-large/config.json"
  "chinese-roberta-wwm-ext-large/tokenizer.json"
  "chinese-roberta-wwm-ext-large/pytorch_model.bin"
  "s1v3.ckpt"
  "v2Pro/s2Gv2Pro.pth"
  "v2Pro/s2Dv2Pro.pth"
  "sv/pretrained_eres2netv2w24s4ep4.ckpt"
)

fetch(){
  local rel="$1" out="$PM/$1"
  if [ -s "$out" ]; then echo "  SKIP $rel"; return 0; fi
  wget -q -c -O "$out.part" "$BASE/$rel" && mv "$out.part" "$out" \
    && echo "  OK   $rel ($(du -h "$out" | cut -f1))" \
    || { echo "  FAIL $rel"; return 1; }
}
export -f fetch; export PM BASE

echo "=== v2Pro pretrained models -> $PM ==="
printf '%s\n' "${FILES[@]}" | xargs -P 4 -I{} bash -c 'fetch "$@"' _ {}

echo
echo "=== faster-whisper medium.en ==="
cd "$VA_ROOT/GPT-SoVITS"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download("Systran/faster-whisper-medium.en",
        local_dir="tools/asr/models/faster-whisper-medium.en",
        allow_patterns=["config.json","model.bin","tokenizer.json","vocabulary.txt","preprocessor_config.json"])
print("whisper ->", p)
PY
echo "=== DOWNLOADS COMPLETE ==="
