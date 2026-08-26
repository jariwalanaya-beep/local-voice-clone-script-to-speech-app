#!/usr/bin/env bash
source /home/shivam/va/env.sh
cd "$VA_ROOT/GPT-SoVITS"
export PYTHONPATH="$VA_ROOT/GPT-SoVITS:$VA_ROOT/GPT-SoVITS/tools"
python tools/asr/fasterwhisper_asr.py \
  -i "$VA_ROOT/data/myvoice" \
  -o "$VA_ROOT/data/logs" \
  -s medium.en -l en -p float16
echo "ASR_EXIT=$?"
