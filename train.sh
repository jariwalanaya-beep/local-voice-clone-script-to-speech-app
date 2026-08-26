#!/usr/bin/env bash
# Re-run the full training pipeline (preprocess -> SoVITS -> GPT).
set -e
source /home/shivam/va/env.sh
exec "$VA_ROOT/gptsovits/bin/python" "$VA_ROOT/run_pipeline.py"
