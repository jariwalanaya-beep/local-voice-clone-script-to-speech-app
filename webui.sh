#!/usr/bin/env bash
# Launch the official GPT-SoVITS Gradio WebUI (training + inference UI).
set -e
source /home/shivam/va/env.sh
cd "$VA_ROOT/GPT-SoVITS"
exec "$VA_ROOT/gptsovits/bin/python" webui.py en_US
