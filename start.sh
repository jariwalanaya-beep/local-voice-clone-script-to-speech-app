#!/usr/bin/env bash
# Start the local voice-clone web app.
#   usage:  ~/va/start.sh          (or:  "/home/shivam/Documents/voice agent/start.sh")
set -e
source /home/shivam/va/env.sh
cd "$VA_ROOT/webapp"
echo "Loading the fine-tuned model onto the RTX 3060 (first start takes ~20-30s)..."
echo "Then open:  http://127.0.0.1:8000"
exec "$VA_ROOT/gptsovits/bin/python" -m uvicorn app:app --host 127.0.0.1 --port 8000
