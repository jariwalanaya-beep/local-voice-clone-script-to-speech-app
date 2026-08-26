# GPT-SoVITS project environment (isolated venv, no system Python, no root)
# NOTE: /home/shivam/va is a symlink to "/home/shivam/Documents/voice agent".
# We always use the space-free path because setuptools/cmake/shell calls
# inside GPT-SoVITS break on paths containing spaces.
# Override by exporting VA_ROOT before sourcing, or edit this line.
# Must not contain spaces - see README.
export VA_ROOT="${VA_ROOT:-/home/shivam/va}"
export VA_TOOLCHAIN="$VA_ROOT/toolchain"
export PATH="$VA_TOOLCHAIN/usr/bin:$VA_ROOT/gptsovits/bin:$PATH"
export CC=gcc
export CXX="$VA_TOOLCHAIN/usr/bin/g++"
export VIRTUAL_ENV="$VA_ROOT/gptsovits"
export PYTHONNOUSERSITE=1
export HF_HOME="$VA_ROOT/hf_cache"
export TOKENIZERS_PARALLELISM=false
# --- telemetry off: no outbound calls beyond model downloads ---
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export MODELSCOPE_LOG_LEVEL=40
export SENTRY_DSN=""
export BITSANDBYTES_NOWELCOME=1

# torchcodec (used by torchaudio.load) dlopen()s NVIDIA libs directly, so the
# pip-installed nvidia/*/lib dirs must be on the loader path.
NV_LIBS=$(ls -d "$VIRTUAL_ENV"/lib/python3.*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
