# LocalVoiceClone

Clone a voice from a five-minute recording and read scripts back in it — entirely
on your own machine. No cloud, no API keys, no audio leaving the box.

Built on [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v2Pro with a small
FastAPI front end: paste a script, pick a delivery style, set the pace, get a WAV.

Verified on Fedora 44 (KDE Plasma) with an NVIDIA RTX 3060 12 GB.

---

## What it does

Paste 120–180 words, choose a delivery style, adjust speed, hit Generate. The
page returns audio in the cloned voice with a player and a download button.

- **Word counter** warns outside 120–180 words but never blocks you.
- **Delivery styles** — calm / dramatic / urgent, each backed by a reference clip.
- **Speed** — 0.5×–1.5×, applied natively in the vocoder so pitch is unaffected.
- **Long scripts** are split on sentence boundaries, generated sequentially, and
  concatenated with scaled pauses, so a 180-word script doesn't break.
- **Loopback only** — binds `127.0.0.1`, unreachable from your LAN.

Measured on a 3060: a 144-word script renders in **~3.5 s**.

---

## How delivery actually works

GPT-SoVITS has no emotion parameter. Delivery is inherited from a short
**reference clip** — pace, pitch movement, energy and attitude all come from
that recording. Picking a "style" really means picking which performance to
imitate.

So the quality of your three reference clips matters more than any setting. The
repo includes a script that finds candidates by measuring per-segment pitch,
pitch variance, energy and syllable rate across your source recording, then
picking the extremes. Better still: record three deliberate 8-second takes.

Each clip needs its transcript in `webapp/config.json` — the model conditions on
it, and delivery transfer is noticeably worse without it.

---

## Requirements

| | |
|---|---|
| GPU | NVIDIA, 8 GB+ VRAM (trained on a 3060 12 GB; peak 8.3 GB) |
| Python | 3.9–3.10 — **never** the system interpreter |
| Tools | `git`, `ffmpeg`, a C++ compiler |
| Disk | ~17 GB (venv 11 GB, models 3 GB, training artefacts 2.6 GB) |
| Audio | 5+ minutes of clean single-speaker speech |

---

## Setup

```bash
git clone https://github.com/jariwalanaya-beep/LocalVoiceClone.git
cd LocalVoiceClone

# 1. isolated env (uv shown; conda works too)
uv venv --python 3.10 gptsovits
uv pip install --python gptsovits/bin/python torch torchaudio torchcodec \
    --index-url https://download.pytorch.org/whl/cu128

# 2. upstream project
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
uv pip install --python gptsovits/bin/python -r GPT-SoVITS/extra-req.txt --no-deps
uv pip install --python gptsovits/bin/python -r GPT-SoVITS/requirements.txt

# 3. v2Pro weights only (~1.3 GB, not the 4.35 GB all-versions bundle)
bash download_v2pro.sh
```

Edit the paths at the top of `env.sh`, then `source env.sh` before anything else.

### Three install traps worth knowing

These cost real time to diagnose, so they are documented rather than papered over.

**1. A path containing a space breaks the build.** setuptools splits `CXX` on
whitespace and dies with `No such file or directory: '/home/you/My'`. Several of
GPT-SoVITS's internal `shell=True` calls break the same way. If your checkout
lives somewhere with a space, symlink it to a space-free path and work through
the symlink.

**2. `g++` may be missing even when `gcc` is present.** `opencc` and
`pyopenjtalk` build C++ and fail with `No CMAKE_CXX_COMPILER could be found`.
On Fedora, `sudo dnf install gcc-c++`. Without root you can extract the RPM into
a user prefix — see the notes in `env.sh`.

**3. `torchaudio.load` fails with `libnppicc.so.12: cannot open shared object
file`.** Modern torchaudio routes through `torchcodec`, which dlopens NVIDIA NPP
libraries that PyTorch's own wheels don't install. Fix:

```bash
uv pip install --python gptsovits/bin/python nvidia-npp-cu12
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH"
```

Installing alone is not enough — the directory must be on the loader path.
`env.sh` handles this. The failure is nasty because the preprocessing step exits
**0** while writing nothing, and training then dies with an unrelated
`ZeroDivisionError` in the dataset loader.

---

## Training

Point `run_pipeline.py` at your audio, then:

```bash
source env.sh
bash train.sh
```

That slices the recording, transcribes it with faster-whisper, extracts features,
and runs both stages. On a 3060 with five minutes of audio the whole thing takes
about **four minutes**.

| Stage | Epochs | Batch | Time | Peak VRAM |
|-------|--------|-------|------|-----------|
| SoVITS (s2) | 8 | 6 | 98 s | 8.3 GB |
| GPT (s1) | 15 | 6 | 39 s | 6.9 GB |

These are the upstream few-shot defaults, chosen to avoid overfitting on a small
dataset. Intermediate checkpoints are kept so you can A/B them.

If you hit OOM, drop `SOVITS_BATCH` / `GPT_BATCH` in `run_pipeline.py`.

Then wire the app to the newest weights:

```bash
python make_config.py
```

---

## Running

```bash
bash start.sh
```

Open **http://127.0.0.1:8000**.

`bash webui.sh` launches the upstream Gradio UI instead, if you want the full
training interface.

---

## Verifying a clone objectively

Don't trust your ears alone — you know what you sound like, which biases you.
Embed the generated audio and the original with a speaker-verification model
(the ERes2Net checkpoint that ships with GPT-SoVITS works) and compare cosine
similarity against two baselines:

- **ceiling** — two disjoint chunks of the *real* speaker
- **floor** — a genuinely different speaker

A good clone lands at or above the ceiling. This build scored **0.77–0.88**
against a 0.80 ceiling and a 0.11 floor.

---

## Layout

```
env.sh                  environment: venv, compiler, LD_LIBRARY_PATH, telemetry off
start.sh                run the web app
train.sh                preprocessing + both training stages
webui.sh                upstream Gradio UI
run_pipeline.py         headless training pipeline
make_config.py          point the app at the newest weights
download_v2pro.sh       fetch only the v2Pro weights
run_asr.sh              transcribe sliced segments
webapp/app.py           FastAPI backend, model loaded once at startup
webapp/static/          single-page front end
webapp/config.example.json
```

Not in the repo, by design: `data/`, `refs/`, `samples/`, model weights, the
GPT-SoVITS clone, and the virtualenv. See `.gitignore`.

---

## A word on consent

This clones a human voice well enough to be mistaken for the real thing.

Clone your own voice, or one you have explicit permission to use. Don't use it
to impersonate anyone, and don't publish weights trained on someone else's voice
without their agreement. A voice model is biometric data — treat it the way you
would treat a fingerprint.

Several jurisdictions now regulate synthetic voice likeness. Worth knowing where
you stand before publishing anything.

---

## Credits

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) by RVC-Boss — MIT
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for transcription

## Licence

MIT — see [LICENSE](LICENSE). Upstream GPT-SoVITS is MIT; the pretrained weights
carry their own terms.
