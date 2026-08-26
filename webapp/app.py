"""
Local voice-cloning web app for a fine-tuned GPT-SoVITS model.

Everything runs on this machine. The model is loaded once at startup and
reused for every request. No network calls are made at inference time.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import uuid
import wave
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GSV = ROOT / "GPT-SoVITS"
OUTPUTS = HERE / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# GPT-SoVITS expects to be imported from its own root, and resolves several
# model paths relative to the current working directory.
sys.path.insert(0, str(GSV))
sys.path.insert(0, str(GSV / "GPT_SoVITS"))
os.chdir(GSV)

CONFIG_PATH = HERE / "config.json"

# ----------------------------------------------------------------------------
# Text handling
# ----------------------------------------------------------------------------

# Abbreviations that end in a period but do not end a sentence.
_ABBREV = r"(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bProf)(?<!\bSt)(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)(?<!\betc)"
_SENT_END = re.compile(rf"{_ABBREV}(?<=[.!?…])[\"')\]]*\s+")

# How many characters we aim for per generated chunk. GPT-SoVITS degrades and
# can drift on very long inputs, so we keep each pass comfortably short.
TARGET_CHARS = 220
MAX_CHARS = 300


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping terminal punctuation."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = [p.strip() for p in _SENT_END.split(text) if p and p.strip()]
    out: list[str] = []
    for p in parts:
        # A sentence longer than MAX_CHARS is broken further on clause
        # boundaries so a single runaway sentence cannot blow up a pass.
        if len(p) <= MAX_CHARS:
            out.append(p)
            continue
        buf = ""
        for clause in re.split(r"(?<=[,;:])\s+", p):
            if buf and len(buf) + len(clause) + 1 > MAX_CHARS:
                out.append(buf.strip())
                buf = clause
            else:
                buf = f"{buf} {clause}".strip()
        if buf.strip():
            out.append(buf.strip())
    return out


def group_sentences(sentences: list[str]) -> list[str]:
    """Pack whole sentences into chunks of roughly TARGET_CHARS."""
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > TARGET_CHARS:
            chunks.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        chunks.append(buf)
    return chunks


# ----------------------------------------------------------------------------
# Audio helpers
# ----------------------------------------------------------------------------

def trim_silence(a: np.ndarray, sr: int, thresh_db: float = -45.0,
                 keep_ms: int = 60) -> np.ndarray:
    """Trim near-silence from both ends, leaving a short natural margin."""
    if a.size == 0:
        return a
    x = a.astype(np.float32) / 32768.0
    win = max(1, int(sr * 0.010))
    n = len(x) // win
    if n < 2:
        return a
    frames = x[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-12
    db = 20 * np.log10(rms)
    voiced = np.where(db > thresh_db)[0]
    if voiced.size == 0:
        return a
    keep = int(sr * keep_ms / 1000)
    start = max(0, voiced[0] * win - keep)
    end = min(len(a), (voiced[-1] + 1) * win + keep)
    return a[start:end]


def concat(pieces: list[np.ndarray], sr: int, gap_ms: int = 260) -> np.ndarray:
    """Join chunks with a short silence, then normalise the whole take once."""
    if not pieces:
        return np.zeros(0, dtype=np.int16)
    gap = np.zeros(int(sr * gap_ms / 1000), dtype=np.int16)
    out: list[np.ndarray] = []
    for i, p in enumerate(pieces):
        if i:
            out.append(gap)
        out.append(p)
    joined = np.concatenate(out).astype(np.float32)

    # Peak-normalise to -1 dBFS. Done once across the full take so chunk
    # boundaries do not have audible level jumps.
    peak = float(np.abs(joined).max())
    if peak > 0:
        joined *= (32767.0 * 0.891) / peak
    return np.clip(joined, -32768, 32767).astype(np.int16)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class Engine:
    def __init__(self) -> None:
        self.tts = None
        self.cfg: dict = {}
        self.styles: dict = {}
        self.lock = threading.Lock()   # the model is not re-entrant
        self.ready = False
        self.error: str | None = None

    def load(self) -> None:
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        self.cfg = json.loads(CONFIG_PATH.read_text())
        self.styles = self.cfg["styles"]

        version = self.cfg.get("version", "v2Pro")
        pm = GSV / "GPT_SoVITS/pretrained_models"

        # TTS_Config reads its settings from the "custom" key. Setting the
        # weight paths here (rather than mutating the object afterwards) is
        # what makes it load OUR fine-tuned checkpoints - update_version()
        # alone leaves the stock v2 paths in place.
        conf = TTS_Config({
            "custom": {
                "device": self.cfg.get("device", "cuda"),
                "is_half": bool(self.cfg.get("is_half", True)),
                "version": version,
                "t2s_weights_path": self.cfg["gpt_weights"],
                "vits_weights_path": self.cfg["sovits_weights"],
                "bert_base_path": str(pm / "chinese-roberta-wwm-ext-large"),
                "cnhuhbert_base_path": str(pm / "chinese-hubert-base"),
            }
        })
        assert conf.t2s_weights_path == self.cfg["gpt_weights"], \
            f"config fell back to stock weights: {conf.t2s_weights_path}"
        assert conf.vits_weights_path == self.cfg["sovits_weights"], \
            f"config fell back to stock weights: {conf.vits_weights_path}"

        t0 = time.time()
        self.tts = TTS(conf)
        self.ready = True
        print(f"[engine] model loaded in {time.time() - t0:.1f}s "
              f"({version}, {conf.device}, half={conf.is_half})", flush=True)

    def synth_chunk(self, text: str, style: dict, speed: float) -> tuple[int, np.ndarray]:
        inputs = {
            "text": text,
            "text_lang": "en",
            "ref_audio_path": style["ref_audio"],
            "prompt_text": style.get("prompt_text", ""),
            "prompt_lang": "en",
            "aux_ref_audio_paths": [],
            "top_k": int(style.get("top_k", 15)),
            "top_p": float(style.get("top_p", 1.0)),
            "temperature": float(style.get("temperature", 1.0)),
            "text_split_method": "cut0",   # we do our own splitting
            "batch_size": 1,
            # >1 speeds up, <1 slows down. Applied natively in the vocoder
            # (upsample_rate / speed_factor), not as a post-hoc time-stretch.
            "speed_factor": float(speed),
            "split_bucket": False,
            "return_fragment": False,
            "fragment_interval": 0.3,
            "seed": -1,
            "parallel_infer": True,
            "repetition_penalty": float(style.get("repetition_penalty", 1.35)),
            "sample_steps": 32,
            "super_sampling": False,
        }
        sr, audio = next(self.tts.run(inputs))
        return sr, audio


ENGINE = Engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CONFIG_PATH.exists():
        ENGINE.error = f"missing {CONFIG_PATH}. Training has not been completed yet."
        print(f"[engine] {ENGINE.error}", flush=True)
    else:
        try:
            ENGINE.load()
        except Exception:
            ENGINE.error = traceback.format_exc()
            print(f"[engine] FAILED TO LOAD:\n{ENGINE.error}", flush=True)
    yield


app = FastAPI(title="Local Voice Clone", lifespan=lifespan)


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    style: str = "calm"
    # 1.0 is the pace of the reference clip; lower is slower.
    speed: float = Field(1.0, ge=0.5, le=1.5)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((HERE / "static/index.html").read_text())


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({
        "ready": ENGINE.ready,
        "error": ENGINE.error,
        "styles": [
            {"id": k, "label": v.get("label", k), "description": v.get("description", ""),
             "ref_audio": os.path.basename(v.get("ref_audio", "")),
             "ref_exists": os.path.exists(v.get("ref_audio", ""))}
            for k, v in ENGINE.styles.items()
        ],
    })


@app.get("/api/reference/{style_id}")
def reference_clip(style_id: str):
    """Let the page play the reference clip that drives the delivery."""
    style = ENGINE.styles.get(style_id)
    if not style or not os.path.exists(style.get("ref_audio", "")):
        raise HTTPException(404, "reference clip not found")
    return FileResponse(style["ref_audio"], media_type="audio/wav")


@app.post("/generate")
def generate(req: GenerateRequest):
    if not ENGINE.ready:
        raise HTTPException(503, ENGINE.error or "model is still loading")
    style = ENGINE.styles.get(req.style)
    if style is None:
        raise HTTPException(400, f"unknown style '{req.style}'")
    if not os.path.exists(style["ref_audio"]):
        raise HTTPException(500, f"reference clip missing: {style['ref_audio']}")

    chunks = group_sentences(split_sentences(req.text))
    if not chunks:
        raise HTTPException(400, "no speakable text")

    t0 = time.time()
    pieces: list[np.ndarray] = []
    sr = 32000
    with ENGINE.lock:                     # one generation at a time on the GPU
        for i, chunk in enumerate(chunks, 1):
            print(f"[gen] chunk {i}/{len(chunks)} ({len(chunk)} chars, speed {req.speed})", flush=True)
            sr, audio = ENGINE.synth_chunk(chunk, style, req.speed)
            pieces.append(trim_silence(audio, sr))

    # Slower delivery wants slightly longer pauses between chunks, or the
    # sentences run together even though the words themselves slowed down.
    final = concat(pieces, sr, gap_ms=int(260 / req.speed))
    name = f"{req.style}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}.wav"
    path = OUTPUTS / name
    write_wav(path, final, sr)

    dur = len(final) / sr
    print(f"[gen] done: {dur:.1f}s audio from {len(chunks)} chunk(s) "
          f"in {time.time() - t0:.1f}s -> {path}", flush=True)

    return FileResponse(
        path, media_type="audio/wav", filename=name,
        headers={
            "X-Audio-Seconds": f"{dur:.2f}",
            "X-Chunks": str(len(chunks)),
            "X-Generation-Seconds": f"{time.time() - t0:.2f}",
            "X-Word-Count": str(word_count(req.text)),
            "X-Speed": f"{req.speed:.2f}",
        },
    )
