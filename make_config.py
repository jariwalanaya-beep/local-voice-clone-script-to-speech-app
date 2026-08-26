#!/usr/bin/env python
"""Point the web app at the newest fine-tuned weights and the reference clips."""
import json, re, sys
from pathlib import Path

ROOT = Path("/home/shivam/va")
GSV  = ROOT / "GPT-SoVITS"

def newest(folder: Path, pattern: str, epoch_re: str):
    cands = sorted(folder.glob(pattern))
    if not cands:
        sys.exit(f"no weights matching {folder}/{pattern} — did training finish?")
    def ep(p):
        m = re.search(epoch_re, p.name)
        return int(m.group(1)) if m else -1
    best = max(cands, key=ep)
    return best, ep(best)

sovits, se = newest(GSV / "SoVITS_weights_v2Pro", "myvoice_e*.pth", r"_e(\d+)")
gpt,    ge = newest(GSV / "GPT_weights_v2Pro",    "myvoice-e*.ckpt", r"-e(\d+)")

prompts = json.loads((ROOT / "refs/prompt_texts.json").read_text())

DESC = {
    "calm":     "even, unhurried, low energy",
    "dramatic": "higher pitch, wide dynamics, deliberate",
    "urgent":   "fast, pressed, driving",
}

cfg = {
    "version": "v2Pro",
    "device": "cuda",
    "is_half": True,
    "sovits_weights": str(sovits),
    "gpt_weights": str(gpt),
    "sovits_epoch": se,
    "gpt_epoch": ge,
    "styles": {
        name: {
            "label": name.capitalize(),
            "description": DESC[name],
            "ref_audio": str(ROOT / f"refs/{name}.wav"),
            # The transcript of the reference clip. GPT-SoVITS conditions on
            # this, and delivery transfer is noticeably worse without it.
            "prompt_text": prompts[name],
            "top_k": 15, "top_p": 1.0, "temperature": 1.0,
            "speed_factor": 1.0, "repetition_penalty": 1.35,
        }
        for name in ("calm", "dramatic", "urgent")
    },
}

out = ROOT / "webapp/config.json"
out.write_text(json.dumps(cfg, indent=2))
print(f"wrote {out}")
print(f"  SoVITS: {sovits}  (epoch {se})")
print(f"  GPT   : {gpt}  (epoch {ge})")
for n, s in cfg["styles"].items():
    ok = Path(s["ref_audio"]).exists()
    print(f"  {n:9s} ref={'OK ' if ok else 'MISSING'} {s['ref_audio']}")
