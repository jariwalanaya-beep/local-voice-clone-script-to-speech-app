#!/usr/bin/env python
"""
Headless GPT-SoVITS pipeline: ASR -> preprocess (1a/1b/1c) -> train SoVITS -> train GPT.

This mirrors exactly what webui.py's open1abc / open1Ba / open1Bb do, but runs
without the Gradio UI so it can be logged and reproduced.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

GSV = Path("/home/shivam/va/GPT-SoVITS")
ROOT = Path("/home/shivam/va")
os.chdir(GSV)
PY = sys.executable

# The prepare_datasets/ and training scripts import both `tools.*` (repo root)
# and `text.*` (the GPT_SoVITS package dir), so both must be importable in
# every subprocess we spawn.
os.environ["PYTHONPATH"] = os.pathsep.join([str(GSV), str(GSV / "GPT_SoVITS")])

EXP = "myvoice"
VERSION = "v2Pro"
EXP_ROOT = "logs"
OPT_DIR = f"{EXP_ROOT}/{EXP}"

INP_TEXT = str(ROOT / "data/logs/myvoice.list")
INP_WAV  = str(ROOT / "data/myvoice")

BERT   = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
HUBERT = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
SV     = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
S2G    = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
S2D    = "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth"
S1     = "GPT_SoVITS/pretrained_models/s1v3.ckpt"

# ---- conservative settings for ~5 minutes of data (project few-shot defaults)
SOVITS_EPOCHS, SOVITS_BATCH, SOVITS_SAVE_EVERY = 8, 6, 4
GPT_EPOCHS,    GPT_BATCH,    GPT_SAVE_EVERY    = 15, 6, 5
TEXT_LOW_LR = 0.4

os.makedirs(OPT_DIR, exist_ok=True)
os.makedirs("TEMP", exist_ok=True)


def run(cmd, env_extra=None, label=""):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in (env_extra or {}).items()})
    print(f"\n{'='*70}\n[{label}] {cmd}\n{'='*70}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, env=env)
    print(f"[{label}] exit={p.returncode} in {time.time()-t0:.1f}s", flush=True)
    if p.returncode != 0:
        sys.exit(f"FAILED at {label}")


def vram(tag):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True).strip()
        used, tot = out.split(", ")
        print(f"[VRAM/{tag}] {used} MiB / {tot} MiB", flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------- 1a: text+bert
path_text = f"{OPT_DIR}/2-name2text.txt"
if not os.path.exists(path_text) or len(open(path_text, encoding="utf8").read().strip().split("\n")) < 2:
    run(f'"{PY}" -s GPT_SoVITS/prepare_datasets/1-get-text.py', {
        "inp_text": INP_TEXT, "inp_wav_dir": INP_WAV, "exp_name": EXP,
        "opt_dir": OPT_DIR, "bert_pretrained_dir": BERT, "is_half": "True",
        "i_part": "0", "all_parts": "1", "_CUDA_VISIBLE_DEVICES": "0",
    }, "1a get-text")
    part = f"{OPT_DIR}/2-name2text-0.txt"
    opt = open(part, encoding="utf8").read().strip("\n").split("\n")
    os.remove(part)
    open(path_text, "w", encoding="utf8").write("\n".join(opt) + "\n")
    assert len("".join(opt)) > 0, "1a produced no text"
print(f"[1a] DONE -> {path_text} ({len(open(path_text,encoding='utf8').readlines())} lines)")
vram("after-1a")

# ------------------------------------------------------- 1b: hubert + sv feats
run(f'"{PY}" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py', {
    "inp_text": INP_TEXT, "inp_wav_dir": INP_WAV, "exp_name": EXP,
    "opt_dir": OPT_DIR, "cnhubert_base_dir": HUBERT, "sv_path": SV,
    "i_part": "0", "all_parts": "1", "_CUDA_VISIBLE_DEVICES": "0",
}, "1b get-hubert")
run(f'"{PY}" -s GPT_SoVITS/prepare_datasets/2-get-sv.py', {
    "inp_text": INP_TEXT, "inp_wav_dir": INP_WAV, "exp_name": EXP,
    "opt_dir": OPT_DIR, "cnhubert_base_dir": HUBERT, "sv_path": SV,
    "i_part": "0", "all_parts": "1", "_CUDA_VISIBLE_DEVICES": "0",
}, "1b get-sv")
print("[1b] DONE")
vram("after-1b")

# ------------------------------------------------------------ 1c: semantic
path_sem = f"{OPT_DIR}/6-name2semantic.tsv"
if not os.path.exists(path_sem) or os.path.getsize(path_sem) < 31:
    run(f'"{PY}" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py', {
        "inp_text": INP_TEXT, "exp_name": EXP, "opt_dir": OPT_DIR,
        "pretrained_s2G": S2G, "s2config_path": f"GPT_SoVITS/configs/s2{VERSION}.json",
        "i_part": "0", "all_parts": "1", "_CUDA_VISIBLE_DEVICES": "0",
    }, "1c get-semantic")
    part = f"{OPT_DIR}/6-name2semantic-0.tsv"
    opt = ["item_name\tsemantic_audio"] + open(part, encoding="utf8").read().strip("\n").split("\n")
    os.remove(part)
    open(path_sem, "w", encoding="utf8").write("\n".join(opt) + "\n")
print(f"[1c] DONE -> {path_sem} ({len(open(path_sem,encoding='utf8').readlines())} lines)")
vram("after-1c")

# ----------------------------------------------------------- stage 1: SoVITS
cfg = json.load(open(f"GPT_SoVITS/configs/s2{VERSION}.json"))
cfg["train"].update({
    "batch_size": SOVITS_BATCH, "epochs": SOVITS_EPOCHS,
    "text_low_lr_rate": TEXT_LOW_LR, "pretrained_s2G": S2G, "pretrained_s2D": S2D,
    "if_save_latest": True, "if_save_every_weights": True,
    "save_every_epoch": SOVITS_SAVE_EVERY, "gpu_numbers": "0",
    "grad_ckpt": False, "lora_rank": 32, "fp16_run": True,
})
cfg["model"]["version"] = VERSION
cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = OPT_DIR
cfg["save_weight_dir"] = "SoVITS_weights_v2Pro"
cfg["name"] = EXP
cfg["version"] = VERSION
os.makedirs(f"{OPT_DIR}/logs_s2_{VERSION}", exist_ok=True)
os.makedirs("SoVITS_weights_v2Pro", exist_ok=True)
json.dump(cfg, open("TEMP/tmp_s2.json", "w"))
print(f"\n[SoVITS] epochs={SOVITS_EPOCHS} batch={SOVITS_BATCH} save_every={SOVITS_SAVE_EVERY}")
run(f'"{PY}" -s GPT_SoVITS/s2_train.py --config "TEMP/tmp_s2.json"', None, "train-SoVITS")
vram("after-SoVITS")

# -------------------------------------------------------------- stage 2: GPT
import yaml
d = yaml.safe_load(open("GPT_SoVITS/configs/s1longer-v2.yaml"))
d["train"].update({
    "batch_size": GPT_BATCH, "epochs": GPT_EPOCHS,
    "save_every_n_epoch": GPT_SAVE_EVERY, "if_save_every_weights": True,
    "if_save_latest": True, "if_dpo": False,
    "half_weights_save_dir": "GPT_weights_v2Pro", "exp_name": EXP,
    "precision": "16-mixed",
})
d["pretrained_s1"] = S1
d["train_semantic_path"] = f"{OPT_DIR}/6-name2semantic.tsv"
d["train_phoneme_path"] = f"{OPT_DIR}/2-name2text.txt"
d["output_dir"] = f"{OPT_DIR}/logs_s1_{VERSION}"
os.makedirs(f"{OPT_DIR}/logs_s1", exist_ok=True)
os.makedirs("GPT_weights_v2Pro", exist_ok=True)
yaml.dump(d, open("TEMP/tmp_s1.yaml", "w"), default_flow_style=False)
print(f"\n[GPT] epochs={GPT_EPOCHS} batch={GPT_BATCH} save_every={GPT_SAVE_EVERY}")
run(f'"{PY}" -s GPT_SoVITS/s1_train.py --config_file "TEMP/tmp_s1.yaml"',
    {"_CUDA_VISIBLE_DEVICES": "0", "hz": "25hz"}, "train-GPT")
vram("after-GPT")

print("\n=== TRAINED WEIGHTS ===")
for root in ("SoVITS_weights_v2Pro", "GPT_weights_v2Pro"):
    for f in sorted(Path(root).glob("*")):
        print(f"  {f}  ({f.stat().st_size/1048576:.1f} MB)")
print("=== PIPELINE COMPLETE ===")
