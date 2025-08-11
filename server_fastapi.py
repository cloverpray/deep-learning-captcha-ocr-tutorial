# -*- coding: utf-8 -*-
import os, io, re
from typing import List
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware

from captcha_ocr.models import get_model
from captcha_ocr.utils.decoding import ctc_greedy_decode

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

CHECKPOINTS_DIR = "checkpoints"
DEFAULT_MODELS = ["vanilla_crnn", "crnn_resnet34", "svtr_tiny"]

def list_available_models() -> List[str]:
    out = []
    for a in DEFAULT_MODELS:
        if os.path.isfile(os.path.join(CHECKPOINTS_DIR, a, "best.pth")):
            out.append(a)
    return out

# -------------------- SVTR 键名双向自适配 --------------------
P_STEM_BARE   = re.compile(r"(^|^)stem\.")
P_ENCOD_BARE  = re.compile(r"(^|^)encoder\.")
P_POS_BARE    = re.compile(r"(^|^)pos\.")

P_STEM_ENC    = re.compile(r"(^|^)enc\.stem\.")
P_ENCOD_ENC   = re.compile(r"(^|^)enc\.encoder\.")
P_POS_ENC     = re.compile(r"(^|^)enc\.pos\.")

def _has(keys, pat: re.Pattern) -> bool:
    return any(pat.search(k) for k in keys)

def _add_enc_prefix(k: str) -> str:
    k = re.sub(r"(^|^)stem\.",    "enc.stem.",    k)
    k = re.sub(r"(^|^)encoder\.", "enc.encoder.", k)
    k = re.sub(r"(^|^)pos\.",     "enc.pos.",     k)
    return k

def _drop_enc_prefix(k: str) -> str:
    k = re.sub(r"(^|^)enc\.stem\.",    "stem.",    k)
    k = re.sub(r"(^|^)enc\.encoder\.", "encoder.", k)
    k = re.sub(r"(^|^)enc\.pos\.",     "pos.",     k)
    return k

def _proj_to_idx(k: str) -> str:
    # ...proj.* -> ...0.* ; ...bn.* -> ...1.*
    k = re.sub(r"(stem\.\d+)\.proj\.", r"\1.0.", k)
    k = re.sub(r"(stem\.\d+)\.bn\.",   r"\1.1.", k)
    k = re.sub(r"(enc\.stem\.\d+)\.proj\.", r"\1.0.", k)
    k = re.sub(r"(enc\.stem\.\d+)\.bn\.",   r"\1.1.", k)
    return k

def _idx_to_proj(k: str) -> str:
    # ...0.* -> ...proj.* ; ...1.* -> ...bn.*
    k = re.sub(r"(stem\.\d+)\.0\.", r"\1.proj.", k)
    k = re.sub(r"(stem\.\d+)\.1\.", r"\1.bn.",   k)
    k = re.sub(r"(enc\.stem\.\d+)\.0\.", r"\1.proj.", k)
    k = re.sub(r"(enc\.stem\.\d+)\.1\.", r"\1.bn.",   k)
    return k

def _model_wants_enc(keys) -> bool:
    return _has(keys, P_STEM_ENC) or _has(keys, P_ENCOD_ENC) or _has(keys, P_POS_ENC)

def _sd_has_enc(keys) -> bool:
    return _has(keys, P_STEM_ENC) or _has(keys, P_ENCOD_ENC) or _has(keys, P_POS_ENC)

def _model_uses_proj(keys) -> bool:
    # 只要在 stem.* 中出现 .proj 或 .bn 即视为 proj/bn 风格
    return any(".proj." in k or ".bn." in k for k in keys if ".stem." in k)

def _sd_uses_proj(keys) -> bool:
    return any(".proj." in k or ".bn." in k for k in keys if ".stem." in k)

def _model_uses_idx(keys) -> bool:
    return bool([k for k in keys if re.search(r"stem\.\d+\.(0|1)\.", k)])

def _sd_uses_idx(keys) -> bool:
    return bool([k for k in keys if re.search(r"stem\.\d+\.(0|1)\.", k)])

def _remap_svtr_to_model(sd: dict, model: nn.Module) -> dict:
    mk = list(model.state_dict().keys())
    sk = list(sd.keys())

    # 先处理 enc 前缀方向
    want_enc  = _model_wants_enc(mk)
    sd_is_enc = _sd_has_enc(sk)

    def _apply_enc(k: str) -> str:
        if want_enc and not sd_is_enc:
            return _add_enc_prefix(k)
        if (not want_enc) and sd_is_enc:
            return _drop_enc_prefix(k)
        return k

    # 再处理 proj/bn 与 0/1 的方向
    model_proj = _model_uses_proj(mk)
    model_idx  = _model_uses_idx(mk)
    sd_proj    = _sd_uses_proj(sk)
    sd_idx     = _sd_uses_idx(sk)

    def _apply_proj_idx(k: str) -> str:
        if model_proj and sd_idx:
            return _idx_to_proj(k)
        if model_idx and sd_proj:
            return _proj_to_idx(k)
        return k

    out = {}
    for k, v in sd.items():
        k1 = _apply_enc(k)
        k2 = _apply_proj_idx(k1)
        out[k2] = v
    return out

# -------------------- 构建 & 加载 --------------------
def _build_model(arch: str, num_classes: int) -> nn.Module:
    return get_model(arch, num_classes=num_classes).eval()

def _safe_load_state_dict(model: nn.Module, arch: str, data: dict):
    sd = data.get("model", data)
    if arch == "svtr_tiny":
        sd = _remap_svtr_to_model(sd, model)
        # fc 宽度自适配
        if "fc.weight" in sd and hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            in_ckpt = sd["fc.weight"].shape[1]
            if in_ckpt != model.fc.in_features:
                model.fc = nn.Linear(in_ckpt, model.fc.out_features, bias=True)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"⚠️ state_dict 差异 | missing={missing} | unexpected={unexpected}")
    return model

def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # 新版
    except TypeError:
        return torch.load(path, map_location="cpu")                     # 旧版兼容

def _load_best(arch: str, ckpt_name: str = "best.pth"):
    p = os.path.join(CHECKPOINTS_DIR, arch, ckpt_name)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"未找到权重: {p}")
    data = _torch_load(p)
    charset = data.get("charset")
    if not charset:
        raise RuntimeError("ckpt 未包含 charset，请用当前训练脚本导出的权重。")
    num_classes = len(charset) + 1
    model = _build_model(arch, num_classes=num_classes)
    model = _safe_load_state_dict(model, arch, data).eval()
    return model, charset

# -------------------- 前处理（与训练对齐） --------------------
def _infer_channels(arch: str) -> int:
    return 1 if arch == "vanilla_crnn" else 3

PAD_MULT = {"vanilla_crnn": 4, "crnn_resnet34": 4, "svtr_tiny": 4}

def _pad_width_to_multiple(arr: np.ndarray, multiple: int) -> np.ndarray:
    C,H,W = arr.shape
    r = W % multiple
    if r == 0:
        return arr
    need = multiple - r
    pad = np.repeat(arr[:, :, -1:], need, axis=2)
    return np.concatenate([arr, pad], axis=2)

def _ensure_mode(img: Image.Image, ch: int) -> Image.Image:
    if ch == 1:
        return img.convert("L") if img.mode != "L" else img
    return img.convert("RGB") if img.mode != "RGB" else img

def _prepare_tensor(img: Image.Image, need_ch: int, arch: str, imgH: int = 0, imgW: int = 0) -> torch.Tensor:
    if imgH and imgW:
        img = img.resize((int(imgW), int(imgH)), Image.BILINEAR)
    img = _ensure_mode(img, need_ch)
    arr = np.array(img).astype("float32") / 255.0
    if need_ch == 1:
        arr = arr[None, ...]
    else:
        arr = arr.transpose(2,0,1)
    arr = _pad_width_to_multiple(arr, PAD_MULT.get(arch, 4))
    return torch.from_numpy(arr).unsqueeze(0)  # [1,C,H,W]

def _to_TNC(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3:
        # 这里既接受 [T,N,C] 也接受 [N,T,C]
        if logits.shape[0] != logits.shape[1] and logits.shape[0] == 1:
            return logits.permute(1,0,2).contiguous()
        return logits
    elif logits.dim() == 2:
        return logits.unsqueeze(0).unsqueeze(1)
    else:
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

# -------------------- API --------------------
@app.get("/models")
def models_api():
    return JSONResponse(list_available_models())

@app.post("/predict")
async def predict_api(
    files: List[UploadFile] = File(...),
    arch: str = Form("svtr_tiny"),
    imgH: int = Form(0),
    imgW: int = Form(0),
):
    try:
        model, charset = _load_best(arch)
    except Exception as e:
        return JSONResponse({"error": f"加载模型失败: {e}"}, status_code=400)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    ch = _infer_channels(arch)

    results = []
    for f in files:
        try:
            raw = await f.read()
            img = Image.open(io.BytesIO(raw))
            x = _prepare_tensor(img, ch, arch, imgH, imgW).to(device)
            with torch.inference_mode():
                logits = model(x)
            logits = _to_TNC(logits)
            pred = ctc_greedy_decode(logits, charset)[0]
            results.append({"filename": f.filename, "pred": pred, "T": int(logits.shape[0])})
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return JSONResponse({"arch": arch, "count": len(results), "results": results})

@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><meta charset="utf-8"/>
<div style="font-family:system-ui;padding:20px;max-width:900px">
<h3>验证码识别（FastAPI）</h3>
<div>可用模型：<code id="models"></code></div>
<form id="f" method="post" enctype="multipart/form-data" action="/predict" style="margin-top:10px">
  <div>模型 <select name="arch" id="arch"></select></div>
  <div>H(0原图)<input name="imgH" value="64" size="4"/> W(0原图)<input name="imgW" value="180" size="4"/></div>
  <div><input type="file" name="files" multiple accept="image/*"/></div>
  <button>识别</button>
</form>
<pre id="out" style="white-space:pre-wrap;background:#f7f7f7;padding:10px;margin-top:10px"></pre>
<script>
fetch('/models').then(r=>r.json()).then(ms=>{
  document.getElementById('models').textContent=(ms||[]).join(', ');
  const sel=document.getElementById('arch'); (ms||[]).forEach(m=>{const o=document.createElement('option');o.value=m;o.textContent=m;sel.appendChild(o);});
  if((ms||[]).includes('svtr_tiny')) sel.value='svtr_tiny';
});
document.getElementById('f').addEventListener('submit', async ev=>{
  ev.preventDefault(); const fd=new FormData(ev.target);
  const r=await fetch('/predict',{method:'POST',body:fd}); document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2);
});
</script>
</div>"""
# 运行： uvicorn server_fastapi:app --host 0.0.0.0 --port 8000

