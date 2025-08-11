# -*- coding: utf-8 -*-
import os, argparse, glob, cv2, numpy as np, pandas as pd, torch
from captcha_ocr.models import get_model
from captcha_ocr.utils.decoding import ctc_greedy_decode
SUFFIXES=("*.png","*.jpg","*.jpeg","*.bmp","*.gif","*.webp")
def load_image_gray(path, H=32, W=160):
    img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(path)
    h,w=img.shape; newH=H; newW=int(w*(newH/h))
    img=cv2.resize(img,(newW,newH),interpolation=cv2.INTER_LINEAR)
    if newW<W:
        pad=np.ones((newH,W-newW),dtype=img.dtype)*255; img=np.concatenate([img,pad],axis=1)
    else: img=img[:, :W]
    img=img.astype(np.float32)/255.0; img=(img-0.5)/0.5; img=np.expand_dims(img,0)
    return torch.from_numpy(img)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["vanilla_crnn","crnn_resnet34","svtr_tiny"], default="vanilla_crnn")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--image", default=None); ap.add_argument("--dir", default=None)
    ap.add_argument("--imgH", type=int, default=32); ap.add_argument("--imgW", type=int, default=160)
    args=ap.parse_args()
    ckpt=args.ckpt or os.path.join("checkpoints",args.arch,"best.pth"); assert os.path.isfile(ckpt), f"未找到权重: {ckpt}"
    data=torch.load(ckpt, map_location="cpu"); charset=data.get("charset"); assert isinstance(charset,str)
    model=get_model(args.arch, num_classes=len(charset)); model.load_state_dict(data["model"], strict=True); model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    files=[]; 
    if args.image: files=[args.image]
    if args.dir:
        for suf in SUFFIXES: files+=glob.glob(os.path.join(args.dir,"**",suf), recursive=True)
    assert files, "请通过 --image 或 --dir 指定输入"; files=sorted(set(files))
    rows=[]; 
    with torch.no_grad():
        for path in files:
            img=load_image_gray(path,args.imgH,args.imgW).unsqueeze(0).to(device); logits=model(img)
            texts=ctc_greedy_decode(logits,charset); rows.append((path,texts[0])); print(f"{path} -> {texts[0]}")
    pd.DataFrame(rows, columns=["file","pred"]).to_csv("preds.csv", index=False, encoding="utf-8-sig"); print("✅ 已写出 preds.csv")
if __name__=='__main__': main()
