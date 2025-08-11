# -*- coding: utf-8 -*-
import os, argparse, json, torch, torch.nn as nn, csv, unicodedata
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from functools import partial
from datetime import datetime

# ================= 数据集 / 字符集 =================
_DYNAMIC_AVAILABLE = False
try:
    from captcha_ocr.datasets.ctc_dataset import OCRDataset, pad_collate_fn as _dyn_collate
    _DYNAMIC_AVAILABLE = True
except Exception:
    from captcha_ocr.datasets.ctc_dataset import OCRDataset, ocr_collate as _dyn_collate

from captcha_ocr.datasets.ctc_charset import build_charset, save_charset, load_charset

# ================= 模型 / 解码 / 评估 =================
from captcha_ocr.models import get_model
from captcha_ocr.utils.decoding import ctc_greedy_decode
from captcha_ocr.utils.metrics import exact_match


def maybe_compile(model, enable: bool):
    if not enable or not hasattr(torch, "compile"):
        return model
    try:
        import triton  # noqa: F401
        return torch.compile(model, backend="inductor")
    except Exception as e:
        print("⚠️ 未检测到可用 Triton/Inductor，回退 eager：", e)
        return model


# ---------- 从 batch 中提取“文件名 / 文本标签” ----------
_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def _maybe_get_filenames_from_batch(batch):
    for item in batch:
        if isinstance(item, (list, tuple)) and item and all(isinstance(x, str) for x in item):
            if all(x.lower().endswith(_IMG_EXT) for x in item):
                return list(item)
    return None

def _maybe_get_texts_from_batch(batch):
    """
    只在明确存在“非文件名字符串列表”时，才返回文本标签列表；
    否则返回 None（后续用 labels + lengths 回切）。
    """
    for item in batch:
        if isinstance(item, (list, tuple)) and item and all(isinstance(x, str) for x in item):
            if not all(x.lower().endswith(_IMG_EXT) for x in item):
                return list(item)
    return None


# ---------- 将索引序列还原为字符串 ----------
def ids_to_text(ids: torch.Tensor, charset):
    res = []
    L = len(charset)
    for i in ids.tolist():
        if 1 <= i <= L:
            res.append(charset[i - 1])
    return "".join(res)


def _ensure_lengths(label_lengths, texts, device, batch_size, labels=None):
    """
    尝试稳定得到 target lengths:
    1) 若传入的 label_lengths 尺寸正确，直接用；
    2) 若有 texts，用 len(text)；
    3) 否则若有 2D labels（N, Lmax），用非零计数；
    4) 否则报错。
    """
    if isinstance(label_lengths, torch.Tensor):
        if label_lengths.dim() == 1 and label_lengths.numel() == batch_size:
            return label_lengths.to(device, non_blocking=True, dtype=torch.long)
    elif isinstance(label_lengths, (list, tuple)):
        if len(label_lengths) == batch_size and all(isinstance(x, (int,)) for x in label_lengths):
            return torch.tensor(label_lengths, dtype=torch.long, device=device)

    if texts is not None:
        lens = [len(t) for t in texts]
        if len(lens) == batch_size:
            return torch.tensor(lens, dtype=torch.long, device=device)

    if isinstance(labels, torch.Tensor) and labels.dim() == 2 and labels.size(0) == batch_size:
        lens = (labels != 0).sum(dim=1).to(torch.long)
        return lens.to(device, non_blocking=True)

    raise RuntimeError("无法确定 label_lengths；请检查 collate 返回或在数据集中提供 texts。")


# ---------- CER ----------
def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def cer(preds, gts):
    total_d, total_l = 0, 0
    for p, g in zip(preds, gts):
        d = _levenshtein(p, g)
        l = max(1, len(g))
        total_d += d
        total_l += l
    return total_d / max(1, total_l)


# ---------- 文本规范化（宽松匹配） ----------
_MAP = {
    "？": "?",  # 全角问号 -> 半角
    "（": "(", "）": ")",
    "，": ",", "。": ".",
    "：": ":", "；": ";",
    # 乘除号如果你希望严格匹配，请把下面两行改为 "×":"×","÷":"/"
    "×": "x",
    "÷": "/",
}
def norm_text(s: str) -> str:
    s2 = unicodedata.normalize("NFKC", s)
    s2 = "".join(_MAP.get(ch, ch) for ch in s2)
    s2 = s2.replace("\u200b", "").replace("\ufeff", "")
    return s2


# ---------- 报表输出 ----------
def save_samples_tsv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)

def append_metrics_csv(path, epoch, acc, cer_value, acc_norm=None, note=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["time", "epoch", "acc(EM)", "acc_norm(NFKC)", "cer", "note"])
        from datetime import datetime
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch, f"{acc:.6f}", ("" if acc_norm is None else f"{acc_norm:.6f}"),
                    f"{cer_value:.6f}", note])


# ================== 单模型训练 ==================
def train_one(args, arch: str):
    # ---- charset ----
    tr_lbl=os.path.join(args.train_root,"labels.txt"); va_lbl=os.path.join(args.val_root,"labels.txt")
    if not os.path.isfile("charset.txt"):
        cs=build_charset(tr_lbl,va_lbl); save_charset(cs,"charset.txt"); print("已生成 charset.txt，大小=",len(cs))
    charset,_=load_charset("charset.txt")
    num_classes=len(charset)+1  # 字符 1..N，blank=0

    # ---- 输入通道：vanilla=1，其它=3 ----
    in_ch = 1 if arch == "vanilla_crnn" else 3

    # ---- 数据集 ----
    train_set=OCRDataset(args.train_root,args.imgH,args.imgW,
                         augment=args.aug,charset_path="charset.txt",
                         channels=in_ch)
    val_set  =OCRDataset(args.val_root,  args.imgH,args.imgW,
                         augment=False,   charset_path="charset.txt",
                         channels=in_ch)

    # ---- collate ----
    if _DYNAMIC_AVAILABLE:
        collate_train = partial(_dyn_collate, pad_to_multiple=args.pad_multiple, bg_value=1.0)
        collate_val   = partial(_dyn_collate, pad_to_multiple=args.pad_multiple, bg_value=1.0)
    else:
        collate_train = _dyn_collate
        collate_val   = _dyn_collate

    # ---- DataLoader ----
    train_loader=torch.utils.data.DataLoader(
        train_set,batch_size=args.batch,shuffle=True,num_workers=args.num_workers,
        pin_memory=True,collate_fn=collate_train,drop_last=True,persistent_workers=False
    )
    val_loader=torch.utils.data.DataLoader(
        val_set,batch_size=max(64,args.batch//2),shuffle=False,
        num_workers=max(0,args.num_workers-1),pin_memory=True,collate_fn=collate_val,
        drop_last=False,persistent_workers=False
    )

    # ---- 模型/优化 ----
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=get_model(arch, num_classes=num_classes).to(device)
    model=maybe_compile(model,args.compile)
    print(f"模型: {arch} | 参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | 输入通道: {in_ch}")

    criterion=nn.CTCLoss(blank=0, zero_infinity=True)
    opt=AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); sch=CosineAnnealingLR(opt, T_max=args.epochs)
    scaler=GradScaler('cuda', enabled=(device.type=='cuda'))

    ckpt_dir=os.path.join("checkpoints",arch); os.makedirs(ckpt_dir,exist_ok=True)
    with open(os.path.join(ckpt_dir,"config.txt"),"w",encoding="utf-8") as f: json.dump({**vars(args),"arch":arch},f,ensure_ascii=False,indent=2)

    # ---- 恢复 ----
    if args.resume:
        last_p = os.path.join(ckpt_dir,"last.pth")
        if os.path.isfile(last_p):
            data = torch.load(last_p, map_location=device)
            try:
                model.load_state_dict(data["model"], strict=False)
                print(f"▶️ 已恢复 {arch} 的权重: {last_p}")
            except Exception as e:
                print("⚠️ 恢复失败，跳过：", e)

    best_acc=0.0
    for epoch in range(1,args.epochs+1):
        # ======= 训练 =======
        model.train(); pbar=tqdm(train_loader,desc=f"[{arch}] Epoch {epoch}/{args.epochs}"); total_loss=0.0
        for batch in pbar:
            images = batch[0]
            labels = batch[1]
            raw_label_lengths = batch[2]

            images=images.to(device,non_blocking=True); labels=labels.to(device)
            N = images.size(0)

            with autocast('cuda', enabled=(device.type=='cuda')):
                logits=model(images)
                # 统一为 [T,N,C]
                if logits.dim()==3 and logits.shape[0] == N:
                    logits = logits.permute(1,0,2).contiguous()
                T,N2,C=logits.shape
                if N2 != N:
                    raise RuntimeError(f"模型输出 batch 维与输入不一致: {N2} vs {N}")

                log_probs=logits.log_softmax(2)
                # 训练阶段优先用 collate 给的长度；不行则用非零计数
                try:
                    label_lengths = _ensure_lengths(raw_label_lengths, None, device=device, batch_size=N, labels=labels)
                except Exception:
                    label_lengths = (labels!=0).sum(1).to(torch.long).to(device)

                input_lengths=torch.full((N,),T,dtype=torch.long,device=device)
                loss=criterion(log_probs,labels,input_lengths,label_lengths)

            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            total_loss+=float(loss); pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}")
        sch.step()

        # ======= 验证 =======
        model.eval()
        with torch.no_grad(), autocast('cuda', enabled=(device.type=='cuda')):
            all_pred,all_gt=[],[]
            all_pred_norm, all_gt_norm = [], []
            sample_rows=[]
            printed = 0

            for batch in val_loader:
                images = batch[0]
                labels = batch[1]
                raw_label_lengths = batch[2]
                texts = _maybe_get_texts_from_batch(batch)   # 只在确实存在文本列表时返回；否则 None
                fns   = _maybe_get_filenames_from_batch(batch)

                images=images.to(device,non_blocking=True)
                logits=model(images)
                if logits.dim()==3 and logits.shape[0] == images.size(0):
                    logits = logits.permute(1,0,2).contiguous()

                pred_texts=ctc_greedy_decode(logits,charset)

                # ---- 还原 GT ----
                if texts is not None and len(texts)==len(pred_texts):
                    gts = list(texts)
                else:
                    N = images.size(0)
                    # 先确定每条样本的长度
                    lens = None
                    try:
                        lens = _ensure_lengths(raw_label_lengths, None, device="cpu", batch_size=N, labels=labels)
                    except Exception:
                        pass

                    gts=[]
                    if isinstance(labels, torch.Tensor) and labels.dim()==2 and labels.size(0)==N:
                        # (N, Lmax) -> 每行非零即字符
                        lab2 = labels.cpu()
                        for i in range(N):
                            ln = int(lab2[i].ne(0).sum().item()) if lens is None else int(lens[i].item())
                            ids = lab2[i, :ln]
                            gts.append(ids_to_text(ids, charset))
                    else:
                        # 扁平情况：必须有 lens
                        if lens is None:
                            raise RuntimeError("验证集未提供 texts，且 labels 不是 2D，无法根据长度切分 GT。")
                        lab_flat = labels.cpu()
                        start=0
                        for ln_t in lens.tolist():
                            ln=int(ln_t)
                            ids = lab_flat[start:start+ln]
                            start+=ln
                            gts.append(ids_to_text(ids, charset))

                # ---- 收集指标 ----
                all_pred+=pred_texts; all_gt+=gts
                all_pred_norm += [norm_text(p) for p in pred_texts]
                all_gt_norm   += [norm_text(g) for g in gts]

                # ---- 样例（前 N 条）----
                if printed < args.print_samples:
                    for i,(gt,pd) in enumerate(zip(gts, pred_texts)):
                        ok = "1" if gt == pd else "0"
                        okn = "1" if norm_text(gt)==norm_text(pd) else "0"
                        fn = fns[i] if (fns is not None and i < len(fns)) else ""
                        sample_rows.append([fn, gt, pd, ok, okn])
                        printed += 1
                        if printed >= args.print_samples: break

            em_acc=exact_match(all_pred,all_gt)
            em_acc_norm=exact_match(all_pred_norm, all_gt_norm)
            cer_val=cer(all_pred,all_gt)

        print(f"[Val-{arch}] acc={em_acc*100:.2f}% | EM={em_acc*100:.2f}% | EM_norm={em_acc_norm*100:.2f}% | CER={cer_val*100:.2f}% | 样例（前{args.print_samples}条）已写出")

        # 样例 TSV
        save_samples_tsv(
            os.path.join(ckpt_dir, f"val_samples_epoch{epoch}.tsv"),
            sample_rows,
            header=["filename","gt","pred","ok","ok_norm"]
        )
        # 指标日志 CSV
        append_metrics_csv(os.path.join(ckpt_dir, "metrics_log.csv"), epoch, em_acc, cer_val, acc_norm=em_acc_norm)

        # 权重
        torch.save({"model":model.state_dict(),"charset":charset,"arch":arch}, os.path.join(ckpt_dir,"last.pth"))
        if em_acc>best_acc:
            best_acc=em_acc
            torch.save({"model":model.state_dict(),"charset":charset,"arch":arch}, os.path.join(ckpt_dir,"best.pth"))
            print(f"✅ 保存最优模型: {arch} acc={best_acc*100:.2f}% -> {ckpt_dir}/best.pth")

    print(f"训练完成。最佳 acc={best_acc*100:.2f}% | 模型={arch} | 权重在 {ckpt_dir}")


# ================== 主入口（含连续训练） ==================
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["vanilla_crnn","crnn_resnet34","svtr_tiny","all"], default="vanilla_crnn")
    ap.add_argument("--train-root", default="dataset/train"); ap.add_argument("--val-root", default="dataset/test")
    ap.add_argument("--imgH", type=int, default=32); ap.add_argument("--imgW", type=int, default=160)
    ap.add_argument("--pad-multiple", type=int, default=4)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--batch", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--aug", action="store_true")
    ap.add_argument("--compile", action="store_true"); ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--print-samples", type=int, default=10)
    args=ap.parse_args(); torch.backends.cudnn.benchmark=True

    if args.arch != "all":
        train_one(args, args.arch)
    else:
        plan = [
            ("vanilla_crnn",  args.imgH, 256, args.batch),
            ("crnn_resnet34", args.imgH, 160, max(64, args.batch//2)),
            ("svtr_tiny",     args.imgH, 160, max(64, args.batch//2)),
        ]
        for arch, h, w, b in plan:
            print("\n" + "="*80)
            print(f"▶️ 连续训练：{arch}，imgH={h}, imgW={w}, batch={b}")
            print("="*80 + "\n")
            sub = argparse.Namespace(**vars(args))
            sub.arch = arch; sub.imgH=h; sub.imgW=w; sub.batch=b
            train_one(sub, arch)


if __name__=='__main__':
    main()
