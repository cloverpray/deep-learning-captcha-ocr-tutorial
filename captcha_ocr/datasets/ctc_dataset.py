# -*- coding: utf-8 -*-
import math
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
except Exception:
    A = None


class Charset:
    def __init__(self, path: str):
        content = Path(path).read_text(encoding="utf-8").strip()
        self.sym_list = list(content)
        self.sym2id = {ch: i + 1 for i, ch in enumerate(self.sym_list)}
        self.blank_id = 0
    def size_with_blank(self) -> int: return len(self.sym_list) + 1
    def encode(self, s: str) -> List[int]: return [self.sym2id[ch] for ch in s if ch in self.sym2id]
    def decode_ids(self, ids: List[int]) -> str:
        return "".join(self.sym_list[i - 1] for i in ids if 1 <= i <= len(self.sym_list))


def _build_aug():
    if A is None:
        return None
    tfms = [
        A.MotionBlur(blur_limit=3, p=0.12),
        A.GaussianBlur(blur_limit=(3, 5), p=0.12),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.15),
    ]
    # 兼容不同版本的 JPEG/图像压缩 API
    added = False
    for cls_name, kwargs in [
        ("JpegCompression", dict(quality_lower=60, quality_upper=95)),
        ("ImageCompression", dict(quality_lower=60, quality_upper=95)),
        ("JpegCompression", dict(quality=(60, 95))),
        ("ImageCompression", dict(quality=(60, 95))),
    ]:
        try:
            comp_cls = getattr(A, cls_name)
            tfms.append(comp_cls(p=0.15, **kwargs))
            added = True
            break
        except Exception:
            continue
    return A.Compose(tfms)


class OCRDataset(Dataset):
    """
    root/
      images/*.png|jpg
      labels.txt  (filename \t label)
    定高等比缩放，宽度<=maxW；不在此处做 pad。
    """
    def __init__(self, root: str, imgH: int = 32, maxW: int = 256,
                 augment: bool = False, charset_path: str = "charset.txt",
                 bg_value: int = 255, channels: int = 1):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        lbl = self.root / "labels.txt"
        if not lbl.exists():
            raise FileNotFoundError(f"{lbl} 不存在")
        self.samples: List[Tuple[str, str]] = []
        for ln in lbl.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln: continue
            if "\t" in ln: fn, lab = ln.split("\t", 1)
            else:
                parts = ln.split()
                if len(parts) < 2: continue
                fn, lab = parts[0], " ".join(parts[1:])
            self.samples.append((fn, lab))

        proj_charset = (self.root.parent / "charset.txt")
        cs_path = proj_charset if proj_charset.exists() else Path(charset_path)
        if not cs_path.exists():
            raise FileNotFoundError(f"未找到 charset.txt：{cs_path.resolve()}")
        self.charset = Charset(str(cs_path))

        self.imgH = int(imgH)
        self.maxW = int(maxW)
        self.aug = _build_aug() if augment else None
        self.bg_value = int(bg_value)
        self.channels = 1 if int(channels) == 1 else 3  # 只支持 1 或 3

    def __len__(self): return len(self.samples)

    @staticmethod
    def _resize_keepH(img: Image.Image, H: int, maxW: int) -> Image.Image:
        w, h = img.size
        if h != H:
            scale = H / float(h)
            new_w = max(1, int(round(w * scale)))
        else:
            new_w = w
        if new_w > maxW: new_w = maxW
        if (new_w, H) != (w, h):
            img = img.resize((new_w, H), Image.BICUBIC)
        return img

    def __getitem__(self, idx):
        fn, label = self.samples[idx]
        path = self.img_dir / fn
        # 按需灰度或 RGB
        img = Image.open(path).convert("L" if self.channels == 1 else "RGB")
        img = self._resize_keepH(img, self.imgH, self.maxW)

        if self.aug is not None:
            arr = np.array(img)
            arr = self.aug(image=arr)["image"]
        else:
            arr = np.array(img)

        arr = arr.astype(np.float32) / 255.0
        if self.channels == 1:
            if arr.ndim == 2:
                arr = arr[None, :, :]  # 1,H,W
            else:
                arr = arr[..., 0][None, :, :]
        else:
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            arr = arr.transpose(2, 0, 1)  # 3,H,W

        tensor = torch.from_numpy(arr)
        ids = torch.tensor(self.charset.encode(label), dtype=torch.long)

        return {
            "image": tensor,
            "label_text": label,
            "label_ids": ids,
            "pix_w": tensor.shape[-1],
            "fname": fn
        }


def pad_collate_fn(batch, pad_to_multiple: int = 4, bg_value: float = 1.0):
    C = batch[0]["image"].shape[0]; H = batch[0]["image"].shape[1]
    widths = [b["image"].shape[-1] for b in batch]
    maxW = max(widths)
    if pad_to_multiple > 1:
        maxW = int((maxW + pad_to_multiple - 1) // pad_to_multiple * pad_to_multiple)

    images, targets, target_lengths, pixw, fnames = [], [], [], [], []
    for b in batch:
        x = b["image"]; pad_w = maxW - x.shape[-1]
        if pad_w > 0:
            pad = torch.ones((C, H, pad_w), dtype=x.dtype) * bg_value
            x = torch.cat([x, pad], dim=-1)
        images.append(x)
        targets.append(b["label_ids"])
        target_lengths.append(b["label_ids"].numel())
        pixw.append(b["pix_w"]); fnames.append(b["fname"])

    images = torch.stack(images, dim=0)
    targets = torch.cat(targets, dim=0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    pixw = torch.tensor(pixw, dtype=torch.long)
    return images, targets, target_lengths, pixw, fnames


def ocr_collate(batch):
    C = batch[0]["image"].shape[0]; H = batch[0]["image"].shape[1]
    widths = [b["image"].shape[-1] for b in batch]
    maxW = max(widths)

    images, labels, label_lengths, fns, texts = [], [], [], [], []
    for b in batch:
        x = b["image"]; pad_w = maxW - x.shape[-1]
        if pad_w > 0:
            pad = torch.ones((C, H, pad_w), dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        images.append(x)
        labels.append(b["label_ids"])
        label_lengths.append(b["label_ids"].numel())
        fns.append(b["fname"]); texts.append(b["label_text"])

    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return images, labels, label_lengths, fns, texts
