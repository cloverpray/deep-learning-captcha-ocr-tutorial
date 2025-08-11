# -*- coding: utf-8 -*-
"""
ctc_charset.py
- 生成/保存/加载字符集（不包含 CTC blank；CTC 的 blank 固定用 0）
- 读取 labels.txt（每行: 文件名 <TAB> 标签）
兼容你的训练脚本: build_charset / save_charset / load_charset
"""

from pathlib import Path
from typing import List, Tuple

# 统一的符号优先级排序（若在数据中出现就按此顺序放入 charset）
_SYMBOLS_ORDER = [
    # 基本运算
    "+", "-", "*", "/", "=", "×", "÷",
    # 问号/括号/常见符号
    "?", "？", "(", ")", "[", "]", "{", "}", "#", "@", "$", "%", "&",
    ".", "_", ":", ",", ";", "^", "~", "<", ">", "\\",
    # 可能出现的全角/变体
    "＋", "－", "＊", "／", "＝", "：", "，", "；", "（", "）", "［", "］", "｛", "｝",
]

def _read_lines_any_encoding(path: Path) -> List[str]:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return path.read_text(encoding=enc).splitlines()
        except Exception:
            pass
    # 最后再抛
    return path.read_text(encoding="utf-8").splitlines()

def read_label_file(labels_path: str) -> List[Tuple[str, str]]:
    """
    读取 labels.txt，返回 [(filename, label), ...]
    - 容错: 支持 \t 或空格分隔；忽略缺失/空行
    """
    p = Path(labels_path)
    if not p.exists():
        raise FileNotFoundError(f"未找到标签文件: {labels_path}")
    out: List[Tuple[str, str]] = []
    for ln in _read_lines_any_encoding(p):
        ln = ln.strip()
        if not ln:
            continue
        if "\t" in ln:
            fn, lab = ln.split("\t", 1)
        else:
            parts = ln.split()
            if len(parts) < 2:
                continue
            fn, lab = parts[0], " ".join(parts[1:])
        out.append((fn, lab))
    return out

def _collect_chars_from_labels(labels_path: str) -> List[str]:
    chars = []
    for _, lab in read_label_file(labels_path):
        chars.extend(list(lab))
    return chars

def build_charset(train_labels_path: str, val_labels_path: str = None) -> str:
    """
    从 train/val 的 labels.txt 里聚合字符，生成 charset 字符串（不含 blank）
    排序规则：数字 -> 英大写 -> 英小写 -> 预设符号序 -> 其余（按 Unicode）
    """
    pool = set()
    for ch in _collect_chars_from_labels(train_labels_path):
        pool.add(ch)
    if val_labels_path and Path(val_labels_path).exists():
        for ch in _collect_chars_from_labels(val_labels_path):
            pool.add(ch)

    digits  = [c for c in "0123456789" if c in pool]
    uppers  = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c in pool]
    lowers  = [c for c in "abcdefghijklmnopqrstuvwxyz" if c in pool]
    symbols = [c for c in _SYMBOLS_ORDER if c in pool]
    others  = sorted(ch for ch in pool if ch not in set(digits + uppers + lowers + symbols))

    charset = "".join(digits + uppers + lowers + symbols + others)
    return charset

def save_charset(charset: str, out_path: str):
    Path(out_path).write_text(charset, encoding="utf-8")

def load_charset(path: str):
    """
    返回 (charset_string, blank_id)
    你的训练代码只用到了第一个返回值的长度；blank 固定为 0 与 CTCLoss(blank=0) 对齐。
    """
    txt = Path(path).read_text(encoding="utf-8")
    return txt, 0
