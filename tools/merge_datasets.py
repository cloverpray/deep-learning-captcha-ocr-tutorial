# -*- coding: utf-8 -*-
"""
合并多来源验证码数据到 dataset/train 和 dataset/test
用法示例见文末。脚本假定每个来源目录结构为：
  <SRC>/
    images/
      xxx.png|jpg...
    labels.txt   # 每行: 文件名 \t 标签

注意：中文/特殊符号统一以 UTF-8 读写；可选去重按 MD5。
"""

import argparse, os, sys, shutil, hashlib, random
from pathlib import Path
from collections import Counter, defaultdict

def read_labels(src_dir: Path):
    """读取来源目录的 labels.txt（UTF-8，容错 BOM/GBK），返回[(Path,label), ...]"""
    lbl = src_dir / "labels.txt"
    if not lbl.exists():
        raise FileNotFoundError(f"{src_dir} 缺少 labels.txt")
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_err = None
    for enc in encodings:
        try:
            lines = lbl.read_text(encoding=enc).splitlines()
            pairs = []
            for ln in lines:
                ln = ln.strip()
                if not ln: continue
                if "\t" in ln:
                    fn, lab = ln.split("\t", 1)
                else:
                    # 兜底：空格分隔
                    parts = ln.split()
                    if len(parts) < 2:
                        continue
                    fn, lab = parts[0], " ".join(parts[1:])
                p = src_dir / "images" / fn
                if not p.exists():
                    # 某些来源可能是 jpg
                    alt = None
                    stem = Path(fn).stem
                    for ext in (".png",".jpg",".jpeg",".JPG",".JPEG",".PNG"):
                        cand = src_dir / "images" / f"{stem}{ext}"
                        if cand.exists():
                            alt = cand; break
                    if alt is None:
                        # 跳过缺失文件
                        continue
                    p = alt
                pairs.append((p, lab))
            return pairs
        except Exception as e:
            last_err = e
    raise last_err

def md5sum(path: Path, block=1<<20):
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_one(src: Path, dst: Path, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "link":
        try:
            os.link(src, dst)  # 硬链接（Windows 需要 NTFS & 权限）
        except Exception:
            shutil.copy2(src, dst)  # 回退到 copy
    else:
        shutil.copy2(src, dst)

def save_charset(all_labels, out_path: Path):
    s = set()
    for _,lab in all_labels:
        for ch in lab:
            s.add(ch)
    # 固定排序：先数字，后英大写、英文小写、常见符号、剩余按 Unicode
    digits = [c for c in "0123456789" if c in s]
    uppers = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c in s]
    lowers = [c for c in "abcdefghijklmnopqrstuvwxyz" if c in s]
    symbols_order = [c for c in "+-*/=×÷？?()[]{}#@$%&._:,;^~<>\\" if c in s]
    others = sorted(ch for ch in s if ch not in set(digits+uppers+lowers+symbols_order))
    charset = "".join(digits + uppers + lowers + symbols_order + others)
    out_path.write_text(charset, encoding="utf-8")
    return charset

def merge_split(dest_root: Path, sources, shuffle=True, seed=42, copy_mode="copy",
                dedupe=False, prefix=None, limit_per_source=None):
    """
    合并到某个 split（train 或 test）
    sources: [(tag, src_dir_path), ...]，src_dir_path 指向单个来源目录（包含 images/ 与 labels.txt）
    """
    dest_images = dest_root / "images"
    ensure_dir(dest_images)
    all_pairs = []
    per_source_cnt = {}
    for tag, src in sources:
        pairs = read_labels(Path(src))
        if limit_per_source:
            pairs = pairs[:limit_per_source]
        all_pairs.extend([(tag, p, lab) for p, lab in pairs])
        per_source_cnt[tag] = len(pairs)

    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(all_pairs)

    # 去重（按 MD5）
    seen = set()
    kept = []
    if dedupe:
        print(f"[*] 开启去重（MD5）...")
        for tag, p, lab in all_pairs:
            h = md5sum(p)
            key = (h, lab)  # 同图同标签才判为重复
            if key in seen: 
                continue
            seen.add(key)
            kept.append((tag, p, lab))
    else:
        kept = all_pairs

    labels_out = dest_root / "labels.txt"
    with labels_out.open("w", encoding="utf-8", newline="\n") as fw:
        idx = 0
        for tag, p, lab in kept:
            idx += 1
            ext = p.suffix.lower()
            if ext not in [".png",".jpg",".jpeg",".bmp",".webp"]:
                ext = ".png"
            name = f"{prefix+'_' if prefix else ''}{idx:08d}{ext}"
            copy_one(p, dest_images / name, copy_mode)
            fw.write(f"{name}\t{lab}\n")

    # 统计
    total = len(kept)
    by_src = Counter(tag for tag,_,_ in kept)
    print(f"\n✅ 合并完成 -> {dest_root}")
    print(f"   共计: {total} 张（{dest_images}）")
    print("== 各来源计数（保留）==")
    for k,v in by_src.items():
        print(f"  - {k:15s}: {v:7d}")
    if dedupe:
        removed = len(all_pairs) - len(kept)
        print(f"== 去重移除 == {removed} 张")
    return [(dest_images / (f"{prefix+'_' if prefix else ''}{i+1:08d}"+Path(kept[i][1]).suffix.lower()), kept[i][2]) for i in range(len(kept))]

def main():
    ap = argparse.ArgumentParser(description="合并验证码数据到 dataset/train|test")
    ap.add_argument("--train-dest", default="dataset/train", help="训练集输出根目录")
    ap.add_argument("--test-dest",  default="dataset/test",  help="测试集输出根目录")
    ap.add_argument("--train-src", action="append", nargs=2, metavar=("TAG","PATH"),
                    help="训练来源 (可重复)。示例: --train-src easy tools/java_captcha/EasyCaptcha/images/easy_train")
    ap.add_argument("--test-src", action="append", nargs=2, metavar=("TAG","PATH"),
                    help="测试来源 (可重复)。示例: --test-src easy tools/java_captcha/EasyCaptcha/images/easy_test")
    ap.add_argument("--no-shuffle", action="store_true", help="不要打乱")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy-mode", choices=["copy","move","link"], default="copy")
    ap.add_argument("--dedupe", action="store_true", help="按 MD5+标签 去重")
    ap.add_argument("--limit-train-per-src", type=int, default=None, help="每个训练来源最多取多少")
    ap.add_argument("--limit-test-per-src", type=int, default=None, help="每个测试来源最多取多少")
    ap.add_argument("--charset-out", default="charset.txt", help="合并后的字符集输出路径（同时拷贝到 train/test 根）")
    args = ap.parse_args()

    train_dest = Path(args.train_dest)
    test_dest  = Path(args.test_dest)
    ensure_dir(train_dest / "images")
    ensure_dir(test_dest  / "images")

    # 合并 train
    if args.train_src:
        print("[*] 合并训练集 ...")
        merge_split(train_dest, args.train_src,
                    shuffle=not args.no_shuffle, seed=args.seed,
                    copy_mode=args.copy_mode, dedupe=args.dedupe,
                    prefix="tr", limit_per_source=args.limit_train_per_src)
    # 合并 test
    if args.test_src:
        print("\n[*] 合并测试集 ...")
        merge_split(test_dest, args.test_src,
                    shuffle=not args.no_shuffle, seed=args.seed+1,
                    copy_mode=args.copy_mode, dedupe=args.dedupe,
                    prefix="te", limit_per_source=args.limit_test_per_src)

    # 生成 charset（基于 train+test 的 labels）
    all_pairs = []
    for split in [train_dest, test_dest]:
        lbl = split / "labels.txt"
        if lbl.exists():
            for ln in lbl.read_text(encoding="utf-8").splitlines():
                if "\t" in ln:
                    fn, lab = ln.split("\t",1)
                    all_pairs.append((fn, lab))
    charset = save_charset(all_pairs, Path(args.charset_out))
    # 同时放一份到 train/test 根
    try:
        shutil.copy2(args.charset_out, train_dest / "charset.txt")
        shutil.copy2(args.charset_out, test_dest / "charset.txt")
    except Exception:
        pass

    # 打印字符集信息
    print("\n== 合并后的字符集 ==")
    print(charset)
    print(f"\n✅ 完成。charset.txt -> {Path(args.charset_out).resolve()}")

if __name__ == "__main__":
    main()
