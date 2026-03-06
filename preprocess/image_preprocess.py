#!/usr/bin/env python3
"""Simple image preprocessing utilities.

Functions:
 - preprocess_dir(in_dir, out_dir, width=None, height=None, mode='none')

Supported modes: 'none', 'blur', 'contrast', 'both'

This uses Pillow only (already used by repo). It resizes images to a unified
resolution and optionally applies a small Gaussian blur and/or contrast boost.
"""
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from typing import Optional


def _is_image_file(p: Path) -> bool:
    """判断文件是否为常见图像格式。"""
    return p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')


def preprocess_image(in_path: str, out_path: str, width: Optional[int], height: Optional[int], mode: str):
    """对单张图执行缩放与可选增强。"""
    img = Image.open(in_path).convert('RGB')

    # 仅做下采样（不放大）；始终保持宽高比。
    w, h = img.size
    if width and height:
        # compute scale to fit inside (width x height)
        scale_w = width / w
        scale_h = height / h
        scale = min(scale_w, scale_h, 1.0)
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
    elif width:
        if w > width:
            scale = width / w
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
    elif height:
        if h > height:
            scale = height / h
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)

    mode = (mode or 'none').lower()
    if mode in ('blur', 'both'):
        # 轻微高斯模糊抑制背景细节噪声。
        img = img.filter(ImageFilter.GaussianBlur(radius=2))

    if mode in ('contrast', 'both'):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.15)

    out_path_parent = Path(out_path).parent
    out_path_parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def preprocess_dir(in_dir: str, out_dir: str, width: Optional[int] = None, height: Optional[int] = None, mode: str = 'none') -> None:
    """Preprocess all images in `in_dir` and write to `out_dir`.

    保持原文件名，忽略非图像文件。
    """
    in_p = Path(in_dir)
    out_p = Path(out_dir)
    if not in_p.exists():
        raise FileNotFoundError(f"Input images dir not found: {in_dir}")
    out_p.mkdir(parents=True, exist_ok=True)

    for p in sorted(in_p.iterdir()):
        if not p.is_file() or not _is_image_file(p):
            continue
        out_file = out_p / p.name
        preprocess_image(str(p), str(out_file), width, height, mode)


def resize_masks_to_images(masks_in_dir: str, images_dir: str, masks_out_dir: str) -> None:
    """Resize masks from masks_in_dir to match each image in images_dir and write to masks_out_dir.

    假设 mask 与图像同名；使用最近邻插值以保留二值边界。
    """
    in_p = Path(masks_in_dir)
    img_p = Path(images_dir)
    out_p = Path(masks_out_dir)
    if not in_p.exists():
        raise FileNotFoundError(f"Masks input dir not found: {masks_in_dir}")
    if not img_p.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    out_p.mkdir(parents=True, exist_ok=True)

    for img_fp in sorted(img_p.iterdir()):
        if not img_fp.is_file():
            continue
        name = img_fp.name
        mask_fp = in_p / name
        if not mask_fp.exists():
            # skip if no corresponding mask
            continue
        img = Image.open(img_fp)
        mask = Image.open(mask_fp).convert('L')
        # PIL 尺寸顺序为 (W, H)。
        mask_resized = mask.resize(img.size, resample=Image.NEAREST)
        mask_resized.save(out_p / name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess a directory of images (resize + blur/contrast)')
    parser.add_argument('--in-dir', required=True, help='Input images directory')
    parser.add_argument('--out-dir', required=True, help='Output directory for preprocessed images')
    parser.add_argument('--width', type=int, default=0, help='Target width (0 = keep)')
    parser.add_argument('--height', type=int, default=0, help='Target height (0 = keep)')
    parser.add_argument('--mode', choices=['none', 'blur', 'contrast', 'both'], default='none', help='Preprocessing mode')

    args = parser.parse_args()
    w = args.width or None
    h = args.height or None
    preprocess_dir(args.in_dir, args.out_dir, w, h, args.mode)
