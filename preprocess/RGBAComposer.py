"""RGBA 合成工具：将 RGB 图像和灰度 mask 合并为带 Alpha 通道图像。"""

import os
import argparse
import cv2
import numpy as np
from typing import Iterable, Tuple


class RGBAComposer:
    """Compose RGBA images from RGB images and grayscale masks using OpenCV."""

    def __init__(self):
        pass

    def compose(self, img_path: str, mask_path: str, out_path: str) -> bool:
        """合成单张 RGBA 图像；成功返回 True。"""
        if not os.path.exists(mask_path):
            return False
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            return False

        # mask 尺寸若与图像不一致，则最近邻缩放到图像尺寸。
        ih, iw = img.shape[:2]
        mh, mw = mask.shape[:2]
        if (ih, iw) != (mh, mw):
            mask = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        # 将 mask 写入 alpha 通道，值域保持 0-255。
        if mask.dtype != np.uint8:
            mask = (mask).astype(np.uint8)
        rgba[:, :, 3] = mask

        # 确保输出目录存在。
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        return cv2.imwrite(out_path, rgba)

    def compose_dir(self, image_dir: str, mask_dir: str, out_dir: str, exts: Iterable[str] = ('.png', '.jpg', '.jpeg')) -> Tuple[int,int]:
        """批量合成目录下所有图像。

        Returns:
            (n_success, n_fail)
        """
        os.makedirs(out_dir, exist_ok=True)
        n_success = 0
        n_fail = 0
        for fname in sorted(os.listdir(image_dir)):
            if not any(fname.lower().endswith(e) for e in exts):
                continue
            img_path = os.path.join(image_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            out_path = os.path.join(out_dir, fname)
            ok = self.compose(img_path, mask_path, out_path)
            if not ok:
                print(f"Failed to compose {fname}")
                n_fail += 1
            else:
                n_success += 1
        return n_success, n_fail


def _cli():
    p = argparse.ArgumentParser(description='Compose RGBA images from RGB images and masks')
    p.add_argument('--image_dir', required=True, help='Directory with input RGB images')
    p.add_argument('--mask_dir', required=True, help='Directory with grayscale masks (same filenames)')
    p.add_argument('--out_dir', required=True, help='Directory to write RGBA images')
    p.add_argument('--exts', nargs='+', default=['.png', '.jpg', '.jpeg'], help='Allowed image extensions')
    args = p.parse_args()

    composer = RGBAComposer()
    success, fail = composer.compose_dir(args.image_dir, args.mask_dir, args.out_dir, exts=tuple(args.exts))
    print(f"Composed {success} images, failed {fail} images")


if __name__ == '__main__':
    _cli()

