"""RGBA 合成工具：将 RGB 图像和灰度 mask 合并为带 Alpha 通道图像。"""

import argparse
from pathlib import Path
from typing import Iterable, Set, Tuple

import cv2
import numpy as np
import concurrent.futures


class RGBAComposer:
    """Compose RGBA images from RGB images and grayscale masks using OpenCV."""

    def compose(self, img_path: str, mask_path: str, out_path: str) -> bool:
        """合成单张 RGBA 图像；成功返回 True。"""
        if not Path(mask_path).exists():
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

        # 确保 mask 为 uint8 且取值范围在 0..255
        if mask.dtype != np.uint8:
            if mask.dtype.kind == "f":
                mask = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)

        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(out_path, rgba)

    @staticmethod
    def _normalized_exts(exts: Iterable[str]) -> Set[str]:
        return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in exts}

    def compose_dir(
        self,
        image_dir: str,
        mask_dir: str,
        out_dir: str,
        exts: Iterable[str] = (".png", ".jpg", ".jpeg"),
        num_workers: int = 1,
    ) -> Tuple[int, int]:
        """批量合成目录下所有图像。

        Returns:
            (n_success, n_fail)
        """
        image_root = Path(image_dir)
        mask_root = Path(mask_dir)
        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        allowed_exts = self._normalized_exts(exts)
        n_success = 0
        n_fail = 0

        img_files = [p for p in sorted(image_root.iterdir()) if p.is_file() and p.suffix.lower() in allowed_exts]
        if not img_files:
            return 0, 0

        if num_workers <= 1:
            for img_path in img_files:
                mask_path = mask_root / img_path.name
                out_path = out_root / img_path.name
                ok = self.compose(str(img_path), str(mask_path), str(out_path))
                if not ok:
                    print(f"Failed to compose {img_path.name}")
                    n_fail += 1
                else:
                    n_success += 1
            return n_success, n_fail

        # 并行处理（使用线程池，IO + CPU 混合任务适合线程）
        def _task(p: Path) -> bool:
            return self.compose(str(p), str(mask_root / p.name), str(out_root / p.name))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
            for ok in ex.map(_task, img_files):
                if ok:
                    n_success += 1
                else:
                    n_fail += 1
        return n_success, n_fail


def _cli():
    p = argparse.ArgumentParser(description='Compose RGBA images from RGB images and masks')
    p.add_argument('--image_dir', required=True, help='Directory with input RGB images')
    p.add_argument('--mask_dir', required=True, help='Directory with grayscale masks (same filenames)')
    p.add_argument('--out_dir', required=True, help='Directory to write RGBA images')
    p.add_argument('--exts', nargs='+', default=['.png', '.jpg', '.jpeg'], help='Allowed image extensions')
    p.add_argument('--num-workers', type=int, default=1, help='并行 worker 数（线程），默认 1')
    args = p.parse_args()

    composer = RGBAComposer()
    success, fail = composer.compose_dir(args.image_dir, args.mask_dir, args.out_dir, exts=tuple(args.exts), num_workers=args.num_workers)
    print(f"Composed {success} images, failed {fail} images")


if __name__ == '__main__':
    _cli()

