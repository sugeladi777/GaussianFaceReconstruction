#!/usr/bin/env python3
"""预处理总入口：图像预处理 + 人脸分割 mask + RGBA 合成。

支持两种模式：
1) 单数据集模式：--images --masks --out
2) 批量模式：--base（遍历子目录）
"""

import argparse
import concurrent.futures
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from FaceMaskGenerator import FaceMaskGenerator
from RGBAComposer import RGBAComposer
from image_preprocess import preprocess_dir


def resolve_worker_devices(gpus_arg: Optional[str]) -> List[str]:
    """解析批处理可用设备列表。"""
    if not torch.cuda.is_available():
        return ["cpu"]

    visible_count = torch.cuda.device_count()
    if visible_count <= 0:
        return ["cpu"]

    if gpus_arg and gpus_arg.lower() != "auto":
        requested = [x.strip() for x in gpus_arg.split(",") if x.strip()]
        if requested:
            use_count = min(len(requested), visible_count)
            return [f"cuda:{idx}" for idx in range(use_count)]

    return [f"cuda:{idx}" for idx in range(visible_count)]


def remove_images_without_masks(images_dir: str, masks_dir: str) -> int:
    """删除 images_dir 中没有对应 mask 的 PNG 图像。"""
    deleted = 0
    try:
        img_names = sorted([name for name in os.listdir(images_dir) if name.lower().endswith('.png')])
        mask_set = set([name for name in os.listdir(masks_dir) if name.lower().endswith('.png')])
        for name in img_names:
            if name not in mask_set:
                try:
                    os.remove(os.path.join(images_dir, name))
                    deleted += 1
                except Exception as exc:
                    print(f"Warning: failed to delete image without mask {name}: {exc}")
    except Exception as exc:
        print(f"Warning: remove_images_without_masks failed: {exc}")

    if deleted:
        print(f"Removed {deleted} images without masks from {images_dir}")
    return deleted


def preprocess_and_generate_masks(
    images_dir: str,
    masks_dir: str,
    resize_w: Optional[int],
    resize_h: Optional[int],
    preprocess_mode: str,
    preprocess_name: str,
    device: str,
    remove_unmasked: bool = False,
) -> Tuple[str, str, bool]:
    """执行预处理与 mask 生成阶段。

    Returns:
        rgba_images_dir: RGBA 合成阶段使用的图像目录
        final_masks_dir: mask 目录
        preprocessed_used: 是否使用了临时预处理目录
    """
    preprocessed_images = None
    if resize_w or resize_h or (preprocess_mode and preprocess_mode.lower() != 'none'):
        preprocessed_images = os.path.join(os.path.dirname(images_dir), preprocess_name)
        print(
            f"Preprocessing images -> {preprocessed_images} "
            f"(width={resize_w} height={resize_h} mode={preprocess_mode})"
        )
        preprocess_dir(images_dir, preprocessed_images, resize_w, resize_h, preprocess_mode)

    # 没有预处理时，直接用原图目录做分割；有预处理时，先在预处理图上分割。
    face_mask_input = preprocessed_images if preprocessed_images else images_dir
    mask_generator = FaceMaskGenerator(device=device)
    mask_generator.generate_masks_from_dir(face_mask_input, masks_dir)

    if remove_unmasked:
        # 保持图像和 mask 严格对齐：删掉没有 mask 的图像。
        remove_images_without_masks(face_mask_input, masks_dir)
        # 如果分割发生在临时预处理目录，也同步清理原图目录中对应缺失项。
        if preprocessed_images:
            remove_images_without_masks(images_dir, masks_dir)

    if preprocessed_images:
        return preprocessed_images, masks_dir, True
    return images_dir, masks_dir, False


def run_one_dataset(args) -> str:
    """批量模式下单个数据集任务（供进程池 map 调用）。"""
    (
        dataset_name,
        base,
        images_name,
        masks_name,
        out_name,
        resize_w,
        resize_h,
        preprocess_mode,
        preprocess_name,
        device,
        remove_unmasked,
    ) = args

    ds_path = Path(base) / dataset_name
    images_dir = ds_path / images_name
    masks_dir = ds_path / masks_name
    out_dir = ds_path / out_name

    if not images_dir.exists():
        print(f"Skipping dataset {dataset_name}: images dir not found: {images_dir}")
        return f"{dataset_name}: skipped (no images dir)"

    masks_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing dataset {dataset_name}: images={images_dir} masks={masks_dir} out={out_dir}")
    rgba_images_dir, final_masks_dir, preprocessed_used = preprocess_and_generate_masks(
        str(images_dir),
        str(masks_dir),
        resize_w,
        resize_h,
        preprocess_mode,
        preprocess_name,
        device,
        remove_unmasked,
    )

    print(f"Composing RGBA into: {out_dir}")
    composer = RGBAComposer()
    composer.compose_dir(rgba_images_dir, final_masks_dir, str(out_dir))

    # 仅清理临时预处理目录，保留 mask 与 RGBA 输出。
    if preprocessed_used:
        try:
            shutil.rmtree(rgba_images_dir, ignore_errors=True)
        except Exception as exc:
            print(f"Warning: failed to cleanup intermediates for {dataset_name}: {exc}")

    return f"{dataset_name}: done"


def process_one(
    images: str,
    masks: str,
    out: str,
    device: str,
    resize_w: Optional[int] = None,
    resize_h: Optional[int] = None,
    preprocess_mode: str = 'none',
    preprocess_name: str = 'images_preprocessed',
    remove_unmasked: bool = False,
) -> None:
    """单数据集执行入口。"""
    images = os.fspath(images)
    masks = os.fspath(masks)
    out = os.fspath(out)

    rgba_images_dir, final_masks_dir, preprocessed_used = preprocess_and_generate_masks(
        images,
        masks,
        resize_w,
        resize_h,
        preprocess_mode,
        preprocess_name,
        device,
        remove_unmasked,
    )

    print(f"Composing RGBA into: {out}")
    composer = RGBAComposer()
    composer.compose_dir(rgba_images_dir, final_masks_dir, out)

    if preprocessed_used:
        try:
            shutil.rmtree(rgba_images_dir, ignore_errors=True)
        except Exception as exc:
            print(f"Warning: failed to cleanup intermediates for {images}: {exc}")


def gather_datasets_from_base(base: str, datasets_arg: Optional[str]) -> List[str]:
    """解析批量模式数据集列表。"""
    base_path = Path(base)
    if not base_path.exists():
        raise FileNotFoundError(f"Base dir not found: {base}")

    if datasets_arg:
        return [dataset.strip() for dataset in datasets_arg.split(',') if dataset.strip()]

    return sorted([path.name for path in base_path.iterdir() if path.is_dir()])


def main_batch_mode(
    base: str,
    datasets_arg: Optional[str],
    images_name: str,
    masks_name: str,
    out_name: str,
    resize_w: Optional[int] = None,
    resize_h: Optional[int] = None,
    preprocess_mode: str = 'none',
    preprocess_name: str = 'images_preprocessed',
    gpus_arg: Optional[str] = None,
    workers: Optional[int] = None,
    remove_unmasked: bool = False,
) -> None:
    """批量模式主流程。"""
    worker_devices = resolve_worker_devices(gpus_arg)
    datasets = gather_datasets_from_base(base, datasets_arg)

    if not datasets:
        print(f"No datasets found under base: {base}")
        return

    print(f"Found {len(datasets)} datasets. Devices={worker_devices}")
    print(f"Datasets: {datasets}")

    if workers is None:
        workers = multiprocessing.cpu_count()
    print(f"Using {workers} workers for parallel dataset processing.")

    args_list = [
        (
            dataset,
            base,
            images_name,
            masks_name,
            out_name,
            resize_w,
            resize_h,
            preprocess_mode,
            preprocess_name,
            worker_devices[idx % len(worker_devices)],
            remove_unmasked,
        )
        for idx, dataset in enumerate(datasets)
    ]

    if workers <= 1:
        results = [run_one_dataset(item) for item in args_list]
        print("Batch processing complete")
        print("Results:", results)
        return

    # 使用 spawn 避免 CUDA 在 fork 子进程里重复初始化。
    try:
        ctx = multiprocessing.get_context('spawn')
    except Exception:
        ctx = None

    if ctx is not None:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            results = list(executor.map(run_one_dataset, args_list))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run_one_dataset, args_list))

    print("Batch processing complete")
    print("Results:", results)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数定义。"""
    parser = argparse.ArgumentParser(description='Preprocess images: head segmentation + RGBA composition (single or batch mode)')

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--images', help='Directory with input RGB images (single-run mode)')
    mode.add_argument('--base', help='Base directory containing dataset subfolders (batch mode)')

    parser.add_argument('--masks', help='Directory to store/read masks (single-run mode)')
    parser.add_argument('--out', help='Directory to store RGBA output (single-run mode)')

    parser.add_argument('--resize-width', type=int, default=0, help='Resize width for preprocessing (0 = keep)')
    parser.add_argument('--resize-height', type=int, default=0, help='Resize height for preprocessing (0 = keep)')
    parser.add_argument('--preprocess-mode', choices=['none', 'blur', 'contrast', 'both'], default='none', help='Image preprocessing mode applied before mask detection')
    parser.add_argument('--preprocess-name', default='images_preprocessed', help='Subfolder name for preprocessed images created next to the original images')

    parser.add_argument('--datasets', help='Comma-separated list of dataset folder names to process under --base (default: all subdirs)')
    parser.add_argument('--gpus', default='auto', help="GPU ids for batch mode, e.g. '0,1' (default: auto)")
    parser.add_argument('--images-name', default='image', help="Subfolder name for images inside each dataset (default 'image')")
    parser.add_argument('--masks-name', default='mask', help="Subfolder name for masks inside each dataset (default 'mask')")
    parser.add_argument('--out-name', default='images', help="Subfolder name for RGBA outputs inside each dataset (default 'images')")
    parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers for batch mode (default: cpu count)')
    parser.add_argument('--remove-unmasked', action='store_true', help='Delete images without corresponding masks after mask generation')
    return parser


def main() -> None:
    """程序入口。"""
    parser = build_parser()
    args = parser.parse_args()

    resize_w = args.resize_width or None
    resize_h = args.resize_height or None

    if args.images:
        if not args.masks or not args.out:
            parser.error('--images requires --masks and --out in single-run mode')

        process_one(
            args.images,
            args.masks,
            args.out,
            'cuda:0' if torch.cuda.is_available() else 'cpu',
            resize_w=resize_w,
            resize_h=resize_h,
            preprocess_mode=args.preprocess_mode,
            preprocess_name=args.preprocess_name,
            remove_unmasked=args.remove_unmasked,
        )
        return

    workers = args.workers or None
    main_batch_mode(
        args.base,
        args.datasets,
        args.images_name,
        args.masks_name,
        args.out_name,
        resize_w=resize_w,
        resize_h=resize_h,
        preprocess_mode=args.preprocess_mode,
        preprocess_name=args.preprocess_name,
        gpus_arg=args.gpus,
        workers=workers,
        remove_unmasked=args.remove_unmasked,
    )


if __name__ == '__main__':
    main()
