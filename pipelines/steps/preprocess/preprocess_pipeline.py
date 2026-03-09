#!/usr/bin/env python3
"""批量预处理入口：人脸分割 + RGBA 合成。"""

import argparse
import concurrent.futures
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from FaceMaskGenerator import FaceMaskGenerator
from RGBAComposer import RGBAComposer

_GENERATOR_CACHE: Dict[str, FaceMaskGenerator] = {}


def resolve_worker_devices(gpus_arg: Optional[str]) -> List[str]:
    n = torch.cuda.device_count()
    if n <= 0:
        return ["cpu"]
    if gpus_arg and gpus_arg.lower() != "auto":
        idxs = [int(tok) for tok in gpus_arg.split(",") if tok.strip().isdigit()]
        idxs = [i for i in idxs if 0 <= i < n]
        if idxs:
            return [f"cuda:{i}" for i in idxs]
    return [f"cuda:{i}" for i in range(n)]


def get_mask_generator(device: str) -> FaceMaskGenerator:
    """按设备复用 FaceMaskGenerator，避免重复加载模型。"""
    generator = _GENERATOR_CACHE.get(device)
    if generator is None:
        generator = FaceMaskGenerator(device=device)
        _GENERATOR_CACHE[device] = generator
    return generator


def remove_images_without_masks(images_dir: Path, masks_dir: Path) -> int:
    # 删除 images_dir 中没有对应 mask 的 PNG 图片并返回删除数量
    if not images_dir.is_dir() or not masks_dir.is_dir():
        return 0
    mask_names = {p.name for p in masks_dir.glob("*.png")}
    to_delete = [p for p in images_dir.glob("*.png") if p.name not in mask_names]
    for p in to_delete:
        try:
            p.unlink()
        except OSError as exc:
            print(f"警告: 删除未匹配 mask 的图片失败 {p.name}: {exc}")
    return len(to_delete)


def generate_masks_and_compose(
    images_dir: Path,
    masks_dir: Path,
    out_dir: Path,
    device: str,
    remove_unmasked: bool,
    compose_workers: int = 1,
) -> None:
    """生成 mask 并输出 RGBA 图像。"""
    gen = get_mask_generator(device)
    gen.generate_masks_from_dir(str(images_dir), str(masks_dir))
    if remove_unmasked:
        remove_images_without_masks(images_dir, masks_dir)
    RGBAComposer().compose_dir(str(images_dir), str(masks_dir), str(out_dir), num_workers=compose_workers)


def run_one_dataset(args: Tuple[str, str, str, str, str, str, bool, int]) -> str:
    dataset_name, base, images_name, masks_name, out_name, device, remove_unmasked, compose_workers = args

    ds_path = Path(base) / dataset_name
    images_dir = ds_path / images_name
    masks_dir = ds_path / masks_name
    out_dir = ds_path / out_name

    masks_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始处理 {dataset_name}: images={images_dir} masks={masks_dir} out={out_dir}")
    generate_masks_and_compose(images_dir, masks_dir, out_dir, device, remove_unmasked, compose_workers)
    return f"{dataset_name}: 完成"


def gather_datasets_from_base(base: Path, datasets_arg: str, images_name: str) -> List[str]:

    if datasets_arg:
        return [name.strip() for name in datasets_arg.split(",") if name.strip()]

    return sorted([p.name for p in base.iterdir() if p.is_dir() and (p / images_name).is_dir()])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量预处理：人脸分割 + RGBA 合成")
    parser.add_argument("--base", required=True, help="数据集根目录（其下每个子目录视为一个数据集）")
    parser.add_argument("--datasets", default="", help="指定要处理的数据集名，多个用逗号分隔；留空则处理全部")
    parser.add_argument("--gpus", default="auto", help="使用的 GPU 列表，如 '0,1'；默认 auto")
    parser.add_argument("--images-name", default="image", help="每个数据集中的输入图片子目录名")
    parser.add_argument("--masks-name", default="mask", help="每个数据集中的 mask 输出子目录名")
    parser.add_argument("--out-name", default="images", help="每个数据集中的 RGBA 输出子目录名")
    parser.add_argument("--workers", type=int, default=0, help="并行 worker 数，0 表示自动使用 CPU 核心数")
    parser.add_argument("--compose-workers", type=int, default=1, help="RGBA 合成并行线程数（默认 1）")
    parser.add_argument("--remove-unmasked", action="store_true", help="生成 mask 后删除没有对应 mask 的图片")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    base = Path(args.base)
    worker_devices = resolve_worker_devices(args.gpus)
    datasets = gather_datasets_from_base(base, args.datasets, args.images_name)

    workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
    workers = max(1, min(workers, len(datasets)))

    task_args = [
        (
            dataset,
            str(base),
            args.images_name,
            args.masks_name,
            args.out_name,
            worker_devices[idx % len(worker_devices)],
            args.remove_unmasked,
            args.compose_workers,
        )
        for idx, dataset in enumerate(datasets)
    ]

    try:
        ctx = multiprocessing.get_context("spawn")
    except Exception:
        ctx = None

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        results = list(executor.map(run_one_dataset, task_args))

    print("批量处理完成")
    print("结果:", results)


if __name__ == "__main__":
    main()
