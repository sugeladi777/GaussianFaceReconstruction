"""步骤2：批量预处理（mask + RGBA）。"""

import argparse
import os
import subprocess
from pathlib import Path


def run_step(args, cwd):
    print("[step_preprocess]", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Step 2: preprocess datasets")
    parser.add_argument("--repo-root", type=str, required=True)
    parser.add_argument("--python-exec", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--gpus", type=str, default="auto")
    parser.add_argument("--workers", type=int, default=0, help="并行 worker 数；0 表示从 PIPELINE_WORKERS / PIPELINE_DATASET_WORKERS 环境自动计算")
    parser.add_argument("--images-name", type=str, default="image", dest="raw_images_name")
    parser.add_argument("--masks-name", type=str, default="mask")
    parser.add_argument("--out-name", type=str, default="images", dest="composed_images_name")
    parser.add_argument("--remove-unmasked", type=int, choices=[0, 1], default=1)
    return parser


def main():
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    # 计算传给 preprocess_pipeline 的 workers：优先 CLI (>0)，否则从环境变量计算；若计算失败则传 0（让 preprocess_pipeline 使用 cpu_count）
    try:
        requested = int(args.workers)
    except Exception:
        requested = 0

    if requested > 0:
        workers_to_pass = requested
    else:
        try:
            total_workers = int(os.environ.get("PIPELINE_WORKERS", "0"))
            dataset_workers = int(os.environ.get("PIPELINE_DATASET_WORKERS", "1"))
            workers_to_pass = max(1, total_workers // max(1, dataset_workers)) if total_workers > 0 else 0
        except Exception:
            workers_to_pass = 0

    cmd = [
        args.python_exec,
        str(repo_root / "preprocess" / "preprocess_pipeline.py"),
        "--base",
        args.data_root,
        "--gpus",
        args.gpus,
            "--images-name",
            args.raw_images_name,
            "--masks-name",
            args.masks_name,
            "--out-name",
            args.composed_images_name,
        "--workers",
        str(workers_to_pass),
    ]
    if args.datasets:
        cmd.extend(["--datasets", args.datasets])
    if args.remove_unmasked == 1:
        cmd.append("--remove-unmasked")

    run_step(cmd, str(repo_root))


if __name__ == "__main__":
    main()
