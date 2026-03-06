"""步骤1：视频拆帧。"""

import argparse
import os
import subprocess
from pathlib import Path


def run_step(args, cwd):
    print("[step_frames]", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Step 1: extract frames from videos")
    parser.add_argument("--repo-root", type=str, required=True)
    parser.add_argument("--python-exec", type=str, required=True)
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--workers", type=int, default=0, help="并行 worker 数；0 表示从 PIPELINE_WORKERS 环境或默认值计算")
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--resize-w", type=int, default=0)
    parser.add_argument("--resize-h", type=int, default=0)
    parser.add_argument("--rotate", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    # 计算实际用于拆帧脚本的 workers：优先使用 CLI，否则使用 PIPELINE_WORKERS 环境变量，最后回退到 1
    try:
        workers_val = int(args.workers)
    except Exception:
        workers_val = 0

    if workers_val <= 0:
        try:
            env_workers = int(os.environ.get("PIPELINE_WORKERS", "0"))
            workers_val = env_workers if env_workers > 0 else 1
        except Exception:
            workers_val = 1

    cmd = [
        args.python_exec,
        str(repo_root / "tools" / "video_to_frames.py"),
        "--video_dir",
        args.video_dir,
        "--data_root",
        args.data_root,
        "--workers",
        str(workers_val),
        "--step_size",
        str(args.step_size),
        "--resize",
        str(args.resize_w),
        str(args.resize_h),
    ]
    if args.rotate:
        cmd.append("--rotate")

    run_step(cmd, str(repo_root))


if __name__ == "__main__":
    main()
