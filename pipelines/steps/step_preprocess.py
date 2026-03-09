import argparse
import os
import subprocess
from pathlib import Path


def run_step(args, cwd):
    print("[step_preprocess]", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Step 2: preprocess datasets")
    parser.add_argument("--repo-root", type=str, required=True, help="项目根目录")
    parser.add_argument("--python-exec", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--gpus", type=str, default="auto")
    parser.add_argument("--workers", type=int, default=0, help="并行 worker 数")
    parser.add_argument("--images-name", type=str, default="image", dest="raw_images_name")
    parser.add_argument("--masks-name", type=str, default="mask")
    parser.add_argument("--out-name", type=str, default="images", dest="composed_images_name")
    parser.add_argument("--remove-unmasked", type=int, choices=[0, 1], default=1)
    return parser



def main():
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    workers = args.workers

    preprocess_script = repo_root / "pipelines" / "steps" / "preprocess" / "preprocess_pipeline.py"
    cmd = [
        args.python_exec,
        str(preprocess_script),
        "--base", args.data_root,
        "--gpus", args.gpus,
        "--images-name", args.raw_images_name,
        "--masks-name", args.masks_name,
        "--out-name", args.composed_images_name,
        "--workers", str(workers),
    ]
    if args.datasets:
        cmd += ["--datasets", args.datasets]
    if args.remove_unmasked == 1:
        cmd.append("--remove-unmasked")

    run_step(cmd, str(repo_root))


if __name__ == "__main__":
    main()
