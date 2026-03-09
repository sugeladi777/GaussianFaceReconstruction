import argparse
import subprocess
from pathlib import Path


def run_step(args, cwd):
    print("[step_frames]", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="步骤1：从视频中抽取帧图像")
    parser.add_argument("--python-exec", type=str, required=True, help="Python 解释器路径")
    parser.add_argument("--video-dir", type=str, required=True, help="输入视频文件夹路径")
    parser.add_argument("--output-root", type=str, required=True, help="输出图片根目录")
    parser.add_argument("--workers", type=int, default=0, help="并行 worker 数，0 表示自动检测 CPU 数量")
    parser.add_argument("--step-size", type=int, default=1, help="帧抽取间隔，1 表示每帧都取")
    parser.add_argument("--resize-w", type=int, default=0, help="缩放输出图片宽度，0 表示不缩放")
    parser.add_argument("--resize-h", type=int, default=0, help="缩放输出图片高度，0 表示不缩放")
    parser.add_argument("--rotate", action="store_true", help="是否对图片顺时针旋转90度")
    return parser


def main():
    args = build_parser().parse_args()
    step_root = Path(__file__).resolve().parent
    video_script = step_root / "tools" / "video_to_frames.py"

    cmd = [
        args.python_exec,
        str(video_script),
        "--video_dir",
        args.video_dir,
        "--data_root",
        args.output_root,
        "--workers",
        str(args.workers),
        "--step_size",
        str(args.step_size),
        "--resize",
        str(args.resize_w),
        str(args.resize_h),
    ]
    if args.rotate:
        cmd.append("--rotate")

    run_step(cmd, str(step_root))


if __name__ == "__main__":
    main()
