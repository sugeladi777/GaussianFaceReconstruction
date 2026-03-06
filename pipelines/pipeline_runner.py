"""Unified pipeline runner: frames -> preprocess -> colmap -> 2dGS -> texture."""

import argparse
import concurrent.futures
import os
import subprocess
from pathlib import Path
from typing import List


def run_step(args: List[str], cwd: Path, env: dict = None) -> None:
    print("[pipeline_runner]", " ".join(args))
    subprocess.run(args, cwd=str(cwd), env=env, check=True)


def parse_gpu_list(gpus: str) -> List[str]:
    if gpus and gpus.lower() != "auto":
        gpu_list = [x.strip() for x in gpus.split(",") if x.strip()]
        return gpu_list

    try:
        result = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
        parsed = []
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if line.startswith("GPU "):
                gpu_id = line.split(":", 1)[0].replace("GPU", "").strip()
                if gpu_id.isdigit():
                    parsed.append(gpu_id)
        if parsed:
            return parsed
    except Exception:
        pass

    return ["0"]


def resolve_colmap_exec(repo_root: Path, colmap_bin: str) -> str:
    if colmap_bin != "auto":
        return os.path.abspath(colmap_bin)
    local_colmap = repo_root / "colmap" / "build" / "src" / "colmap" / "exe" / "colmap"
    return str(local_colmap) if local_colmap.is_file() else "colmap"


def find_datasets(data_root: Path) -> List[Path]:
    datasets = []
    for path in sorted(data_root.iterdir()):
        if path.is_dir() and ((path / "image").is_dir() or (path / "images").is_dir() or (path / "raw_frames").is_dir()):
            datasets.append(path)
    return datasets


def filter_datasets(datasets: List[Path], datasets_arg: str) -> List[Path]:
    if not datasets_arg:
        return datasets
    names = {x.strip() for x in datasets_arg.split(",") if x.strip()}
    return [d for d in datasets if d.name in names]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline runner: frames -> preprocess -> colmap -> 2dGS -> texture")
    parser.add_argument("--video_dir", type=str, default="/home/lichengkai/RGB_Recon/input/test", help="输入视频文件夹路径")
    parser.add_argument("--data_root", type=str, default="/home/lichengkai/RGB_Recon/output/test", help="输出数据根目录")
    parser.add_argument("--gpus", type=str, default="auto", help="使用的GPU编号，如'0,1'，auto为自动检测")

    parser.add_argument("--preprocess_python", type=str, default="/home/lichengkai/anaconda3/envs/preprocess/bin/python", help="预处理阶段Python解释器路径")
    parser.add_argument("--recon_python", type=str, default="/home/lichengkai/anaconda3/envs/2dGS/bin/python", help="重建阶段Python解释器路径")
    parser.add_argument("--texture_python", type=str, default="/home/lichengkai/anaconda3/envs/texture/bin/python", help="贴图阶段Python解释器路径")
    parser.add_argument("--colmap_bin", type=str, default="auto", help="COLMAP可执行文件路径，auto为自动查找")

    parser.add_argument("--step_size", type=int, default=1, help="视频帧抽帧间隔，1为每帧都取")
    parser.add_argument("--rotate", action="store_true", help="是否对图片进行旋转")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"), default=(0, 0), help="图片缩放尺寸，格式为 宽 高，0表示不缩放")

    parser.add_argument("--workers", type=int, default=30, help="并行处理的工作线程数（CPU 核心数），其他步骤将自动根据此值分配线程）")
    parser.add_argument("--remove_unmasked", type=int, choices=[0, 1], default=1, help="是否移除未被mask覆盖的区域，1为移除，0为保留")

    parser.add_argument("--use_colmap_mask", type=int, choices=[0, 1], default=0, help="COLMAP是否使用mask，1为使用，0为不使用")
    parser.add_argument("--mesh_res", type=int, default=1024, help="重建mesh的分辨率")
    parser.add_argument("--num_view", type=int, default=16, help="贴图渲染时的视角数量")

    parser.add_argument("--datasets", type=str, default="", help="指定要处理的数据集名称，多个用逗号分隔")
    parser.add_argument("--dataset_workers", type=int, default=0, help="数据集级并发数，0表示自动=GPU数量")

    parser.add_argument("--skip_frames", action="store_true", help="跳过帧提取步骤")
    parser.add_argument("--skip_preprocess", action="store_true", help="跳过预处理步骤")
    parser.add_argument("--skip_colmap", action="store_true", help="跳过COLMAP重建步骤")
    parser.add_argument("--skip_2dgs", action="store_true", help="跳过2D高斯重建步骤")
    parser.add_argument("--skip_texture", action="store_true", help="跳过贴图生成步骤")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    steps_root = repo_root / "pipelines" / "steps"
    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    preprocess_python = os.path.abspath(args.preprocess_python)
    recon_python = os.path.abspath(args.recon_python)
    texture_python = os.path.abspath(args.texture_python)
    colmap_bin = resolve_colmap_exec(repo_root, args.colmap_bin)
    gpu_list = parse_gpu_list(args.gpus)
    dataset_workers = args.dataset_workers if args.dataset_workers > 0 else len(gpu_list)
    is_single_gpu = len(gpu_list) == 1
    print(f"[pipeline_runner] GPUs: {','.join(gpu_list)}")
    print(f"[pipeline_runner] Single GPU: {is_single_gpu}")
    print(f"[pipeline_runner] Dataset workers: {dataset_workers}")
    print(f"[pipeline_runner] COLMAP bin: {colmap_bin}")

    if not args.skip_frames:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
        env["PIPELINE_WORKERS"] = str(args.workers)
        env["PIPELINE_DATASET_WORKERS"] = str(dataset_workers)
        run_step(
            [
                preprocess_python,
                str(steps_root / "step_frames.py"),
                "--repo-root",
                str(repo_root),
                "--python-exec",
                preprocess_python,
                "--video-dir",
                str(Path(args.video_dir).resolve()),
                "--data-root",
                str(data_root),
                "--workers",
                str(args.workers),
                "--step-size",
                str(args.step_size),
                "--resize-w",
                str(args.resize[0]),
                "--resize-h",
                str(args.resize[1]),
                *( ["--rotate"] if args.rotate else [] ),
            ],
            cwd=repo_root,
            env=env,
        )

    datasets = filter_datasets(find_datasets(data_root), args.datasets)
    if not datasets:
        print(f"[pipeline_runner] no datasets found in {data_root}")
        return

    if not args.skip_preprocess:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
        env["PIPELINE_WORKERS"] = str(args.workers)
        env["PIPELINE_DATASET_WORKERS"] = str(dataset_workers)
        run_step(
            [
                preprocess_python,
                str(steps_root / "step_preprocess.py"),
                "--repo-root",
                str(repo_root),
                "--python-exec",
                preprocess_python,
                "--data-root",
                str(data_root),
                "--datasets",
                ",".join(d.name for d in datasets),
                "--gpus",
                ",".join(gpu_list),
                "--workers",
                str(args.workers),
                "--remove-unmasked",
                str(args.remove_unmasked),
            ],
            cwd=repo_root,
            env=env,
        )

    def run_dataset_pipeline(idx: int, dataset: Path) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list[idx % len(gpu_list)]
        env["PIPELINE_WORKERS"] = str(args.workers)
        env["PIPELINE_DATASET_WORKERS"] = str(dataset_workers)
        print(f"[pipeline_runner] dataset={dataset.name} assigned_gpu={env['CUDA_VISIBLE_DEVICES']}")

        if not args.skip_colmap:
            # 根据全局 --workers 与数据集并发数自动分配给单个数据集的线程数
            workers_for_datasets = dataset_workers if dataset_workers > 0 else max(1, len(gpu_list))
            num_threads = max(1, int(args.workers) // max(1, workers_for_datasets))
            run_step(
                [
                    recon_python,
                    str(steps_root / "step_colmap.py"),
                    "--repo-root",
                    str(repo_root),
                    "--python-exec",
                    recon_python,
                    "--data-root",
                    str(dataset),
                    "--use-mask",
                    str(args.use_colmap_mask),
                    "--num-threads",
                    str(num_threads),
                    "--colmap-bin",
                    colmap_bin,
                ],
                cwd=repo_root,
                env=env,
            )

        if not args.skip_2dgs:
            run_step(
                [
                    recon_python,
                    str(steps_root / "step_2dgs.py"),
                    "--repo-root",
                    str(repo_root),
                    "--python-exec",
                    recon_python,
                    "--data-root",
                    str(dataset),
                    "--mesh-res",
                    str(args.mesh_res),
                ],
                cwd=repo_root,
                env=env,
            )

        if not args.skip_texture:
            run_step(
                [
                    texture_python,
                    str(steps_root / "step_texture.py"),
                    "--repo-root",
                    str(repo_root),
                    "--python-exec",
                    texture_python,
                    "--data-root",
                    str(dataset),
                    "--num-view",
                    str(args.num_view),
                ],
                cwd=repo_root,
                env=env,
            )

    if dataset_workers <= 1 or len(datasets) <= 1:
        for idx, dataset in enumerate(datasets):
            run_dataset_pipeline(idx, dataset)
        return

    # 数据集级并发：每个任务启动独立子进程，按GPU轮转绑定。
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(dataset_workers, len(datasets))) as executor:
        futures = [executor.submit(run_dataset_pipeline, idx, dataset) for idx, dataset in enumerate(datasets)]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
