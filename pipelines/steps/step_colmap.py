"""步骤3：COLMAP 相机与稀疏重建。"""

import argparse
import os
import subprocess
from pathlib import Path
import shutil
from typing import Dict, List


def run_colmap_cmd(cmd: List[str], env: Dict[str, str] = None, cwd: str = None) -> None:
    """Run a COLMAP command with optional env and working directory.

    Prints the command and raises CalledProcessError on failure.
    """
    env = os.environ.copy() if env is None else env
    print("[step_colmap]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env, cwd=cwd)
    except subprocess.CalledProcessError as exc:
        print(f"[step_colmap] 命令失败: {' '.join(cmd)} (exit {exc.returncode})")
        raise


def build_colmap_commands(
    data_root: Path,
    use_mask: bool,
    num_threads: int,
    colmap_bin: str,
    args,
) -> Dict[str, List[str]]:
    img_root = data_root / "images"
    mask_root = data_root / "mask"
    database_root = data_root / "database.db"
    sparse_root = data_root / "sparse"
    recon_root = sparse_root / "0"

    # Use STFR-style defaults (fixed, not exposed): larger max image size, GPU on
    feat_cmd = [
        colmap_bin, "feature_extractor",
        "--database_path", str(database_root),
        "--image_path", str(img_root),
        "--ImageReader.camera_model", args.camera_model,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_image_size", "4000",
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.num_threads", str(num_threads),
    ]
    if use_mask and mask_root.is_dir():
        feat_cmd += ["--ImageReader.mask_path", str(mask_root)]

    # STFR uses exhaustive matching by default (more robust for faces)
    match_cmd = [
        colmap_bin, "exhaustive_matcher",
        "--database_path", str(database_root),
        "--FeatureMatching.guided_matching", "1",
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.num_threads", str(num_threads),
    ]

    # Keep mapper invocation minimal as in STFR
    sfm_cmd = [
        colmap_bin, "mapper",
        "--database_path", str(database_root),
        "--image_path", str(img_root),
        "--output_path", str(sparse_root),
        "--Mapper.num_threads", str(num_threads),
    ]

    # Use STFR BA settings
    ba_cmd = [
        colmap_bin, "bundle_adjuster",
        "--input_path", str(recon_root),
        "--output_path", str(recon_root),
        "--BundleAdjustment.max_num_iterations", "100",
    ]

    to_txt_cmd = [
        colmap_bin, "model_converter",
        "--input_path", str(recon_root),
        "--output_path", str(recon_root),
        "--output_type", "TXT",
    ]

    return {
        "feat": feat_cmd,
        "match": match_cmd,
        "sfm": sfm_cmd,
        "ba": ba_cmd,
        "to_txt": to_txt_cmd,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Step 3: run COLMAP")
    parser.add_argument("--repo-root", type=str, required=True)
    parser.add_argument("--python-exec", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--use-mask", type=int, choices=[0, 1], default=1)
    parser.add_argument("--num-threads", type=int, default=0, help="COLMAP 使用的线程数；0 表示自动从 PIPELINE_WORKERS / PIPELINE_DATASET_WORKERS 计算")
    parser.add_argument("--colmap-bin", type=str, default="colmap")

    parser.add_argument("--camera-model", type=str, choices=["PINHOLE"], default="PINHOLE")
    return parser


def main():
    args = build_parser().parse_args()
    data_root = Path(args.data_root).resolve()
    sparse_root = data_root / "sparse"

    use_mask = bool(args.use_mask)

    env = os.environ.copy()

    database_path = data_root / "database.db"
    if database_path.exists():
        database_path.unlink()
    recon_root = sparse_root / "0"
    shutil.rmtree(recon_root, ignore_errors=True)

    requested = int(args.num_threads)
    if requested > 0:
        num_threads = max(1, requested)
    else:
        total_workers = int(os.environ.get("PIPELINE_WORKERS", "0"))
        dataset_workers = int(os.environ.get("PIPELINE_DATASET_WORKERS", "1")) or 1
        if total_workers > 0:
            num_threads = max(1, total_workers // max(1, dataset_workers))
        else:
            import multiprocessing as _mp

            num_threads = max(1, _mp.cpu_count())

    print(
        "[step_colmap] config:",
        f"camera_model={args.camera_model}",
        "matcher=exhaustive",
        f"use_mask={int(use_mask)}",
        f"threads={num_threads}",
    )

    commands = build_colmap_commands(data_root, use_mask, num_threads, args.colmap_bin, args)

    sparse_root.mkdir(parents=True, exist_ok=True)
    for step in ("feat", "match", "sfm", "ba", "to_txt"):
        run_colmap_cmd(commands[step], env=env, cwd=str(data_root))


if __name__ == "__main__":
    main()
