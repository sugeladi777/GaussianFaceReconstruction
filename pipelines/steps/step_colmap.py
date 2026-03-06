"""步骤3：COLMAP 相机与稀疏重建。"""

import argparse
import os
import subprocess
from pathlib import Path
import shutil
from typing import Dict, List


def run_step(args, cwd):
    print("[step_colmap]", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def build_colmap_env() -> Dict[str, str]:
    return os.environ.copy()


def run_colmap_cmd(cmd: List[str], env: Dict[str, str]) -> None:
    print("[step_colmap]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def build_colmap_commands(data_root: Path, use_mask: bool, num_threads: int, colmap_bin: str) -> Dict[str, List[str]]:
    img_root = data_root / "images"
    mask_root = data_root / "mask"
    database_root = data_root / "database.db"
    sparse_root = data_root / "sparse"
    recon_root = sparse_root / "0"

    feat_cmd = [
        colmap_bin, "feature_extractor",
        "--database_path", str(database_root),
        "--image_path", str(img_root),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.max_image_size", "4000",
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.num_threads", str(num_threads),
        "--SiftExtraction.max_num_features", "4096",
    ]
    if use_mask and mask_root.is_dir():
        feat_cmd += ["--ImageReader.mask_path", str(mask_root)]

    match_cmd = [
        colmap_bin, "exhaustive_matcher",
        "--database_path", str(database_root),
        "--FeatureMatching.guided_matching", "1",
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.num_threads", str(num_threads),
    ]

    sfm_cmd = [
        colmap_bin, "mapper",
        "--database_path", str(database_root),
        "--image_path", str(img_root),
        "--output_path", str(sparse_root),
        "--Mapper.num_threads", str(num_threads),
    ]

    ba_cmd = [
        colmap_bin, "bundle_adjuster",
        "--input_path", str(recon_root),
        "--output_path", str(recon_root),
        "--BundleAdjustmentCeres.max_num_iterations", "100",
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
    parser.add_argument("--use-mask", type=int, choices=[0, 1], default=0)
    parser.add_argument("--num-threads", type=int, default=0, help="COLMAP 使用的线程数；0 表示自动从 PIPELINE_WORKERS / PIPELINE_DATASET_WORKERS 计算")
    parser.add_argument("--colmap-bin", type=str, default="colmap")
    return parser


def main():
    args = build_parser().parse_args()
    data_root = Path(args.data_root).resolve()
    sparse_root = data_root / "sparse"

    use_mask = bool(args.use_mask)
    print("[step_colmap] mask mode:", "enabled" if use_mask else "disabled (default)")

    env = build_colmap_env()

    # 保证每次重建从干净状态开始，避免旧相机模型残留导致后续读取失败。
    database_path = data_root / "database.db"
    if database_path.exists():
        database_path.unlink()
    recon_root = sparse_root / "0"
    if recon_root.exists():
        shutil.rmtree(recon_root, ignore_errors=True)

    # 计算实际用于 COLMAP 的线程数：
    # - 如果通过 CLI 明确传入 (>0)，则使用该值。
    # - 如果为 0，则尝试读取 pipeline_runner 传入的环境变量并平均分配。
    requested = int(args.num_threads)
    if requested > 0:
        num_threads = max(1, requested)
    else:
        try:
            total_workers = int(os.environ.get("PIPELINE_WORKERS", "0"))
            dataset_workers = int(os.environ.get("PIPELINE_DATASET_WORKERS", "1"))
            workers_for_datasets = max(1, dataset_workers)
            num_threads = max(1, total_workers // workers_for_datasets) if total_workers > 0 else 1
        except Exception:
            num_threads = 1

    commands = build_colmap_commands(data_root, use_mask, num_threads, args.colmap_bin)

    sparse_root.mkdir(parents=True, exist_ok=True)
    run_colmap_cmd(commands["feat"], env)
    run_colmap_cmd(commands["match"], env)
    run_colmap_cmd(commands["sfm"], env)

    run_colmap_cmd(commands["ba"], env)
    run_colmap_cmd(commands["to_txt"], env)


if __name__ == "__main__":
    main()
