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

    feat_cmd = [
        colmap_bin, "feature_extractor",
        "--database_path", str(database_root),
        "--image_path", str(img_root),
        "--ImageReader.camera_model", args.camera_model,
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.max_image_size", str(args.max_image_size),
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.num_threads", str(num_threads),
        "--SiftExtraction.max_num_features", str(args.sift_max_num_features),
        "--SiftExtraction.peak_threshold", str(args.sift_peak_threshold),
        "--SiftExtraction.edge_threshold", str(args.sift_edge_threshold),
        "--SiftExtraction.estimate_affine_shape", str(args.sift_estimate_affine_shape),
        "--SiftExtraction.domain_size_pooling", str(args.sift_domain_size_pooling),
    ]
    if use_mask and mask_root.is_dir():
        feat_cmd += ["--ImageReader.mask_path", str(mask_root)]

    if args.matcher == "sequential":
        match_cmd = [
            colmap_bin, "sequential_matcher",
            "--database_path", str(database_root),
            "--FeatureMatching.guided_matching", "1",
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.num_threads", str(num_threads),
            "--SequentialMatching.overlap", str(args.sequential_overlap),
            "--SequentialMatching.quadratic_overlap", "1",
            "--SequentialMatching.loop_detection", str(args.sequential_loop_detection),
        ]
    else:
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
        "--Mapper.min_num_matches", str(args.mapper_min_num_matches),
        "--Mapper.init_min_num_inliers", str(args.mapper_init_min_num_inliers),
        "--Mapper.abs_pose_min_num_inliers", str(args.mapper_abs_pose_min_num_inliers),
        "--Mapper.abs_pose_min_inlier_ratio", str(args.mapper_abs_pose_min_inlier_ratio),
        "--Mapper.filter_max_reproj_error", str(args.mapper_filter_max_reproj_error),
        "--Mapper.filter_min_tri_angle", str(args.mapper_filter_min_tri_angle),
        "--Mapper.tri_ignore_two_view_tracks", str(args.mapper_tri_ignore_two_view_tracks),
        "--Mapper.ba_refine_principal_point", str(args.mapper_ba_refine_principal_point),
    ]

    ba_cmd = [
        colmap_bin, "bundle_adjuster",
        "--input_path", str(recon_root),
        "--output_path", str(recon_root),
        "--BundleAdjustmentCeres.max_num_iterations", str(args.ba_max_num_iterations),
        "--BundleAdjustment.refine_principal_point", str(args.mapper_ba_refine_principal_point),
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

    parser.add_argument("--camera-model", type=str, choices=["PINHOLE"], default="PINHOLE")
    parser.add_argument("--max-image-size", type=int, default=3200)
    parser.add_argument("--sift-max-num-features", type=int, default=8192)
    parser.add_argument("--sift-peak-threshold", type=float, default=0.004)
    parser.add_argument("--sift-edge-threshold", type=float, default=10.0)
    parser.add_argument("--sift-estimate-affine-shape", type=int, choices=[0, 1], default=1)
    parser.add_argument("--sift-domain-size-pooling", type=int, choices=[0, 1], default=1)

    parser.add_argument("--matcher", type=str, choices=["sequential", "exhaustive"], default="sequential")
    parser.add_argument("--sequential-overlap", type=int, default=30)
    parser.add_argument("--sequential-loop-detection", type=int, choices=[0, 1], default=1)

    parser.add_argument("--mapper-min-num-matches", type=int, default=20)
    parser.add_argument("--mapper-init-min-num-inliers", type=int, default=60)
    parser.add_argument("--mapper-abs-pose-min-num-inliers", type=int, default=30)
    parser.add_argument("--mapper-abs-pose-min-inlier-ratio", type=float, default=0.20)
    parser.add_argument("--mapper-filter-max-reproj-error", type=float, default=4.0)
    parser.add_argument("--mapper-filter-min-tri-angle", type=float, default=1.0)
    parser.add_argument("--mapper-tri-ignore-two-view-tracks", type=int, choices=[0, 1], default=0)
    parser.add_argument("--mapper-ba-refine-principal-point", type=int, choices=[0, 1], default=0)
    parser.add_argument("--ba-max-num-iterations", type=int, default=200)
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
        f"matcher={args.matcher}",
        f"use_mask={int(use_mask)}",
        f"threads={num_threads}",
    )

    commands = build_colmap_commands(data_root, use_mask, num_threads, args.colmap_bin, args)

    sparse_root.mkdir(parents=True, exist_ok=True)
    for step in ("feat", "match", "sfm", "ba", "to_txt"):
        run_colmap_cmd(commands[step], env=env, cwd=str(data_root))


if __name__ == "__main__":
    main()
