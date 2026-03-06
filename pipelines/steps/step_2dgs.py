"""步骤4：2DGS 训练与网格导出。"""

import argparse
import glob
import json
import socket
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import trimesh


def run_step(args, cwd):
    print("[step_2dgs]", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def pick_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _iter_num(path: str) -> int:
    name = Path(path).name
    try:
        return int(name.split("_")[-1])
    except Exception:
        return -1


def find_latest_recon_dir(data_root: Path) -> Path:
    train_root = data_root / "recon" / "train"
    ours_dirs = glob.glob(str(train_root / "ours_*"))
    if not ours_dirs:
        raise FileNotFoundError(f"No reconstruction folders found under: {train_root}")
    return Path(sorted(ours_dirs, key=_iter_num)[-1])


def pick_mesh_path(latest_recon_dir: Path) -> Path:
    mesh_candidates = [
        latest_recon_dir / "fuse_post.ply",
        latest_recon_dir / "fuse.ply",
        latest_recon_dir / "fuse_unbounded_post.ply",
        latest_recon_dir / "fuse_unbounded.ply",
    ]
    for candidate in mesh_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No mesh found in latest reconstruction folder: {latest_recon_dir}")


def export_main_component_mesh(mesh_path: Path, out_obj_path: Path) -> None:
    mesh = trimesh.load(mesh_path)
    split_all = mesh.split(only_watertight=False)
    mesh = sorted(split_all, key=lambda item: len(item.faces))[-1]
    mesh.export(out_obj_path)


def build_transforms(camera_list: List[Dict], image_root: Path) -> Dict:
    save_info = {
        "fl_x": camera_list[0]["fx"],
        "fl_y": camera_list[0]["fy"],
        "w": camera_list[0]["width"],
        "h": camera_list[0]["height"],
    }
    save_info["cx"] = save_info["w"] / 2
    save_info["cy"] = save_info["h"] / 2

    frames = []
    for cam_info in camera_list:
        transforms = np.eye(4)
        transforms[:3, :3] = cam_info["rotation"]
        transforms[:3, 3] = cam_info["position"]
        frames.append(
            {
                "file_path": str(image_root / f"{cam_info['img_name']}.png"),
                "transform_matrix": transforms.tolist(),
            }
        )

    save_info["frames"] = frames
    return save_info


def export_recon_assets(data_root: Path) -> None:
    latest_recon_dir = find_latest_recon_dir(data_root)
    mesh_path = pick_mesh_path(latest_recon_dir)
    export_main_component_mesh(mesh_path, data_root / "2dgs_recon.obj")

    cam_path = data_root / "recon" / "cameras.json"
    with open(cam_path, "r") as file:
        camera_list = json.load(file)

    image_root = data_root / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    save_info = build_transforms(camera_list, image_root)
    with open(data_root / "transforms.json", "w") as file:
        json.dump(save_info, file, indent=4)


def build_parser():
    parser = argparse.ArgumentParser(description="Step 4: run 2DGS train/render")
    parser.add_argument("--repo-root", type=str, required=True)
    parser.add_argument("--python-exec", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--mesh-res", type=int, default=1024)
    parser.add_argument("--port", type=int, default=0, help="0 means auto-pick free port")
    return parser


def main():
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    two_d_gs_root = repo_root / "2d-gaussian-splatting"
    data_root = Path(args.data_root).resolve()

    recon_root = data_root / "recon"
    train_port = args.port if args.port > 0 else pick_free_port()

    run_step(
        [
            args.python_exec,
            "train.py",
            "-s",
            str(data_root),
            "-m",
            str(recon_root),
            "--port",
            str(train_port),
        ],
        str(two_d_gs_root),
    )

    run_step(
        [
            args.python_exec,
            "render.py",
            "-s",
            str(data_root),
            "-m",
            str(recon_root),
            "--mesh_res",
            str(args.mesh_res),
        ],
        str(two_d_gs_root),
    )

    export_recon_assets(data_root)


if __name__ == "__main__":
    main()
