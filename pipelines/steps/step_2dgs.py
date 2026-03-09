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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _iter_num(path: str) -> int:
    name = Path(path).name
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return -1


def find_latest_recon_dir(data_root: Path) -> Path:
    train_root = data_root / "recon" / "train"
    dirs = list(train_root.glob("ours_*"))
    if not dirs:
        raise FileNotFoundError(f"No reconstruction folders found under: {train_root}")
    return max(dirs, key=_iter_num)


def export_main_component_mesh(mesh_path: Path, out_obj_path: Path) -> None:
    mesh = trimesh.load(mesh_path)
    parts = mesh.split(only_watertight=False)
    if parts:
        main = max(parts, key=lambda item: len(getattr(item, "faces", [])))
    else:
        main = mesh
    main.export(out_obj_path)


def build_transforms(camera_list: List[Dict], image_root: Path) -> Dict:
    if not camera_list:
        raise ValueError("camera_list is empty")
    first = camera_list[0]
    save_info = {
        "fl_x": first["fx"],
        "fl_y": first["fy"],
        "w": first["width"],
        "h": first["height"],
    }
    save_info["cx"] = save_info["w"] / 2
    save_info["cy"] = save_info["h"] / 2

    frames = []
    for cam in camera_list:
        mat = np.eye(4)
        mat[:3, :3] = np.asarray(cam["rotation"])
        mat[:3, 3] = np.asarray(cam["position"])
        frames.append({
            "file_path": str(image_root / f"{cam['img_name']}.png"),
            "transform_matrix": mat.tolist(),
        })

    save_info["frames"] = frames
    return save_info


def export_recon_assets(data_root: Path) -> None:
    latest_recon_dir = find_latest_recon_dir(data_root)
    mesh_path_pose = latest_recon_dir / "fuse_pose.ply"
    mesh_path = mesh_path_pose if mesh_path_pose.exists() else (latest_recon_dir / "fuse.ply")
    export_main_component_mesh(mesh_path, data_root / "2dgs_recon.obj")

    cam_path = data_root / "recon" / "cameras.json"
    with cam_path.open("r") as f:
        camera_list = json.load(f)

    image_root = data_root / "images"

    save_info = build_transforms(camera_list, image_root)
    with (data_root / "transforms.json").open("w") as f:
        json.dump(save_info, f, indent=4)


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
    two_d_gs_root = repo_root / "pipelines" / "steps" / "2d-gaussian-splatting"
    data_root = Path(args.data_root).resolve()

    recon_root = data_root / "recon"
    train_port = args.port if args.port > 0 else pick_free_port()


    train_py = two_d_gs_root / "train.py"
    run_step(
        [
            args.python_exec,
            str(train_py),
            "-s",
            str(data_root),
            "-m",
            str(recon_root),
            "--port",
            str(train_port),
        ],
        str(two_d_gs_root),
    )

    render_py = two_d_gs_root / "render.py"
    run_step(
        [
            args.python_exec,
            str(render_py),
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
