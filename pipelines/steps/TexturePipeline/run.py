"""TexturePipeline 总入口。

阶段顺序：
1) 选帧（select_frames.py）
2) 渲染位置图（render_position_map.py）
3) 优化纹理网络（build_texture.py）
4) 导出带颜色网格（add_texture_to_mesh.py）
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args():
    # 输入资产：原图、网格、相机、输出目录。
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--cam_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--num_view", type=int, default=16)
    return parser.parse_args()


def run_step(code_root, step_args):
    # 统一子进程调用，便于阶段日志追踪。
    print("[TexturePipeline]", " ".join(step_args))
    subprocess.run(step_args, cwd=code_root, check=True)


def run_stage(code_root: str, python_path: str, title: str, args: List[str]) -> None:
    """打印阶段标题并执行对应脚本。"""
    print("=" * 50)
    print(title)
    print("=" * 50)
    run_step(code_root, [python_path, *args])


def main():
    # 执行顺序：选帧 -> 位置图渲染 -> 纹理优化 -> 网格导出。
    args = parse_args()
    os.makedirs(args.save_root, exist_ok=True)

    code_root = str(Path(__file__).resolve().parent)
    python_path = sys.executable

    sample_data_root = os.path.join(args.save_root, "sample")
    tex_log_root = os.path.join(args.save_root, "texture")
    os.makedirs(sample_data_root, exist_ok=True)

    run_stage(
        code_root,
        python_path,
        "Sample Dataset by Sharpness ...",
        [
            "select_frames.py",
            "--img_root",
            args.img_root,
            "--cam_path",
            args.cam_path,
            "--save_root",
            sample_data_root,
            "--num_view",
            str(args.num_view),
        ],
    )

    run_stage(
        code_root,
        python_path,
        "Render Position Maps ...",
        [
            "render_position_map.py",
            "--mesh_path",
            args.mesh_path,
            "--cam_path",
            os.path.join(sample_data_root, "select_sharp.json"),
            "--img_root",
            args.img_root,
            "--save_root",
            sample_data_root,
        ],
    )

    run_stage(
        code_root,
        python_path,
        "Optimizing 3D Texture ...",
        [
            "build_texture.py",
            "--img_root",
            os.path.join(sample_data_root, "image"),
            "--pointmap_root",
            os.path.join(sample_data_root, "pointmap"),
            "--mask_root",
            os.path.join(sample_data_root, "pointmap_mask"),
            "--save_root",
            tex_log_root,
        ],
    )

    run_stage(
        code_root,
        python_path,
        "Save Textured Mesh ...",
        [
            "add_texture_to_mesh.py",
            "--mesh_path",
            args.mesh_path,
            "--save_path",
            os.path.join(args.save_root, "textured_mesh.obj"),
            "--ckpt_path",
            os.path.join(tex_log_root, "latest.pth"),
        ],
    )


if __name__ == "__main__":
    main()
