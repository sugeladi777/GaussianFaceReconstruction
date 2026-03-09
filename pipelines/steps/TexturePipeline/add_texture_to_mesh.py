"""将训练好的纹理网络烘焙到网格顶点颜色。"""

import argparse
import numpy as np
import torch
import trimesh

from network import VolumeTexture


def parse_args():
    # 输入网格与训练好的纹理网络参数，输出带顶点颜色的 mesh。
    import os
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent.parent.parent  # 定位到RGB_Recon根目录
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', type=str, default=str(script_dir / 'output/test/2dgs_recon.obj'))
    parser.add_argument('--save_path', type=str, default=str(script_dir / 'output/test/texture/textured_mesh.obj'))
    parser.add_argument('--ckpt_path', type=str, default=str(script_dir / 'output/test/texture/texture/latest.pth'))
    return parser.parse_args()


def normalize_vertices(vertices):
    # 将顶点归一化到近似 [-1, 1]，与训练阶段 pointmap 规范一致。
    offset = torch.mean(vertices, dim=0, keepdim=False)
    can_vertices = vertices - offset
    scale = torch.max(can_vertices, dim=0)[0] - torch.min(can_vertices, dim=0)[0]
    scale = torch.max(scale)
    scale = 2 / scale
    return can_vertices * scale


def load_geometry(mesh_path):
    # 读取网格并转换为网络推理使用的 canonical 坐标。
    mesh = trimesh.load_mesh(mesh_path)
    vertices = torch.from_numpy(mesh.vertices).cuda().float()
    can_vertices = normalize_vertices(vertices)
    return can_vertices, mesh


def bake_vertex_color(network, can_vertices):
    # 用纹理网络直接预测每个顶点颜色。
    network = network.eval()
    with torch.no_grad():
        color = network((can_vertices + 1) / 2).cpu().numpy()
        color = np.clip(color, 0, 1)
    return color


def export_colored_mesh(mesh, color, save_path):
    # 写回 OBJ：保持几何不变，仅附加 vertex color。
    trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_colors=(color * 255).astype(np.uint8),
    ).export(save_path)


def main():
    # 加载网络权重，执行顶点烘焙并导出。
    args = parse_args()
    network = VolumeTexture().cuda()
    network.load_state_dict(torch.load(args.ckpt_path))

    can_vertices, mesh = load_geometry(args.mesh_path)
    color = bake_vertex_color(network, can_vertices)
    export_colored_mesh(mesh, color, args.save_path)


if __name__ == '__main__':
    main()
