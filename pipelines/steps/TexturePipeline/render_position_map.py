"""位置图渲染脚本。

将网格在选定相机视角下渲染为：
- pointmap: 每像素 3D 位置（canonical）
- pointmap_mask: 可见性掩码
- pointmap_vis: 可视化图
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from mesh_renderer_nv import RGBDRenderer as NVMeshRenderer


def parse_args():
    # 读取相机与网格，渲染每个视角的 pointmap / pointmap_mask。
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--syn', type=int, default=1)
    parser.add_argument('--cam_path', type=str, default='data/hxl_indoor_dt_step-10/texture/sample/select_sharp.json')
    parser.add_argument('--mesh_path', type=str, default='data/hxl_indoor_dt_step-10/2dgs_recon.obj')
    parser.add_argument('--img_root', type=str, default='data/hxl_indoor_dt_step-10/texture/sample/image')
    parser.add_argument('--save_root', type=str, default='data/hxl_indoor_dt_step-10/texture/sample')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def nerf_matrix_to_ngp(pose, scale=0.33):
    # 将 NeRF 风格位姿转换到当前渲染器坐标系。
    return np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=pose.dtype)


class MetaShapeDataset(Dataset):
    def __init__(self, device, meta_file_path, syn=1):
        super().__init__()
        self.device = device
        self.syn = syn
        meta = json.loads(Path(meta_file_path).read_text())
        self.height = int(meta['h'])
        self.width = int(meta['w'])
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = meta['fl_x']
        intrinsic[1, 1] = meta['fl_y']
        intrinsic[0, 2] = meta['cx']
        intrinsic[1, 2] = meta['cy']
        self.intrinsic = torch.from_numpy(intrinsic).to(self.device)
        self.intrinsic_rel = self.intrinsic.clone()
        self.intrinsic_rel[0] /= self.width
        self.intrinsic_rel[1] /= self.height
        self.frames = sorted(meta['frames'], key=lambda item: item['file_path'])
        self.cam2world = []
        self.img_name_list = []
        for frame in self.frames:
            cur_pose = np.array(frame['transform_matrix'], dtype=np.float32)
            if self.syn == 0:
                cur_pose = nerf_matrix_to_ngp(cur_pose, scale=1.)
            self.cam2world.append(cur_pose)
            self.img_name_list.append(Path(frame['file_path']).name)
        self.cam2world = torch.from_numpy(np.stack(self.cam2world, axis=0)).to(self.device)
    def __len__(self):
        return len(self.frames)
    def __getitem__(self, index):
        return {
            'c2w': self.cam2world[index],
            'intrinsic': self.intrinsic,
            'intrinsic_rel': self.intrinsic_rel,
            'img_name': self.img_name_list[index],
        }



class PositionMapRenderer:
    def __init__(self, device, cam_path, mesh_path, save_root, syn=1):
        self.device = device
        self.mesh_renderer = NVMeshRenderer(self.device)
        dataset = MetaShapeDataset(device, cam_path, syn)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.height = dataset.height
        self.width = dataset.width
        self._load_geometry(mesh_path)
        save_root = Path(save_root)
        self.position_save_root = save_root / 'pointmap'
        self.position_mask_save_root = save_root / 'pointmap_mask'
        self.position_vis_save_root = save_root / 'pointmap_vis'
        self.position_save_root.mkdir(parents=True, exist_ok=True)
        self.position_mask_save_root.mkdir(parents=True, exist_ok=True)
        self.position_vis_save_root.mkdir(parents=True, exist_ok=True)

    def _load_geometry(self, mesh_path):
        mesh = trimesh.load_mesh(mesh_path)
        v = torch.from_numpy(mesh.vertices).to(self.device).float()
        f = torch.from_numpy(mesh.faces).to(self.device)
        self.vertices = v[None, ...]
        self.faces = f[None, ...]
        offset = self.vertices.mean(dim=1, keepdim=True)
        centered = self.vertices - offset
        scale = (centered.max(dim=1)[0] - centered.min(dim=1)[0]).max()
        scale = 2 / scale
        # Keep canonical params 4D ([B, C, 1, 1]) to avoid broadcasting to 5D.
        self.canonical_offset = offset.squeeze(1)[..., None, None]
        self.canonical_scale = scale

    def _build_mesh_dict(self, vertices, faces):
        attrs = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        return {
            'faces': faces[0].int(),
            'vertice': vertices,
            'attributes': attrs,
            'size': (self.height, self.width),
        }

    def render(self):
        for i, data in enumerate(tqdm(self.dataloader, desc='Render position maps')):
            c2w = data['c2w']
            cam_ext = torch.inverse(c2w)[:, :3]
            cam_int = data['intrinsic_rel']
            faces = self.faces.repeat(1, 1, 1)
            vertices = self.vertices.repeat(1, 1, 1)
            img_name = Path(data['img_name'][0]).stem
            mesh_dict = self._build_mesh_dict(vertices, faces)
            attr_img, _ = self.mesh_renderer.render_mesh(mesh_dict, cam_int, cam_ext)
            point_img = attr_img[:, :3]
            mask_img = attr_img[:, 3:4]
            point_img = (point_img - self.canonical_offset) * self.canonical_scale
            if point_img.dim() == 5 and point_img.size(1) == 1:
                point_img = point_img.squeeze(1)
            save_image(mask_img, self.position_mask_save_root / f'{img_name}.png')
            torch.save(point_img, self.position_save_root / f'{img_name}.pkl')
            save_image((point_img + 1) / 2, self.position_vis_save_root / f'{img_name}.jpg')



def main():
    args = parse_args()
    renderer = PositionMapRenderer(
        device=args.device,
        cam_path=args.cam_path,
        mesh_path=args.mesh_path,
        save_root=args.save_root,
        syn=args.syn,
    )
    renderer.render()


if __name__ == '__main__':
    main()
