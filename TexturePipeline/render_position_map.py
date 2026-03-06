"""位置图渲染脚本。

将网格在选定相机视角下渲染为：
- pointmap: 每像素 3D 位置（canonical）
- pointmap_mask: 可见性掩码
- pointmap_vis: 可视化图
"""

import argparse
import json
import os

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
    def __init__(self, device, meta_file_path, image_dir, syn=1, mode='train', cache_image=True):
        super().__init__()
        self.device = device
        self.mode = mode
        self.cache_image = cache_image
        self.image_dir = image_dir
        self.syn = syn

        with open(meta_file_path, 'r') as file:
            meta = json.load(file)

        self.height = int(meta['h'])
        self.width = int(meta['w'])

        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = meta['fl_x']
        intrinsic[1, 1] = meta['fl_y']
        intrinsic[0, 2] = meta['cx']
        intrinsic[1, 2] = meta['cy']
        self.intrinsic = torch.from_numpy(intrinsic).to(self.device)
        # 归一化内参用于 nvdiffrast 渲染接口。
        self.intrinsic_rel = torch.clone(self.intrinsic)
        self.intrinsic_rel[0] /= self.width
        self.intrinsic_rel[1] /= self.height

        frames = sorted(meta['frames'], key=lambda item: item['file_path'])
        if mode == 'val':
            frames = frames[::10]
        self.frames = frames
        self.num_frames = len(self.frames)

        self.cam2world = []
        self.images = []
        self.img_name_list = []
        for frame in tqdm(self.frames, desc=f'Load {mode} frames'):
            cur_pose = np.array(frame['transform_matrix'], dtype=np.float32)
            if self.syn == 0:
                cur_pose = nerf_matrix_to_ngp(cur_pose, scale=1.)
            self.cam2world.append(cur_pose)

            img_name = os.path.basename(frame['file_path'])
            self.img_name_list.append(img_name)
            if self.cache_image:
                self.images.append(self._load_img(os.path.join(self.image_dir, img_name)))
            else:
                self.images.append(os.path.join(self.image_dir, img_name))

        if self.cache_image:
            self.images = torch.stack(self.images, dim=0)

        self.cam2world = np.stack(self.cam2world, axis=0).astype(np.float32)
        self.cam2world = torch.from_numpy(self.cam2world).to(self.device)
        print(f'Create [{self.mode}] dataset, total [{self.num_frames}] frames')

    def __len__(self):
        return self.num_frames

    def _load_img(self, path):
        # 图片仅用于保持数据接口一致，不参与本脚本核心渲染计算。
        if os.path.exists(path):
            image = transforms.ToTensor()(Image.open(path))[:3]
            return image.to(self.device)
        return torch.zeros(3, self.height, self.width).to(self.device)

    def __getitem__(self, index):
        if self.cache_image:
            image = self.images[index]
        else:
            image = self._load_img(self.images[index])

        return {
            'pixels': image,
            'c2w': self.cam2world[index],
            'intrinsic': self.intrinsic,
            'intrinsic_rel': self.intrinsic_rel,
            'img_name': self.img_name_list[index],
        }


class PositionMapRenderer:
    def __init__(self, device, cam_path, mesh_path, img_root, save_root, syn=1):
        self.device = device
        self.mesh_renderer = NVMeshRenderer(self.device)

        dataset = MetaShapeDataset(
            device=self.device,
            meta_file_path=cam_path,
            image_dir=img_root,
            syn=syn,
            mode='train',
        )
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        self.height = dataset.height
        self.width = dataset.width
        # 加载网格并计算 canonical 归一化参数（用于后续纹理网络输入范围）。
        self._load_geometry(mesh_path)

        self.position_save_root = os.path.join(save_root, 'pointmap')
        self.position_mask_save_root = os.path.join(save_root, 'pointmap_mask')
        self.position_vis_save_root = os.path.join(save_root, 'pointmap_vis')
        os.makedirs(self.position_save_root, exist_ok=True)
        os.makedirs(self.position_mask_save_root, exist_ok=True)
        os.makedirs(self.position_vis_save_root, exist_ok=True)

    def _load_geometry(self, mesh_path):
        # 读取几何并准备批渲染时复用的顶点/面缓存。
        mesh = trimesh.load_mesh(mesh_path)
        vertices = torch.from_numpy(mesh.vertices).to(self.device).float()
        faces = torch.from_numpy(mesh.faces).to(self.device)
        self.vertices = vertices[None, ...]
        self.faces = faces[None, ...]

        offset = torch.mean(self.vertices, dim=1, keepdim=True)[0, 0]
        centered = self.vertices - offset
        scale = torch.max(centered, dim=1)[0] - torch.min(centered, dim=1)[0]
        scale = torch.max(scale)
        scale = 2 / scale

        self.canonical_offset = offset[..., None, None]
        self.canonical_scale = scale[..., None, None]

    def _build_mesh_dict(self, vertices, faces):
        """构建 nvdiffrast 渲染输入字典。"""
        attrs = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        return {
            'faces': faces[0].int(),
            'vertice': vertices,
            'attributes': attrs,
            'size': (self.height, self.width),
        }

    def render(self):
        # 对每个视角渲染 attribute 图，并拆分出 pointmap 与可见性 mask。
        for data in tqdm(self.dataloader, desc='Render position maps'):
            c2w = data['c2w']
            batch_size = c2w.shape[0]
            cam_ext = torch.inverse(c2w)[:, :3]
            cam_int = data['intrinsic_rel']

            faces = self.faces.repeat(batch_size, 1, 1)
            vertices = self.vertices.repeat(batch_size, 1, 1)
            img_name = os.path.splitext(data['img_name'][0])[0]

            mesh_dict = self._build_mesh_dict(vertices, faces)
            attr_img, _ = self.mesh_renderer.render_mesh(mesh_dict, cam_int, cam_ext)

            point_img = attr_img[:, :3]
            mask_img = attr_img[:, 3:4]
            point_img = (point_img - self.canonical_offset) * self.canonical_scale

            save_image(mask_img, os.path.join(self.position_mask_save_root, f'{img_name}.png'))
            torch.save(point_img, os.path.join(self.position_save_root, f'{img_name}.pkl'))
            save_image((point_img + 1) / 2, os.path.join(self.position_vis_save_root, f'{img_name}.jpg'))


def main():
    args = parse_args()
    renderer = PositionMapRenderer(
        device=args.device,
        cam_path=args.cam_path,
        mesh_path=args.mesh_path,
        img_root=args.img_root,
        save_root=args.save_root,
        syn=args.syn,
    )
    renderer.render()


if __name__ == '__main__':
    main()
