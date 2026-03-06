"""纹理场训练脚本。

训练目标：根据 pointmap 预测颜色，与图像监督一致。
损失：L1 + (后期)LPIPS
"""

import argparse
import os

import kornia
import lpips
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from network import VolumeTexture


def parse_args():
    # 纹理优化输入：图像、点图、掩码，以及训练可视化与迭代参数。
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str, default='data/hxl_indoor_dt_step-10/texture/sample/image')
    parser.add_argument('--pointmap_root', type=str, default='data/hxl_indoor_dt_step-10/texture/sample/pointmap')
    parser.add_argument('--save_root', type=str, default='data/hxl_indoor_dt_step-10/texture/log')
    parser.add_argument('--mask_root', type=str, default='data/hxl_indoor_dt_step-10/texture/sample/pointmap_mask')
    parser.add_argument('--vis_freq', type=int, default=25)
    parser.add_argument('--max_iter', type=int, default=151)
    return parser.parse_args()


class PointMapDataset(Dataset):
    def __init__(self, img_root, pointmap_root, mask_root, device):
        super().__init__()
        # 将有效样本整体加载到 GPU，训练时减少 IO 抖动。
        self.images = []
        self.pointmaps = []
        self.masks = []

        image_names = [
            name for name in sorted(os.listdir(img_root))
            if os.path.isfile(os.path.join(img_root, name))
        ]

        valid_image_names = []
        for name in image_names:
            img_path = os.path.join(img_root, name)
            mask_path = os.path.join(mask_root, name)
            pointmap_name = f"{os.path.splitext(name)[0]}.pkl"
            pointmap_path = os.path.join(pointmap_root, pointmap_name)

            if not os.path.exists(mask_path):
                print(f"[WARN] Missing mask: {mask_path}, skip")
                continue
            if not os.path.exists(pointmap_path):
                print(f"[WARN] Missing pointmap: {pointmap_path}, skip")
                continue

            image, mask = self._load_image_and_mask(img_path, mask_path)
            pointmap = torch.load(pointmap_path, map_location='cpu')

            self.images.append(image)
            self.masks.append(mask)
            self.pointmaps.append(pointmap)
            valid_image_names.append(name)

        if len(self.images) == 0:
            raise ValueError(
                'No valid training samples found. Check image/mask/pointmap alignment.'
            )

        self.images = torch.stack(self.images, dim=0).to(device)
        self.pointmaps = torch.cat(self.pointmaps, dim=0).to(device)
        self.masks = torch.stack(self.masks, dim=0).to(device)
        self.image_names = valid_image_names

        print(f"[INFO] Loaded {len(self.images)} valid samples")

    @staticmethod
    def _load_image_and_mask(img_path, mask_path):
        """图像/掩码统一为可训练格式（RGB + 单通道 mask + 尺寸对齐）。"""
        image = transforms.ToTensor()(Image.open(img_path))
        mask = transforms.ToTensor()(Image.open(mask_path))

        if image.shape[0] > 3:
            image = image[:3]

        if mask.shape[0] > 1:
            mask = mask[:1]

        if mask.shape[-2:] != image.shape[-2:]:
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=image.shape[-2:],
                mode='nearest',
            ).squeeze(0)

        mask = mask.float().clamp(0, 1)
        return image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            'img': self.images[idx],
            'pointmap': self.pointmaps[idx],
            'mask': self.masks[idx],
        }


class TextureOptimizer:
    def __init__(self, network, dataset, save_root, vis_freq=25, max_iter=151):
        # 优化器负责训练、评估和可视化落盘。
        self.device = torch.device('cuda')
        self.network = network.to(self.device)
        self.dataset = dataset
        self.save_root = save_root
        self.vis_freq = vis_freq
        self.max_iter = max_iter

        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.val_dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)
        self.lpips_loss.requires_grad_(False)

    def _predict(self, pointmap):
        # 将每个像素位置的 3D 点映射到颜色，再恢复为图像张量。
        b, _, h, w = pointmap.shape
        points = pointmap.permute(0, 2, 3, 1).reshape(b * h * w, 3)
        points = (points + 1) / 2
        render = self.network(points)
        pred = render.reshape(b, h, w, 3).permute(0, 3, 1, 2).contiguous()
        return pred

    def _train_step(self, data, optimizer, grad_scaler, step, writer):
        # 单步训练：mask 区域监督，早期用 L1，后期叠加 LPIPS。
        pred = self._predict(data['pointmap'])
        mask = data['mask']
        gt = data['img']

        img_pred = pred * mask
        img_gt = gt * mask

        loss = F.l1_loss(img_pred, img_gt)
        writer.add_scalar('L1 Loss', loss.item(), step)

        if step > 100:
            loss_lpips = self.lpips_loss(img_pred, img_gt, normalize=True).mean()
            loss = loss + 0.1 * loss_lpips
            writer.add_scalar('LPIPS Loss', loss.item(), step)

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

    def _evaluate_and_log(self, step, writer, log_dir):
        # 按当前模型在全验证集计算指标并导出可视化图。
        self.network.eval()
        with torch.no_grad():
            render_list = []
            gt_list = []
            psnr_list = []
            ssim_list = []
            lpips_list = []

            for data in self.val_dataloader:
                pred = self._predict(data['pointmap'])
                mask = data['mask']
                gt = data['img']

                img_pred = pred * mask
                img_gt = gt * mask

                render_list.append(img_pred)
                gt_list.append(img_gt)

                psnr_list.append(kornia.metrics.psnr(img_pred, img_gt, max_val=1.).item())
                ssim_list.append(
                    kornia.metrics.ssim(img_pred, img_gt, window_size=3).mean().item()
                )
                lpips_list.append(
                    self.lpips_loss(img_pred, img_gt, normalize=True).mean().item()
                )

            render_tensor = torch.cat(render_list, dim=0)
            gt_tensor = torch.cat(gt_list, dim=0)
            save_image(render_tensor, os.path.join(log_dir, 'render.jpg'))
            save_image(gt_tensor, os.path.join(log_dir, 'gt.jpg'))

            psnr_score = sum(psnr_list) / len(psnr_list)
            ssim_score = sum(ssim_list) / len(ssim_list)
            lpips_score = sum(lpips_list) / len(lpips_list)

            writer.add_scalar('ssim', ssim_score, step)
            writer.add_scalar('psnr', psnr_score, step)
            writer.add_scalar('lpips', lpips_score, step)
            print(
                '[iter %05d][PSNR %.4f][SSIM %.4f][LPIPS %.4f]'
                % (step, psnr_score, ssim_score, lpips_score)
            )
        self.network.train()

    def optimize(self):
        # 主训练循环：按 vis_freq 定期保存 checkpoint + 评估结果。
        os.makedirs(self.save_root, exist_ok=True)
        writer = SummaryWriter(self.save_root)

        optimizer = torch.optim.Adam([{'params': self.network.parameters()}], lr=0.01)
        grad_scaler = torch.cuda.amp.GradScaler(2 ** 10)

        self.network.train()
        for step in tqdm(range(self.max_iter)):
            for data in self.dataloader:
                self._train_step(data, optimizer, grad_scaler, step, writer)

            if step % self.vis_freq == 0:
                torch.save(
                    self.network.state_dict(),
                    os.path.join(self.save_root, 'latest.pth'),
                )
                cur_log_dir = os.path.join(self.save_root, f'{step:05d}')
                os.makedirs(cur_log_dir, exist_ok=True)
                self._evaluate_and_log(step, writer, cur_log_dir)


def main():
    # 构建数据集与网络，启动纹理优化。
    opt = parse_args()
    device = torch.device('cuda')
    dataset = PointMapDataset(
        img_root=opt.img_root,
        pointmap_root=opt.pointmap_root,
        mask_root=opt.mask_root,
        device=device,
    )
    network = VolumeTexture()
    optimizer = TextureOptimizer(
        network=network,
        dataset=dataset,
        save_root=opt.save_root,
        vis_freq=opt.vis_freq,
        max_iter=opt.max_iter,
    )
    optimizer.optimize()


if __name__ == '__main__':
    main()
