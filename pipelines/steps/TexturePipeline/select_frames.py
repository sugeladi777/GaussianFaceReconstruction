"""基于清晰度的关键帧选择。

输入：完整图像目录 + transforms 相机文件
输出：sample/image 子集图像 + select_sharp.json 子集相机
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import cv2
import torch
import numpy as np
from tqdm import tqdm


def parse_args():
    # 输入原始图像目录与相机 json，输出抽样后的图像与对应相机子集。
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--img_root', type=str, required=True)
    parser.add_argument('--cam_path', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--num_view', type=int, default=16)
    return parser.parse_args()


def compute_sharpness(image_path):
    # 用拉普拉斯方差近似图像清晰度，值越大通常越清晰。
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return float(np.var(cv2.Laplacian(image, cv2.CV_64F)))


def build_sharpness_map(img_root, image_names):
    # 只计算给定图片的清晰度，避免多余遍历。
    sharpness = {}
    for name in tqdm(image_names, desc='Compute sharpness'):
        sharpness[name] = compute_sharpness(img_root / name)
    return sharpness


def subset_camera_frames(frames, selected_names):
    # 只保留被选中的帧，顺序与采样一致。
    frame_dict = {Path(frame['file_path']).name: frame for frame in frames}
    return [frame_dict[name] for name in selected_names if name in frame_dict]


def select_and_copy_images(img_root, selected_root, sharp_info, frames, interval):
    # 按 frames 顺序采样，保证相机与图片一一对应。
    img_root = Path(img_root)
    selected_root = Path(selected_root)
    selected_root.mkdir(parents=True, exist_ok=True)
    image_names = [Path(frame['file_path']).name for frame in frames]
    selected_names = []
    for i in range(0, len(image_names), interval):
        chunk = image_names[i:i+interval]
        chunk = [name for name in chunk if name in sharp_info]
        if chunk:
            best_name = max(chunk, key=lambda name: sharp_info[name])
            shutil.copy2(img_root / best_name, selected_root / best_name)
            selected_names.append(best_name)
    return selected_names



def main():
    args = parse_args()
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    # 读取相机轨迹，按 frames 顺序采样，保证图片与相机一一对应。
    with open(args.cam_path, 'r') as file:
        meta = json.load(file)
    frames = meta.get('frames', [])
    image_names = [Path(frame['file_path']).name for frame in frames]

    # 计算清晰度
    sharpness = build_sharpness_map(Path(args.img_root), image_names)
    torch.save(sharpness, save_root / 'sharpness.pkl')

    interval = max(1, math.ceil(len(frames) / args.num_view))
    selected_image_root = save_root / 'image'
    selected_names = select_and_copy_images(
        img_root=args.img_root,
        selected_root=selected_image_root,
        sharp_info=sharpness,
        frames=frames,
        interval=interval,
    )

    # 只保留已选图像对应的相机条目
    meta['frames'] = subset_camera_frames(frames, selected_names)
    with (save_root / 'select_sharp.json').open('w') as outfile:
        json.dump(meta, outfile, indent=4)


if __name__ == '__main__':
    main()
