"""基于清晰度的关键帧选择。

输入：完整图像目录 + transforms 相机文件
输出：sample/image 子集图像 + select_sharp.json 子集相机
"""

import argparse
import json
import math
import os
import shutil

import cv2
import torch
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
    # 用拉普拉斯响应均值近似图像清晰度，值越大通常越清晰。
    image = cv2.imread(image_path, 0)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    lap = cv2.Laplacian(image, cv2.CV_8UC1, ksize=3)
    return cv2.mean(lap)[0]


def build_sharpness_map(img_root):
    # 预先计算每张图的清晰度，供分段选优使用。
    image_names = [
        name for name in sorted(os.listdir(img_root))
        if os.path.isfile(os.path.join(img_root, name))
    ]
    sharpness = {}
    for name in tqdm(image_names, desc='Compute sharpness'):
        sharpness[name] = compute_sharpness(os.path.join(img_root, name))
    return sharpness


def subset_camera_frames(frames, selected_names):
    """从原 frames 中提取与 selected_names 同名的子集。"""
    frame_dict = {os.path.basename(frame['file_path']): frame for frame in frames}
    new_frames = []
    for image_name in selected_names:
        if image_name in frame_dict:
            new_frames.append(frame_dict[image_name])
        else:
            print(f'Warning: {image_name} not found in frames, skipping')
    return new_frames


def select_and_copy_images(raw_frame_root, selected_root, sharp_info, interval):
    # 在每个时间段内挑选最清晰的一帧并复制到 sample/image。
    os.makedirs(selected_root, exist_ok=True)
    image_names = [
        name for name in sorted(os.listdir(raw_frame_root))
        if os.path.isfile(os.path.join(raw_frame_root, name))
    ]

    selected_names = []
    num_images = len(image_names)
    left = 0
    while left < num_images:
        right = min(left + interval, num_images)
        current_chunk = image_names[left:right]
        current_chunk = [name for name in current_chunk if name in sharp_info]

        if current_chunk:
            best_name = max(current_chunk, key=lambda name: sharp_info[name])
            shutil.copy2(
                os.path.join(raw_frame_root, best_name),
                os.path.join(selected_root, best_name),
            )
            selected_names.append(best_name)

        left = right

    return selected_names


def main():
    args = parse_args()
    # 1) 计算并保存清晰度统计（调试/复现可用）。
    sharpness = build_sharpness_map(args.img_root)
    os.makedirs(args.save_root, exist_ok=True)
    torch.save(sharpness, os.path.join(args.save_root, 'sharpness.pkl'))

    # 2) 读取相机轨迹，并按目标视角数计算采样间隔。
    with open(args.cam_path, 'r') as file:
        meta = json.load(file)
    frames = meta.get('frames', [])
    if len(frames) == 0:
        raise ValueError(f'No frames found in camera file: {args.cam_path}')

    interval = max(1, math.ceil(len(frames) / args.num_view))
    selected_image_root = os.path.join(args.save_root, 'image')
    selected_names = select_and_copy_images(
        raw_frame_root=args.img_root,
        selected_root=selected_image_root,
        sharp_info=sharpness,
        interval=interval,
    )

    # 3) 只保留已选图像对应的相机条目，形成下游使用的精简相机文件。
    meta['frames'] = subset_camera_frames(frames, selected_names)
    with open(os.path.join(args.save_root, 'select_sharp.json'), 'w') as outfile:
        json.dump(meta, outfile, indent=4)


if __name__ == '__main__':
    main()
