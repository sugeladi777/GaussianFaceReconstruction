import argparse
import concurrent.futures
import multiprocessing
from pathlib import Path

import cv2


VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv'}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_dir', type=str, required=True, help='待处理视频所在目录，递归查找所有视频文件')
    p.add_argument('--data_root', type=str, required=True, help='输出图片根目录，<video_name>/image/')
    p.add_argument('--step_size', type=int, default=1, help='帧抽取间隔，1 表示每帧都取')
    p.add_argument('--workers', type=int, default=0, help='并行处理视频的进程数，0=自动')
    p.add_argument('--resize', nargs=2, type=int, metavar=('W', 'H'), default=(0, 0), help='缩放输出图片尺寸，0 0 表示不缩放')
    p.add_argument('--rotate', action='store_true', help='是否对图片顺时针旋转90度')
    return p


def iter_video_paths(video_dir: str):
    root = Path(video_dir)
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def process_video_file(video_path: str, out_dir: str, step_size: int, rotate: bool, resize):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    i = 0
    saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            i += 1
            if i % step_size:
                continue
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if resize[0] and resize[1]:
                frame = cv2.resize(frame, resize)
            saved += 1
            cv2.imwrite(str(out / f'{saved:05d}.png'), frame)
    finally:
        cap.release()

    print(f'Saved {saved} frames to {out}')


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    tasks = [
        (str(p), str(data_root / p.stem / 'image'), args.step_size, args.rotate, tuple(args.resize))
        for p in iter_video_paths(args.video_dir)
    ]

    if not tasks:
        print('未找到可处理的视频文件')
        return

    workers = args.workers or multiprocessing.cpu_count()
    workers = max(1, min(workers, len(tasks)))

    if workers == 1:
        for t in tasks:
            process_video_file(*t)
        return

    print(f'使用 {workers} 个进程进行帧抽取')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_video_file, *t) for t in tasks]
        for f in futures:
            f.result()


if __name__ == '__main__':
    main()
