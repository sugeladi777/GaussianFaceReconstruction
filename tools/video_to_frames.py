import cv2
import os
import argparse
import concurrent.futures
import multiprocessing


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--video_path', type=str, default=None, help='Single video file to process')
parser.add_argument('--video_dir', type=str, default="/home/lichengkai/RGB_Recon/input/20251011_night", help='Directory containing videos to process')
parser.add_argument('--data_root', type=str, default="/home/lichengkai/RGB_Recon/input/20251011_night", help='Output base directory where images/<video_name>/ will be created')
parser.add_argument('--step_size', default=1, type=int)
parser.add_argument('--workers', default=0, type=int, help='Parallel workers for multiple videos (0=cpu count)')
parser.add_argument('--resize', nargs=2, type=int, metavar=('W', 'H'), default=(0, 0), help='Resize frames to W H, pass 0 0 to keep original')
parser.add_argument('--rotate', action='store_true', help='Rotate frames clockwise 90 degrees (used for some MOV videos)')
opt, _ = parser.parse_known_args()

VIDEO_EXTS = ['.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV']

def is_video_file(p):
    return os.path.splitext(p)[1] in VIDEO_EXTS

def process_video_file(video_path, out_dir, step_size=1, rotate=False, resize=(720,960)):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fid = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fid += 1
        if fid % step_size != 0:
            continue
        if rotate:
            try:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            except Exception:
                pass
        if resize[0] > 0 and resize[1] > 0:
            frame = cv2.resize(frame, tuple(resize))
        saved += 1
        out_name = os.path.join(out_dir, "%05d.png" % saved)
        cv2.imwrite(out_name, frame)
    cap.release()
    print(f"Saved {saved} frames to {out_dir}")


def run_video_task(task):
    vp, out_dir, step_size, rotate, resize = task
    process_video_file(vp, out_dir, step_size=step_size, rotate=rotate, resize=resize)


def main():
    if opt.video_path is None and opt.video_dir is None:
        raise SystemExit('Either --video_path or --video_dir must be provided')

    base_out = opt.data_root
    os.makedirs(base_out, exist_ok=True)

    targets = []
    if opt.video_path is not None:
        if os.path.isdir(opt.video_path):
            # treat as directory
            for fn in sorted(os.listdir(opt.video_path)):
                p = os.path.join(opt.video_path, fn)
                if os.path.isfile(p) and is_video_file(p):
                    targets.append(p)
        else:
            targets.append(opt.video_path)

    if opt.video_dir is not None:
        if os.path.isdir(opt.video_dir):
            for root, _, files in os.walk(opt.video_dir):
                for fn in sorted(files):
                    p = os.path.join(root, fn)
                    if is_video_file(p):
                        targets.append(p)
        else:
            raise SystemExit(f'--video_dir not a directory: {opt.video_dir}')

    # deduplicate targets
    targets = list(dict.fromkeys(targets))
    if len(targets) == 0:
        print('No video files found to process')
        return

    tasks = []
    for vp in targets:
        name = os.path.splitext(os.path.basename(vp))[0]
        out_dir = os.path.join(base_out, name)
        out_dir = os.path.join(out_dir, "image")
        print(f'Processing {vp} -> {out_dir}')
        tasks.append((vp, out_dir, opt.step_size, opt.rotate, opt.resize))

    workers = opt.workers if opt.workers and opt.workers > 0 else multiprocessing.cpu_count()
    workers = max(1, min(workers, len(tasks)))
    if workers == 1:
        for task in tasks:
            run_video_task(task)
        return

    print(f'Using {workers} workers for frame extraction')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(run_video_task, tasks))


if __name__ == '__main__':
    main()
