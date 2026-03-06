"""人脸分割 mask 生成器。"""

import os
import glob
import fcntl
from pathlib import Path
import torch
from torchvision.utils import save_image
from tqdm import tqdm
import facer


class FaceMaskGenerator:
    """基于 facer 的检测+解析模型生成前景 mask。"""

    def __init__(self, device="cuda:0"):
        # 检测器负责定位人脸，解析器负责输出像素级语义分割。
        self.device = device
        with self._model_download_lock():
            try:
                self.face_detector = facer.face_detector('retinaface/mobilenet', device=device)
                self.face_parser = facer.face_parser('farl/lapa/448', device=device)
            except Exception:
                self._cleanup_known_parser_cache()
                self.face_detector = facer.face_detector('retinaface/mobilenet', device=device)
                self.face_parser = facer.face_parser('farl/lapa/448', device=device)

    @staticmethod
    def _model_download_lock():
        lock_dir = Path.home() / ".cache" / "facer"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file = open(lock_dir / ".model_download.lock", "w")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        class _LockCtx:
            def __enter__(self_nonlocal):
                return lock_file

            def __exit__(self_nonlocal, exc_type, exc, tb):
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                finally:
                    lock_file.close()

        return _LockCtx()

    @staticmethod
    def _cleanup_known_parser_cache() -> None:
        targets = [
            Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "face_parsing.farl.lapa.main_ema_136500_jit191.pt",
            Path.home() / ".cache" / "facer" / "face_parsing.farl.lapa.main_ema_136500_jit191.pt",
        ]
        for file_path in targets:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass

    def generate_masks_from_dir(self, img_dir, mask_dir):
        """遍历目录中的 PNG 图像并写出同名 mask。"""
        os.makedirs(mask_dir, exist_ok=True)
        img_path_list = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        for img_path in tqdm(img_path_list, desc=f"FaceMaskGenerator: {img_dir}"):
            try:
                image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=self.device)
                with torch.inference_mode():
                    faces = self.face_detector(image)
                    faces = self.face_parser(image, faces)
                seg_logits = faces['seg']['logits']
                seg_probs = seg_logits.softmax(dim=1)[0]
                vis_seg_probs = seg_probs.argmax(dim=0)
                # 背景=0，其余类别视为前景并保留。
                face_mask = (vis_seg_probs >= 1).float()
                img_name = os.path.basename(img_path)
                save_image(face_mask, os.path.join(mask_dir, img_name))
            except Exception as e:
                print(f"FaceMaskGenerator: error processing {img_path}: {e}")
                continue
