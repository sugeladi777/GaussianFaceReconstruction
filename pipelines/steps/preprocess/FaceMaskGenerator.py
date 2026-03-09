"""人脸分割 mask 生成器。"""

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Tuple

import facer
import torch
from torchvision.utils import save_image
from tqdm import tqdm


class FaceMaskGenerator:
    _MODEL_CACHE: Dict[str, Tuple[object, object]] = {}

    @staticmethod
    def _ensure_model_file():
        """多进程安全地加锁，确保模型文件只被一个进程下载/加载。"""
        import os, time
        model_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "face_parsing.farl.lapa.main_ema_136500_jit191.pt"
        lock_path = model_path.with_suffix(model_path.suffix + ".lock")
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
                break
            except FileExistsError:
                time.sleep(1)
        try:
            pass  # 只加锁，不做拷贝
        finally:
            try:
                os.remove(str(lock_path))
            except Exception:
                pass

    def __init__(self, device="cuda:0"):
        self.device = device
        FaceMaskGenerator._ensure_model_file()
        self.face_detector, self.face_parser = self._load_models(device)

    @classmethod
    def _load_models(cls, device: str):
        cached = cls._MODEL_CACHE.get(device)
        if cached is not None:
            return cached

        with cls._model_download_lock():
            # 锁内二次检查，避免并发进程重复下载同一模型。
            cached = cls._MODEL_CACHE.get(device)
            if cached is not None:
                return cached
            try:
                detector = facer.face_detector("retinaface/mobilenet", device=device)
                parser = facer.face_parser("farl/lapa/448", device=device)
            except Exception:
                cls._cleanup_known_parser_cache()
                detector = facer.face_detector("retinaface/mobilenet", device=device)
                parser = facer.face_parser("farl/lapa/448", device=device)

        cls._MODEL_CACHE[device] = (detector, parser)
        return detector, parser

    @staticmethod
    @contextmanager
    def _model_download_lock():
        lock_path = Path.home() / ".cache" / "facer" / ".model_download.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield lock_file
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _foreground_mask(seg_logits: torch.Tensor) -> torch.Tensor:
        labels = seg_logits.argmax(dim=1, keepdim=True)
        return (labels > 0).float()

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
            except OSError:
                pass

    def generate_masks_from_dir(self, img_dir, mask_dir):
        mask_dir_path = Path(mask_dir)
        mask_dir_path.mkdir(parents=True, exist_ok=True)
        img_paths = sorted(Path(img_dir).glob("*.png"))
        if not img_paths:
            return

        for img_path in tqdm(img_paths, desc=f"FaceMaskGenerator: {img_dir}"):
            try:
                img = facer.hwc2bchw(facer.read_hwc(str(img_path)))
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                img = img.to(device=self.device)

                with torch.inference_mode():
                    faces = self.face_detector(img)
                    faces = self.face_parser(img, faces)

                logits = faces["seg"]["logits"]
                face_mask = self._foreground_mask(logits[:1])
                if face_mask.sum() == 0:
                    continue
                save_image(face_mask, str(mask_dir_path / img_path.name))
            except Exception as exc:
                print(f"FaceMaskGenerator: 写入失败 {img_path}: {exc}")
