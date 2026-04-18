import os
import re
import logging
from glob import glob
from typing import List, Tuple

import numpy as np
import cv2
cv2.setNumThreads(0)
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config

def _list_files(folder: str) -> List[str]:
    exts = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
    out = []
    for e in exts:
        out.extend(glob(os.path.join(folder, e)))
    return sorted(out)

def _stem(path: str) -> str:
    base = os.path.basename(path)
    return re.sub(r"\.(tif|tiff|TIF|TIFF)$", "", base)

def collect_pairs(rgb_dir: str, f1_dir: str, f2_dir: str, mask_dir: str) -> List[Tuple[str, str, str, str]]:
    rgb_files = _list_files(rgb_dir)
    f1_files = _list_files(f1_dir)
    f2_files = _list_files(f2_dir)
    m_files = _list_files(mask_dir)

    rgb_map = {_stem(p): p for p in rgb_files}
    f1_map = {_stem(p): p for p in f1_files}
    f2_map = {_stem(p): p for p in f2_files}
    m_map = {_stem(p): p for p in m_files}

    keys = sorted(set(rgb_map) & set(f1_map) & set(f2_map) & set(m_map))
    if not keys:
        raise RuntimeError("No matched samples found")

    miss = {
        "rgb_only": sorted(set(rgb_map) - set(keys))[:5],
        "f1_only": sorted(set(f1_map) - set(keys))[:5],
        "f2_only": sorted(set(f2_map) - set(keys))[:5],
        "mask_only": sorted(set(m_map) - set(keys))[:5],
    }
    for k, v in miss.items():
        if v:
            logging.warning(f"Unmatched {k}: {v}")

    pairs = [(rgb_map[k], f1_map[k], f2_map[k], m_map[k]) for k in keys]
    logging.info(f"Matched pairs: {len(pairs)}")
    return pairs

def _normalize_to_01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mx = float(np.max(img)) if img.size else 0.0
    if mx <= 1.5:
        return img
    if mx > 255.0:
        return img / 65535.0
    return img / 255.0

def _read_rgb(path: str, H: int, W: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    else:
        img = img[..., :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    img = _normalize_to_01(img)
    return img.astype(np.float32)

def _read_gray(path: str, H: int, W: int, interp) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[..., 0]
    img = cv2.resize(img, (W, H), interpolation=interp)
    img = _normalize_to_01(img)
    return img[..., None].astype(np.float32)

def _fuse_prior(f1: np.ndarray, f2: np.ndarray, mode: str) -> np.ndarray:
    if mode.lower() == "max":
        return np.maximum(f1, f2)
    return 0.5 * (f1 + f2)

def load_sample_numpy(rgb_p: str, f1_p: str, f2_p: str, m_p: str,
                      H: int, W: int, prior_fuse: str):
    rgb = _read_rgb(rgb_p, H, W)
    f1 = _read_gray(f1_p, H, W, cv2.INTER_LINEAR)
    f2 = _read_gray(f2_p, H, W, cv2.INTER_LINEAR)
    prior = _fuse_prior(f1, f2, prior_fuse)
    mask = _read_gray(m_p, H, W, cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.float32)
    x = np.concatenate([rgb, prior], axis=-1).astype(np.float32)
    y = mask.astype(np.float32)
    return x, y

def augment_np(x: np.ndarray, y: np.ndarray):
    # x: [H, W, C], y: [H, W, 1]
    
    if np.random.rand() < 0.5:
        x = np.fliplr(x)
        y = np.fliplr(y)
        
    if np.random.rand() < 0.5:
        x = np.flipud(x)
        y = np.flipud(y)
        
    k = np.random.randint(0, 4)
    if k > 0:
        x = np.rot90(x, k, axes=(0, 1))
        y = np.rot90(y, k, axes=(0, 1))
        
    rgb = x[..., :3]
    prior = x[..., 3:4]
    
    # color jitter on rgb
    if np.random.rand() < 0.5:
        delta = np.random.uniform(-0.08, 0.08)
        rgb = np.clip(rgb + delta, 0.0, 1.0)
        
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.9, 1.1)
        mean = np.mean(rgb, axis=(0, 1), keepdims=True)
        rgb = np.clip((rgb - mean) * factor + mean, 0.0, 1.0)
        
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.01, size=rgb.shape).astype(np.float32)
        rgb = np.clip(rgb + noise, 0.0, 1.0)
        
    x = np.concatenate([rgb, prior], axis=-1)
    
    # Ensure memory is contiguous after rotations/flips
    return np.ascontiguousarray(x), np.ascontiguousarray(y)

class RoadAttNetDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, str, str]], cfg: Config, training: bool):
        self.pairs = pairs
        self.cfg = cfg
        self.training = training

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_p, f1_p, f2_p, m_p = self.pairs[idx]
        x_np, y_np = load_sample_numpy(
            rgb_p, f1_p, f2_p, m_p,
            self.cfg.img_height, self.cfg.img_width, self.cfg.prior_fuse
        )
        
        if self.training and self.cfg.augment:
            x_np, y_np = augment_np(x_np, y_np)
            
        # Convert to CHW
        x_tensor = torch.from_numpy(x_np).permute(2, 0, 1).float()
        y_tensor = torch.from_numpy(y_np).permute(2, 0, 1).float()
        
        return x_tensor, y_tensor

def build_dataset(pairs: List[Tuple[str, str, str, str]], cfg: Config, training: bool) -> DataLoader:
    if not pairs:
        raise ValueError("No pairs were provided to build_dataset")
        
    ds = RoadAttNetDataset(pairs, cfg, training)
    
    num_workers = cfg.num_parallel_calls if cfg.num_parallel_calls > 0 else 0
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    return loader
