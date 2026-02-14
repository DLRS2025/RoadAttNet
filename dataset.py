import os
import re
import logging
from glob import glob
from typing import List, Tuple

import numpy as np
import cv2
import tensorflow as tf

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


def load_sample_numpy(rgb_p: bytes, f1_p: bytes, f2_p: bytes, m_p: bytes,
                      H: int, W: int, prior_fuse: str):
    rgb = _read_rgb(rgb_p.decode("utf-8"), H, W)
    f1 = _read_gray(f1_p.decode("utf-8"), H, W, cv2.INTER_LINEAR)
    f2 = _read_gray(f2_p.decode("utf-8"), H, W, cv2.INTER_LINEAR)
    prior = _fuse_prior(f1, f2, prior_fuse)
    mask = _read_gray(m_p.decode("utf-8"), H, W, cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.float32)
    x = np.concatenate([rgb, prior], axis=-1).astype(np.float32)
    y = mask.astype(np.float32)
    return x, y


def augment_tf(x: tf.Tensor, y: tf.Tensor):
    r = tf.random.uniform([], 0, 1.0)
    x = tf.cond(r < 0.5, lambda: tf.image.flip_left_right(x), lambda: x)
    y = tf.cond(r < 0.5, lambda: tf.image.flip_left_right(y), lambda: y)

    r = tf.random.uniform([], 0, 1.0)
    x = tf.cond(r < 0.5, lambda: tf.image.flip_up_down(x), lambda: x)
    y = tf.cond(r < 0.5, lambda: tf.image.flip_up_down(y), lambda: y)

    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    y = tf.image.rot90(y, k)

    rgb = x[..., :3]
    prior = x[..., 3:4]

    rgb = tf.image.random_brightness(rgb, max_delta=0.08)
    rgb = tf.image.random_contrast(rgb, lower=0.9, upper=1.1)
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)

    noise = tf.random.normal(tf.shape(rgb), mean=0.0, stddev=0.01, dtype=rgb.dtype)
    rgb = tf.clip_by_value(rgb + noise, 0.0, 1.0)

    x = tf.concat([rgb, prior], axis=-1)
    return x, y


def build_dataset(pairs: List[Tuple[str, str, str, str]], cfg: Config, training: bool) -> tf.data.Dataset:
    rgb_list = [p[0] for p in pairs]
    f1_list = [p[1] for p in pairs]
    f2_list = [p[2] for p in pairs]
    m_list = [p[3] for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((rgb_list, f1_list, f2_list, m_list))
    if training:
        ds = ds.shuffle(buffer_size=min(len(pairs), 1024), seed=cfg.seed, reshuffle_each_iteration=True)

    num_calls = tf.data.AUTOTUNE if cfg.num_parallel_calls == -1 else cfg.num_parallel_calls
    H, W = cfg.img_height, cfg.img_width

    def _load_map(rgb_p, f1_p, f2_p, m_p):
        x, y = tf.numpy_function(
            func=load_sample_numpy,
            inp=[rgb_p, f1_p, f2_p, m_p, H, W, cfg.prior_fuse],
            Tout=[tf.float32, tf.float32],
        )
        x.set_shape([H, W, cfg.img_channels])
        y.set_shape([H, W, 1])
        return x, y

    ds = ds.map(_load_map, num_parallel_calls=num_calls)

    if training and cfg.augment:
        ds = ds.map(lambda x, y: augment_tf(x, y), num_parallel_calls=num_calls)

    ds = ds.batch(cfg.batch_size, drop_remainder=training)

    options = tf.data.Options()
    options.experimental_deterministic = cfg.deterministic
    ds = ds.with_options(options)

    if (not training) and cfg.cache_val:
        ds = ds.cache()

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
