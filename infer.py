import os
import argparse
import logging

import numpy as np
import cv2
import tensorflow as tf

from config import Config, setup_logging, setup_acceleration, load_config, ensure_dir
from model import build_roadattnet_core, RoadAttNet


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


def _read_gray(path: str, H: int, W: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[..., 0]
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    img = _normalize_to_01(img)
    return img[..., None].astype(np.float32)


def _fuse_prior(f1: np.ndarray, f2: np.ndarray, mode: str) -> np.ndarray:
    if mode.lower() == "max":
        return np.maximum(f1, f2)
    return 0.5 * (f1 + f2)


def overlay_on_rgb(rgb01: np.ndarray, mask01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    m = (mask01 > 0.5).astype(np.uint8)
    overlay = rgb.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + (m * 180), 0, 255)
    out = (rgb * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return out


def postprocess_mask(bin_mask: np.ndarray, cfg: Config) -> np.ndarray:
    m = (bin_mask > 0).astype(np.uint8) * 255
    k = int(max(1, cfg.postprocess_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    if cfg.remove_small_area and cfg.remove_small_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
        cleaned = np.zeros_like(m)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= cfg.remove_small_area:
                cleaned[labels == i] = 255
        m = cleaned
    return (m > 0).astype(np.uint8)


def sliding_window_predict(model: tf.keras.Model, x: np.ndarray, cfg: Config) -> np.ndarray:
    H, W, C = x.shape
    tile = int(cfg.tile_size)
    ov = int(cfg.tile_overlap)
    step = max(1, tile - ov)

    prob_sum = np.zeros((H, W), dtype=np.float32)
    w_sum = np.zeros((H, W), dtype=np.float32)

    yy, xx = np.meshgrid(np.linspace(-1, 1, tile), np.linspace(-1, 1, tile), indexing="ij")
    w = (1 - np.abs(yy)) * (1 - np.abs(xx))
    w = np.clip(w, 0.05, 1.0).astype(np.float32)

    patches = []
    coords = []

    def flush():
        if not patches:
            return
        batch = np.stack(patches, axis=0).astype(np.float32)
        preds = model.predict(batch, verbose=0)[0]
        for (y0, x0), pr in zip(coords, preds):
            pr2 = pr[..., 0].astype(np.float32)
            hh = min(tile, H - y0)
            ww = min(tile, W - x0)
            prob_sum[y0:y0 + hh, x0:x0 + ww] += pr2[:hh, :ww] * w[:hh, :ww]
            w_sum[y0:y0 + hh, x0:x0 + ww] += w[:hh, :ww]
        patches.clear()
        coords.clear()

    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            patch = np.zeros((tile, tile, C), dtype=np.float32)
            patch[:(y1 - y0), :(x1 - x0), :] = x[y0:y1, x0:x1, :]
            patches.append(patch)
            coords.append((y0, x0))
            if len(patches) >= int(max(1, cfg.tile_batch)):
                flush()

    flush()
    prob = prob_sum / np.maximum(w_sum, 1e-6)
    return prob.astype(np.float32)


def run_predict(cfg: Config, weights: str, rgb_path: str, f1_path: str, f2_path: str, out_prefix: str, sliding: bool):
    setup_acceleration(cfg)

    core = build_roadattnet_core(
        input_shape=(cfg.img_height, cfg.img_width, cfg.img_channels),
        base_filters=cfg.base_filters,
        oca_length=cfg.oca_length,
    )
    model = RoadAttNet(core, grad_accum_steps=1, grad_clip_norm=0.0)
    model.build((None, cfg.img_height, cfg.img_width, cfg.img_channels))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    model.load_weights(weights)

    rgb = _read_rgb(rgb_path, cfg.img_height, cfg.img_width)
    f1 = _read_gray(f1_path, cfg.img_height, cfg.img_width)
    f2 = _read_gray(f2_path, cfg.img_height, cfg.img_width)
    prior = _fuse_prior(f1, f2, cfg.prior_fuse)
    x = np.concatenate([rgb, prior], axis=-1).astype(np.float32)

    if sliding:
        prob = sliding_window_predict(model, x, cfg)
    else:
        pred = model.predict(x[None, ...], verbose=0)[0]
        prob = pred[0, ..., 0].astype(np.float32)

    bin_mask = (prob >= cfg.threshold).astype(np.uint8)
    if cfg.enable_postprocess:
        bin_mask = postprocess_mask(bin_mask, cfg)

    ensure_dir(os.path.dirname(out_prefix) or ".")
    prob_img = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    overlay = overlay_on_rgb(rgb, prob, alpha=0.45)

    cv2.imwrite(out_prefix + "_prob.png", prob_img)
    cv2.imwrite(out_prefix + "_bin.png", bin_mask * 255)
    cv2.imwrite(out_prefix + "_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    np.save(out_prefix + "_prob.npy", prob.astype(np.float32))
    np.save(out_prefix + "_bin.npy", bin_mask.astype(np.uint8))
    logging.info(out_prefix)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--rgb", type=str, required=True)
    p.add_argument("--f1", type=str, required=True)
    p.add_argument("--f2", type=str, required=True)
    p.add_argument("--out", type=str, default="./predict/out")
    p.add_argument("--sliding", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging("./predict.log")
    cfg = load_config(args.config)
    run_predict(cfg, args.weights, args.rgb, args.f1, args.f2, args.out, args.sliding)


if __name__ == "__main__":
    main()
