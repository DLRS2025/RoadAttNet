import os
import argparse
import logging

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import Config, setup_logging, set_global_determinism, setup_acceleration, load_config, ensure_dir
from dataset import collect_pairs, build_dataset
from model import build_roadattnet_core, RoadAttNet
from visualize import PredictionVisualizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--out", type=str, default="./test_out")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out)
    setup_logging(os.path.join(args.out, "test.log"))
    cfg = load_config(args.config)

    set_global_determinism(cfg.seed, cfg.deterministic)
    setup_acceleration(cfg)

    pairs = collect_pairs(cfg.rgb_dir, cfg.feature1_dir, cfg.feature2_dir, cfg.mask_dir)
    idx = np.arange(len(pairs))
    tr_idx, va_idx = train_test_split(idx, test_size=cfg.val_ratio, random_state=cfg.seed, shuffle=True)
    val_pairs = [pairs[i] for i in va_idx]
    val_ds = build_dataset(val_pairs, cfg, training=False)

    core = build_roadattnet_core(
        input_shape=(cfg.img_height, cfg.img_width, cfg.img_channels),
        base_filters=cfg.base_filters,
        oca_length=cfg.oca_length,
    )
    model = RoadAttNet(core, grad_accum_steps=1, grad_clip_norm=0.0)
    model.build((None, cfg.img_height, cfg.img_width, cfg.img_channels))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    model.load_weights(args.weights)

    results = model.evaluate(val_ds, verbose=1)
    logging.info(str(dict(zip(model.metrics_names, results))))

    vis_dir = ensure_dir(os.path.join(args.out, "visuals"))
    vis_cb = PredictionVisualizer(val_ds, vis_dir, cfg, max_batches=2)
    vis_cb.set_model(model)
    vis_cb.on_epoch_end(0)


if __name__ == "__main__":
    main()
