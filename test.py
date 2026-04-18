import os
import argparse
import logging
import json

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from config import Config, setup_logging, set_global_determinism, setup_acceleration, load_config, ensure_dir
from dataset import collect_pairs, build_dataset
from model import build_roadattnet_core, RoadAttNet
from visualize import PredictionVisualizer
from train import calculate_metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--out", type=str, default="./test_out")
    p.add_argument("--require-gpu", action="store_true")
    return p.parse_args()

def split_train_val_pairs(pairs, cfg: Config):
    if len(pairs) < 2:
        raise RuntimeError("Need at least 2 matched samples to create train/val splits")
    if not 0.0 < cfg.val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {cfg.val_ratio}")

    idx = np.arange(len(pairs))
    tr_idx, va_idx = train_test_split(idx, test_size=cfg.val_ratio, random_state=cfg.seed, shuffle=True)
    val_pairs = [pairs[i] for i in va_idx]
    if not val_pairs:
        raise RuntimeError("The current dataset split produced an empty validation set")
    return val_pairs

def main():
    args = parse_args()
    ensure_dir(args.out)
    setup_logging(os.path.join(args.out, "test.log"))
    cfg = load_config(args.config)
    cfg.require_gpu = bool(cfg.require_gpu or args.require_gpu)

    set_global_determinism(cfg.seed, cfg.deterministic)
    device = setup_acceleration(cfg, purpose="evaluation")

    pairs = collect_pairs(cfg.rgb_dir, cfg.feature1_dir, cfg.feature2_dir, cfg.mask_dir)
    val_pairs = split_train_val_pairs(pairs, cfg)
    val_loader = build_dataset(val_pairs, cfg, training=False)

    core = build_roadattnet_core(base_filters=cfg.base_filters, oca_length=cfg.oca_length)
    model = RoadAttNet(core).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    from collections import defaultdict
    final_metrics = defaultdict(float)
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            main, aux1, aux2, aux3 = model(x)
            loss, L_main, L_aux = model.compute_loss(y, main, aux1, aux2, aux3)
            acc, recall, iou = calculate_metrics(main, y)
            final_metrics['loss'] += loss.item()
            final_metrics['accuracy'] += acc
            final_metrics['recall'] += recall
            final_metrics['iou'] += iou
    for k in final_metrics:
        final_metrics[k] /= len(val_loader)
        
    logging.info(str(dict(final_metrics)))

    vis_dir = ensure_dir(os.path.join(args.out, "visuals"))
    vis_cb = PredictionVisualizer(val_loader, vis_dir, cfg, max_batches=2)
    vis_cb.on_epoch_end(0, model, device)

if __name__ == "__main__":
    main()
