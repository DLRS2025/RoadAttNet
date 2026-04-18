import os
import math
import logging
import json
import shutil
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config, ensure_dir, timestamp, setup_logging, set_global_determinism, setup_acceleration, save_config, load_config
from dataset import collect_pairs, build_dataset
from model import build_roadattnet_core, RoadAttNet
from visualize import PredictionVisualizer, plot_history


def get_lr_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def split_train_val_pairs(pairs, cfg: Config):
    from sklearn.model_selection import train_test_split
    if len(pairs) < 2:
        raise RuntimeError("Need at least 2 matched samples to create train/val splits")
    if not 0.0 < cfg.val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {cfg.val_ratio}")

    idx = np.arange(len(pairs))
    tr_idx, va_idx = train_test_split(idx, test_size=cfg.val_ratio, random_state=cfg.seed, shuffle=True)
    train_pairs = [pairs[i] for i in tr_idx]
    val_pairs = [pairs[i] for i in va_idx]
    if not train_pairs or not val_pairs:
        raise RuntimeError("The current dataset split produced an empty train or validation set")
    return train_pairs, val_pairs


def calculate_metrics(y_pred, y_true):
    pred = (y_pred > 0.5).float()
    correct = (pred == y_true).float()
    acc = correct.mean().item()
    
    true_positives = (pred * y_true).sum().item()
    possible_positives = y_true.sum().item()
    predicted_positives = pred.sum().item()
    
    recall = true_positives / (possible_positives + 1e-6)
    
    intersection = (pred * y_true).sum().item()
    union = predicted_positives + possible_positives - intersection
    iou = intersection / (union + 1e-6)
    
    return acc, recall, iou


def train(cfg: Config):
    exp_name = cfg.exp_name.strip() or f"roadattnet_{timestamp()}"
    exp_dir = ensure_dir(os.path.join(cfg.exp_root, exp_name))
    ckpt_dir = ensure_dir(os.path.join(exp_dir, "checkpoints"))
    vis_dir = ensure_dir(os.path.join(exp_dir, "visuals"))
    tb_dir = ensure_dir(os.path.join(exp_dir, "tb"))

    setup_logging(os.path.join(exp_dir, "run.log"))
    save_config(cfg, os.path.join(exp_dir, "config.json"))
    logging.info(f"Experiment: {exp_dir}")

    set_global_determinism(cfg.seed, cfg.deterministic)
    device = setup_acceleration(cfg, purpose="training")

    pairs = collect_pairs(cfg.rgb_dir, cfg.feature1_dir, cfg.feature2_dir, cfg.mask_dir)
    train_pairs, val_pairs = split_train_val_pairs(pairs, cfg)
    logging.info(f"Train/Val: {len(train_pairs)} / {len(val_pairs)}")

    train_loader = build_dataset(train_pairs, cfg, training=True)
    val_loader = build_dataset(val_pairs, cfg, training=False)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs

    core = build_roadattnet_core(base_filters=cfg.base_filters, oca_length=cfg.oca_length)
    model = RoadAttNet(core).to(device)

    if cfg.weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    min_lr_ratio = cfg.cosine_min_lr / cfg.lr if cfg.lr > 0 else 0
    scheduler = get_lr_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio)

    scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_amp and torch.cuda.is_available())

    best_path = os.path.join(ckpt_dir, "best.weights.pt")
    last_path = os.path.join(ckpt_dir, "last.weights.pt")

    tb_writer = SummaryWriter(log_dir=tb_dir)
    vis_cb = PredictionVisualizer(val_loader, vis_dir, cfg, max_batches=1)
    vis_cb.set_tb_writer(tb_writer)

    history = defaultdict(list)
    best_val_iou = -1.0
    patience_counter = 0
    patience = 15

    for epoch in range(cfg.epochs):
        model.train()
        train_metrics = defaultdict(float)
        
        logging.info(f"Starting Epoch {epoch+1}/{cfg.epochs}")
        pbar = tqdm(train_loader, desc="Training", unit="batch", leave=False)
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast('cuda', enabled=cfg.use_amp and torch.cuda.is_available()):
                main, aux1, aux2, aux3 = model(x)
                loss, L_main, L_aux = model.compute_loss(y, main, aux1, aux2, aux3)
                loss = loss / cfg.grad_accum_steps
                
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % cfg.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if cfg.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            # Record metrics
            batch_loss = loss.item() * cfg.grad_accum_steps
            acc, recall, iou = calculate_metrics(main, y)
            
            train_metrics['loss'] += batch_loss
            train_metrics['main_loss'] += L_main.item()
            train_metrics['aux_loss'] += L_aux.item()
            train_metrics['accuracy'] += acc
            train_metrics['recall'] += recall
            train_metrics['iou'] += iou
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "iou": f"{iou:.4f}",
                "lr": f"{current_lr:.2e}"
            })

        pbar.close()

        # Compute average training metrics
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
            history[k].append(train_metrics[k])
            tb_writer.add_scalar(f"train/{k}", train_metrics[k], epoch)

        # Validation phase
        model.eval()
        val_metrics = defaultdict(float)
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", unit="batch", leave=False)
            for x, y in val_pbar:
                x, y = x.to(device), y.to(device)
                main, aux1, aux2, aux3 = model(x)
                loss, L_main, L_aux = model.compute_loss(y, main, aux1, aux2, aux3)
                
                val_loss = loss.item()
                acc, recall, iou = calculate_metrics(main, y)
                
                val_metrics['val_loss'] += val_loss
                val_metrics['val_main_loss'] += L_main.item()
                val_metrics['val_aux_loss'] += L_aux.item()
                val_metrics['val_accuracy'] += acc
                val_metrics['val_recall'] += recall
                val_metrics['val_iou'] += iou
                
                val_pbar.set_postfix({"v_loss": f"{val_loss:.4f}", "v_iou": f"{iou:.4f}"})
            val_pbar.close()
                
        # Compute average validation metrics
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
            history[k].append(val_metrics[k])
            tb_writer.add_scalar(f"val/{k.replace('val_', '')}", val_metrics[k], epoch)
            
        # Log summary for the epoch
        logging.info(f"Epoch {epoch+1}/{cfg.epochs} Completed | "
                     f"Train Loss: {train_metrics['loss']:.4f} | Train IoU: {train_metrics['iou']:.4f} | "
                     f"Val Loss: {val_metrics['val_loss']:.4f} | Val IoU: {val_metrics['val_iou']:.4f}")

        torch.save(model.state_dict(), last_path)
        
        if val_metrics['val_iou'] > best_val_iou:
            best_val_iou = val_metrics['val_iou']
            torch.save(model.state_dict(), best_path)
            logging.info(f"-> Saved new best model to {best_path} with Val IoU: {best_val_iou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        vis_cb.on_epoch_end(epoch, model, device)

        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    plot_history(history, os.path.join(exp_dir, "training_history.png"))

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        logging.info(f"Loaded best: {best_path}")

    model.eval()
    final_metrics = defaultdict(float)
    with torch.no_grad():
        test_pbar = tqdm(val_loader, desc="Final Eval", unit="batch", leave=False)
        for x, y in test_pbar:
            x, y = x.to(device), y.to(device)
            main, aux1, aux2, aux3 = model(x)
            loss, L_main, L_aux = model.compute_loss(y, main, aux1, aux2, aux3)
            acc, recall, iou = calculate_metrics(main, y)
            final_metrics['loss'] += loss.item()
            final_metrics['accuracy'] += acc
            final_metrics['recall'] += recall
            final_metrics['iou'] += iou
        test_pbar.close()
            
    for k in final_metrics:
        final_metrics[k] /= len(val_loader)
    logging.info(str(dict(final_metrics)))

    download_dir = os.path.join(exp_dir, "DOWNLOAD_ME_FOR_LOCAL_INFERENCE")
    ensure_dir(download_dir)
    
    if os.path.exists(best_path):
        shutil.copy2(best_path, os.path.join(download_dir, "best.weights.pt"))
    
    cfg_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(cfg_path):
        shutil.copy2(cfg_path, os.path.join(download_dir, "config.json"))
        
    logging.info("\n" + "*"*60)
    logging.info(f"==> Training completed! Download the folder to your local machine: {download_dir}")
    logging.info("*"*60 + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="")
    p.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback instead of requiring a visible GPU")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.require_gpu = not args.allow_cpu
    train(cfg)


if __name__ == "__main__":
    main()
