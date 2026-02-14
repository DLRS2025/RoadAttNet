import os
import json
import argparse
import logging

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, BackupAndRestore

from config import Config, ensure_dir, timestamp, setup_logging, set_global_determinism, setup_acceleration, save_config, load_config
from dataset import collect_pairs, build_dataset
from model import build_roadattnet_core, RoadAttNet
from visualize import PredictionVisualizer, plot_history


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps=0, min_lr=1e-6):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        warm = tf.cast(self.warmup_steps, tf.float32)

        def warmup():
            return self.base_lr * (step / tf.maximum(1.0, warm))

        def cosine():
            progress = (step - warm) / tf.maximum(1.0, (total - warm))
            cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * tf.clip_by_value(progress, 0.0, 1.0)))
            return (self.base_lr - self.min_lr) * cosine_decay + self.min_lr

        return tf.cond(step < warm, warmup, cosine)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


def build_optimizer(cfg: Config, total_steps: int):
    warmup_steps = int(cfg.warmup_epochs * (total_steps / max(cfg.epochs, 1)))
    lr_schedule = WarmupCosine(cfg.lr, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=cfg.cosine_min_lr)
    if cfg.weight_decay > 0:
        opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=cfg.weight_decay)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    if cfg.use_amp:
        try:
            from tensorflow.keras.mixed_precision import LossScaleOptimizer
            opt = LossScaleOptimizer(opt)
        except Exception:
            pass
    return opt


def export_model(model: RoadAttNet, out_dir: str, cfg: Config):
    ensure_dir(out_dir)
    if cfg.export_saved_model:
        sm_dir = os.path.join(out_dir, "saved_model")
        try:
            model.core.save(sm_dir)
            logging.info(f"SavedModel: {sm_dir}")
        except Exception as e:
            logging.warning(str(e))
    if cfg.export_tflite:
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model.core)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite = converter.convert()
            p = os.path.join(out_dir, "model.tflite")
            with open(p, "wb") as f:
                f.write(tflite)
            logging.info(f"TFLite: {p}")
        except Exception as e:
            logging.warning(str(e))


def train(cfg: Config):
    exp_name = cfg.exp_name.strip() or f"roadattnet_{timestamp()}"
    exp_dir = ensure_dir(os.path.join(cfg.exp_root, exp_name))
    ckpt_dir = ensure_dir(os.path.join(exp_dir, "checkpoints"))
    vis_dir = ensure_dir(os.path.join(exp_dir, "visuals"))
    tb_dir = ensure_dir(os.path.join(exp_dir, "tb"))
    backup_dir = ensure_dir(os.path.join(exp_dir, "backup"))

    setup_logging(os.path.join(exp_dir, "run.log"))
    save_config(cfg, os.path.join(exp_dir, "config.json"))
    logging.info(f"Experiment: {exp_dir}")

    set_global_determinism(cfg.seed, cfg.deterministic)
    setup_acceleration(cfg)

    pairs = collect_pairs(cfg.rgb_dir, cfg.feature1_dir, cfg.feature2_dir, cfg.mask_dir)
    idx = np.arange(len(pairs))
    tr_idx, va_idx = train_test_split(idx, test_size=cfg.val_ratio, random_state=cfg.seed, shuffle=True)
    train_pairs = [pairs[i] for i in tr_idx]
    val_pairs = [pairs[i] for i in va_idx]
    logging.info(f"Train/Val: {len(train_pairs)} / {len(val_pairs)}")

    train_ds = build_dataset(train_pairs, cfg, training=True)
    val_ds = build_dataset(val_pairs, cfg, training=False)

    steps_per_epoch = max(1, len(train_pairs) // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs

    core = build_roadattnet_core(
        input_shape=(cfg.img_height, cfg.img_width, cfg.img_channels),
        base_filters=cfg.base_filters,
        oca_length=cfg.oca_length,
    )
    model = RoadAttNet(core, grad_accum_steps=cfg.grad_accum_steps, grad_clip_norm=cfg.grad_clip_norm)
    model.build((None, cfg.img_height, cfg.img_width, cfg.img_channels))

    optimizer = build_optimizer(cfg, total_steps)

    model.compile(optimizer=optimizer, run_eagerly=False, steps_per_execution=cfg.steps_per_execution)

    best_path = os.path.join(ckpt_dir, "best.weights.h5")
    last_path = os.path.join(ckpt_dir, "last.weights.h5")

    tb_writer = tf.summary.create_file_writer(tb_dir)
    vis_cb = PredictionVisualizer(val_ds, vis_dir, cfg, max_batches=1)
    vis_cb.set_tb_writer(tb_writer)

    cbs = [
        BackupAndRestore(backup_dir),
        ModelCheckpoint(best_path, monitor="val_iou", mode="max", save_best_only=True, save_weights_only=True, verbose=1),
        ModelCheckpoint(last_path, monitor="val_iou", mode="max", save_best_only=False, save_weights_only=True, verbose=0),
        EarlyStopping(monitor="val_iou", mode="max", patience=15, verbose=1, restore_best_weights=False),
        CSVLogger(os.path.join(exp_dir, "history.csv"), append=True),
        TensorBoard(log_dir=tb_dir, update_freq="epoch", profile_batch=0),
        vis_cb,
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=cbs, verbose=1)
    plot_history(history, os.path.join(exp_dir, "training_history.png"))

    if os.path.exists(best_path):
        model.load_weights(best_path)
        logging.info(f"Loaded best: {best_path}")

    results = model.evaluate(val_ds, verbose=1)
    logging.info(str(dict(zip(model.metrics_names, results))))

    export_model(model, os.path.join(exp_dir, "export"), cfg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
