import os
import json
import time
import random
import logging
from dataclasses import dataclass, asdict

import numpy as np
import tensorflow as tf


@dataclass
class Config:
    rgb_dir: str = r"D:/YWWY/Projects/ZQproject/Road-extraction/archive/tiff_min/tiff_min_crop/img1_shuffle"
    feature1_dir: str = r"D:/YWWY/Projects/ZQproject/Road-extraction/archive/tiff_min/tiff_min_crop/img2_shuffle"
    feature2_dir: str = r"D:/YWWY/Projects/ZQproject/Road-extraction/archive/tiff_min/tiff_min_crop/img3_shuffle"
    mask_dir: str = r"D:/YWWY/Projects/ZQproject/Road-extraction/archive/tiff_min/tiff_min_crop/mask_shuffle"

    img_height: int = 512
    img_width: int = 512
    img_channels: int = 4

    batch_size: int = 1
    epochs: int = 50
    val_ratio: float = 0.2
    seed: int = 42

    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 2
    cosine_min_lr: float = 1e-6
    grad_clip_norm: float = 5.0
    grad_accum_steps: int = 1
    steps_per_execution: int = 1

    use_amp: bool = True
    use_xla: bool = False
    deterministic: bool = False

    prior_fuse: str = "avg"
    augment: bool = True
    cache_val: bool = True
    num_parallel_calls: int = -1

    base_filters: int = 64
    oca_length: int = 9

    exp_root: str = "./experiments"
    exp_name: str = ""
    keep_last_n_visuals: int = 50

    threshold: float = 0.5
    enable_postprocess: bool = False
    postprocess_kernel: int = 3
    remove_small_area: int = 0

    tile_size: int = 512
    tile_overlap: int = 128
    tile_batch: int = 1

    export_saved_model: bool = True
    export_tflite: bool = False


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def setup_logging(log_path: str):
    ensure_dir(os.path.dirname(log_path) or ".")
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def set_global_determinism(seed: int, deterministic: bool):
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def setup_acceleration(cfg: Config):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for d in gpus:
                tf.config.experimental.set_memory_growth(d, True)
            logging.info(f"GPU(s): {gpus}")
        except Exception as e:
            logging.warning(str(e))

    if cfg.use_xla:
        try:
            tf.config.optimizer.set_jit(True)
            logging.info("XLA enabled")
        except Exception as e:
            logging.warning(str(e))

    if cfg.use_amp:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            logging.info("AMP enabled")
        except Exception as e:
            logging.warning(str(e))


def save_config(cfg: Config, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)


def load_config(path: str) -> Config:
    cfg = Config()
    if not path:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
