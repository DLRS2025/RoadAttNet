import os
import json
import time
import random
import logging
import sys
from dataclasses import dataclass, asdict
import torch
import numpy as np

@dataclass
class Config:
    rgb_dir: str = "/mnt/d/data/Massachusetts/archive/tiff/train"
    feature1_dir: str = "/mnt/d/data/Massachusetts/archive/tiff/train_blurred"
    feature2_dir: str = "/mnt/d/data/Massachusetts/archive/tiff/train_cwl"
    mask_dir: str = "/mnt/d/data/Massachusetts/archive/tiff/train_labels"

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

    require_gpu: bool = False
    gpu_smoke_test: bool = True
    use_amp: bool = True
    use_xla: bool = False
    deterministic: bool = False

    prior_fuse: str = "avg"
    augment: bool = True
    cache_val: bool = True
    num_parallel_calls: int = 4

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

    export_saved_model: bool = False
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_acceleration(cfg: Config, purpose: str = "runtime"):
    logging.info(f"PyTorch {torch.__version__} | python={sys.executable}")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    built_with_cuda = torch.cuda.is_available()
    
    logging.info(
        f"Acceleration for {purpose} | require_gpu={cfg.require_gpu} | "
        f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} | cuda_available={built_with_cuda}"
    )

    device = torch.device("cpu")
    if built_with_cuda:
        device = torch.device("cuda:0")
        logging.info(f"GPU(s): {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
        if cfg.gpu_smoke_test:
            try:
                x = torch.tensor([1.0, 2.0, 3.0], device=device)
                y = torch.sum(x)
                _ = float(y.cpu().numpy())
                logging.info("GPU smoke test passed")
            except Exception as e:
                raise RuntimeError(f"GPU smoke test failed: {e}")
    else:
        msg = f"No PyTorch GPU device detected for {purpose}."
        if cfg.require_gpu:
            raise RuntimeError(msg)
        logging.warning(msg)

    if cfg.use_amp:
        if built_with_cuda:
            logging.info("AMP enabled")
        else:
            logging.info("AMP skipped because no GPU device is available")

    if cfg.use_xla:
        logging.info("PyTorch does not support XLA out of the box in the same way as TF, ignoring use_xla=True.")

    return device

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
