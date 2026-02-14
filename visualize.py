import os
import math
from glob import glob

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from config import Config, ensure_dir


def overlay_on_rgb(rgb01: np.ndarray, mask01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    m = (mask01 > 0.5).astype(np.uint8)
    overlay = rgb.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + (m * 180), 0, 255)
    out = (rgb * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return out


class PredictionVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, val_ds: tf.data.Dataset, out_dir: str, cfg: Config, max_batches: int = 1):
        super().__init__()
        self.val_ds = val_ds
        self.out_dir = ensure_dir(out_dir)
        self.cfg = cfg
        self.max_batches = max_batches
        self._fixed = []
        for x, y in val_ds.take(max_batches):
            self._fixed.append((x, y))
        self.tb_writer = None

    def set_tb_writer(self, writer):
        self.tb_writer = writer

    def on_epoch_end(self, epoch, logs=None):
        epoch_id = epoch + 1
        for bi, (x, y) in enumerate(self._fixed):
            preds = self.model.predict(x, verbose=0)[0]
            x_np = x.numpy()
            y_np = y.numpy()
            p_np = preds

            n = min(x_np.shape[0], 3)
            fig = plt.figure(figsize=(18, 6 * n))
            for i in range(n):
                rgb = x_np[i, ..., :3]
                prior = x_np[i, ..., 3]
                gt = y_np[i, ..., 0]
                pr = p_np[i, ..., 0]
                ov = overlay_on_rgb(rgb, pr, alpha=0.45)

                ax1 = plt.subplot(n, 5, 5 * i + 1)
                ax1.imshow(rgb)
                ax1.set_title("RGB")
                ax1.axis("off")

                ax2 = plt.subplot(n, 5, 5 * i + 2)
                ax2.imshow(prior, cmap="gray")
                ax2.set_title("Prior")
                ax2.axis("off")

                ax3 = plt.subplot(n, 5, 5 * i + 3)
                ax3.imshow(gt, cmap="gray")
                ax3.set_title("GT")
                ax3.axis("off")

                ax4 = plt.subplot(n, 5, 5 * i + 4)
                ax4.imshow(pr, cmap="gray", vmin=0, vmax=1)
                ax4.set_title("Pred")
                ax4.axis("off")

                ax5 = plt.subplot(n, 5, 5 * i + 5)
                ax5.imshow(ov)
                ax5.set_title("Overlay")
                ax5.axis("off")

            plt.tight_layout()
            out_path = os.path.join(self.out_dir, f"epoch{epoch_id:03d}_b{bi}.png")
            plt.savefig(out_path, dpi=180)
            plt.close(fig)

            if self.tb_writer is not None:
                try:
                    img = cv2.imread(out_path, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img[None, ...]
                    with self.tb_writer.as_default():
                        tf.summary.image(f"pred/epoch_{epoch_id:03d}_b{bi}", img, step=epoch_id)
                except Exception:
                    pass

        self._cleanup()

    def _cleanup(self):
        keep = int(max(1, self.cfg.keep_last_n_visuals))
        files = sorted(glob(os.path.join(self.out_dir, "*.png")))
        if len(files) > keep:
            for p in files[: len(files) - keep]:
                try:
                    os.remove(p)
                except Exception:
                    pass


def plot_history(history, out_path: str):
    h = history.history
    keys = list(h.keys())
    panels = [
        ("loss", "val_loss", "Loss"),
        ("iou", "val_iou", "IoU"),
        ("recall", "val_recall", "Recall"),
        ("accuracy", "val_accuracy", "Accuracy"),
        ("main_loss", "val_main_loss", "MainLoss"),
    ]
    panels = [(a, b, t) for a, b, t in panels if (a in keys and b in keys)]
    n = max(1, len(panels))
    cols = 2
    rows = int(math.ceil(n / cols))

    fig = plt.figure(figsize=(14, 8))
    for i, (a, b, title) in enumerate(panels, start=1):
        ax = plt.subplot(rows, cols, i)
        ax.plot(h[a], label="train")
        ax.plot(h[b], label="val")
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
