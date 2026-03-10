# Load MedMNIST-C (Zenodo zip extracted) stored as per-corruption .npz files.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -----------------------
# helpers
# -----------------------
def _find_xy(npz):
    keys = list(npz.keys())

    # Standard naming
    if "images" in keys and "labels" in keys:
        return npz["images"], npz["labels"]

    # MedMNIST official naming
    if "test_images" in keys and "test_labels" in keys:
        return npz["test_images"], npz["test_labels"]

    # Alternative common naming
    if "x" in keys and "y" in keys:
        return npz["x"], npz["y"]

    raise KeyError(
        f"Cannot find image array in npz. Keys={keys}. "
        f"Expected one of: (images, labels), (test_images, test_labels), (x, y)"
    )

def _maybe_select_severity(x: np.ndarray, y: np.ndarray, severity: int):
    """
    Support two MedMNIST-C severity layouts:

    A) axis layout: x shape [5, N, H, W, C] (or [5, N, H, W])
    B) stacked layout: x shape [5*N, H, W, C] where severities are concatenated
       along the sample dimension in order (sev1 block, sev2 block, ...).

    severity in 1..5
    """
    if severity is None:
        return x, y

    s = int(severity) - 1
    if s < 0:
        raise ValueError("severity must be >= 1")

    # ---- A) axis layout ----
    if x.ndim >= 5 and x.shape[0] in [5, 6]:
        x_sel = x[s]
        if isinstance(y, np.ndarray) and y.ndim >= 2 and y.shape[0] == x.shape[0]:
            y_sel = y[s]
        else:
            y_sel = y
        return x_sel, y_sel

    # ---- B) stacked layout ----
    # Typical: x shape (M, H, W, C) and M divisible by 5
    if x.ndim == 4 and x.shape[0] % 5 == 0:
        n = x.shape[0] // 5
        x_sel = x[s * n : (s + 1) * n]
        if isinstance(y, np.ndarray) and y.shape[0] == x.shape[0]:
            y_sel = y[s * n : (s + 1) * n]
        else:
            y_sel = y
        return x_sel, y_sel

    # (Optional) handle grayscale stacked (M, H, W)
    if x.ndim == 3 and x.shape[0] % 5 == 0:
        n = x.shape[0] // 5
        x_sel = x[s * n : (s + 1) * n]
        if isinstance(y, np.ndarray) and y.shape[0] == x.shape[0]:
            y_sel = y[s * n : (s + 1) * n]
        else:
            y_sel = y
        return x_sel, y_sel

    # fallback: no severity structure detected
    return x, y


class NpzMedMNISTCDataset(Dataset):
    def __init__(self, npz_path: Path, tf):
        self.npz_path = Path(npz_path)
        self.tf = tf

        npz = np.load(self.npz_path, allow_pickle=True)
        x, y = _find_xy(npz)

        # convert labels to 1D int
        y = np.array(y)
        if y.ndim == 0:
            y = y.reshape(1)
        if y.ndim > 1:
            # if shape is (N,1) or one-hot/multi-label
            if y.shape[-1] == 1:
                y = y.squeeze(-1)
            else:
                y = y.argmax(axis=-1)

        self.x = x
        self.y = y.astype(np.int64)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        img = self.x[idx]
        label = int(self.y[idx])

        # img could be uint8 already; ensure HWC for ToPILImage
        # Possible shapes:
        # - (H, W) grayscale
        # - (H, W, C) RGB
        if img.ndim == 2:
            pass
        elif img.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        img = self.tf(img)
        return img, label


def make_medmnist_c_loader(
    *,
    c_root: str,
    dataset: str,          # "dermamnist" or "pathmnist"
    corruption: str,       # e.g., "gaussian_noise"
    severity: Optional[int],
    batch_size: int,
    num_workers: int,
    img_size: int,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
):
    c_root = Path(c_root)
    npz_path = c_root / dataset / f"{corruption}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing npz: {npz_path}")

    # test transforms only (no random aug in stage3)
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # load once to resolve severity axis (if any)
    npz = np.load(npz_path, allow_pickle=True)
    x, y = _find_xy(npz)
    x, y = _maybe_select_severity(x, y, severity)

    # write a small temp in-memory dataset without re-reading file:
    # simplest: wrap arrays directly
    class _ArrDS(Dataset):
        def __init__(self, x, y, tf):
            self.x = x
            self.y = y
            self.tf = tf
        def __len__(self): return int(self.x.shape[0])
        def __getitem__(self, i):
            img = self.x[i]
            lab = self.y[i]
            lab = int(lab) if np.ndim(lab) == 0 else int(np.asarray(lab).reshape(-1)[0])
            return self.tf(img), lab

    # normalize label format
    y = np.array(y)
    if y.ndim > 1:
        if y.shape[-1] == 1:
            y = y.squeeze(-1)
        else:
            y = y.argmax(axis=-1)
    y = y.astype(np.int64)

    ds = _ArrDS(x, y, tf)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader