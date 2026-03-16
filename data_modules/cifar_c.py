# data_modules/cifar_c.py
# Purpose: Load CIFAR-C .npy files and build evaluation DataLoaders
# for corruption × severity robustness evaluation.
# Supports model-aware evaluation resizing / preprocessing.

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Optional canonical corruption list
CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
    "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow",
    "spatter", "speckle_noise", "zoom_blur"
]


def _load_c_corruption(root: str, name: str):
    x = np.load(os.path.join(root, f"{name}.npy"))    # typically [50000,32,32,3]
    y = np.load(os.path.join(root, "labels.npy"))     # could be [10000] or [50000]
    return x, y


def _normalize_nchw(x: torch.Tensor, mean=None, std=None) -> torch.Tensor:
    """
    x: float tensor in [0,1], shape [N,C,H,W]
    If mean/std is None, return x unchanged.
    """
    if mean is None or std is None:
        return x
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


def _apply_eval_resize_nchw(
    x: torch.Tensor,
    img_size: int,
    preprocessing: str = "cifar",
) -> torch.Tensor:
    """
    Apply model-aware evaluation resize on a batch tensor [N, C, H, W].
    - cifar: direct resize to img_size (usually no-op for 32)
    - imagenet: resize to ~256/224 * img_size, then center crop to img_size
    """
    preprocessing = str(preprocessing).lower()

    if preprocessing == "imagenet":
        resize_size = max(img_size, int(round(img_size * 256 / 224)))
        x = F.interpolate(x, size=(resize_size, resize_size), mode="bilinear", align_corners=False)

        top = (resize_size - img_size) // 2
        left = (resize_size - img_size) // 2
        x = x[:, :, top:top + img_size, left:left + img_size]
        return x

    # default: direct resize / keep size
    if x.shape[-2] != img_size or x.shape[-1] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)

    return x


def make_cifar_c_loaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    corruptions=None,      # None -> scan directory
    mean=None,
    std=None,
    img_size: int = 32,
    preprocessing: str = "cifar",
):
    """
    Returns:
        loaders: dict[str, list[DataLoader]]
          loaders[corruption_name][s-1] gives DataLoader for severity s in {1..5}
    """
    loaders = {}

    # Backward-compatible directory scan
    if corruptions is None:
        names = []
        for fname in os.listdir(root):
            if fname.endswith(".npy") and fname != "labels.npy":
                names.append(fname.replace(".npy", ""))
        names.sort()
    else:
        names = [c for c in corruptions]

    for cname in names:
        x_all, y_all = _load_c_corruption(root, cname)

        y_all = np.asarray(y_all)
        y_len = int(y_all.shape[0])

        severity_loaders = []
        for s in range(5):
            xs = x_all[s * 10000:(s + 1) * 10000]

            if y_len == 10000:
                ys = y_all
            else:
                ys = y_all[s * 10000:(s + 1) * 10000]

            # NHWC -> NCHW and scale to [0,1]
            xs = torch.from_numpy(xs.transpose(0, 3, 1, 2)).float() / 255.0

            # Model-aware eval resize
            xs = _apply_eval_resize_nchw(
                xs,
                img_size=img_size,
                preprocessing=preprocessing,
            )

            # Normalize after resize/crop
            xs = _normalize_nchw(xs, mean=mean, std=std)

            ys = torch.from_numpy(np.asarray(ys)).long()

            ds = TensorDataset(xs, ys)
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            severity_loaders.append(dl)

        loaders[cname] = severity_loaders

    return loaders