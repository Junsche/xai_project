# utils/seed.py
# 作用（中文）：固定 Python/NumPy/PyTorch 的随机状态；设定 cuDNN 的确定性。
# Purpose (EN): Set deterministic behavior for Python/NumPy/PyTorch + cuDNN.

import os, random, numpy as np
import torch

def seed_everything(seed: int = 1437):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN: 确定性 + 关闭 benchmark（保证可复现）
    # cuDNN: deterministic + disable benchmark for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 可选：环境变量（某些库会读取）
    # Optional: some libs read this env var
    os.environ["PYTHONHASHSEED"] = str(seed)