# models/factory.py
# 作用（中文）：按名字构建模型，为后续扩展预留统一入口。
# Purpose (EN): Build model by name; single factory for future extensions.

import torchvision.models as tvm
import torch.nn as nn

def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "resnet18":
        m = tvm.resnet18(weights=None)  # 不用预训练权重 / no pretrained
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unknown model: {name}")