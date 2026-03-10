import torch.nn as nn

def build_model(model_cfg, num_classes: int):
    """
    Build a model according to configuration.

    Supported:
        - resnet18 with optional ImageNet pretrained weights
    """
    name = model_cfg["name"].lower()
    weights_key = str(model_cfg.get("weights", "none")).lower()

    if name == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights

        if weights_key in ["imagenet1k_v1", "imagenet", "default"]:
            weights = ResNet18_Weights.IMAGENET1K_V1
        elif weights_key in ["none", "null", "false", "0"]:
            weights = None
        else:
            raise ValueError(f"Unknown weights option: {weights_key}")

        model = resnet18(weights=weights)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model: {name}")
