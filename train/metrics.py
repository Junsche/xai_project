# train/metrics.py
# 作用（中文）：训练/验证常用指标；ECE 衡量模型置信度校准情况。
# Purpose (EN): Common metrics; ECE for confidence calibration.

import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

def compute_balanced_acc(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return balanced_accuracy_score(y_true, y_pred)

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def expected_calibration_error(logits, targets, n_bins=15):
    probs = F.softmax(logits, dim=1)
    conf, preds = probs.max(dim=1)
    ece, total = 0.0, conf.numel()
    # 分箱边界 / bin edges
    bin_edges = torch.linspace(0, 1, steps=n_bins+1, device=logits.device)
    for i in range(n_bins):
        mask = (conf > bin_edges[i]) & (conf <= bin_edges[i+1])
        if mask.sum() == 0:
            continue
        acc_bin = (preds[mask] == targets[mask]).float().mean()
        conf_bin = conf[mask].mean()
        ece += (mask.float().mean() * torch.abs(acc_bin - conf_bin))
    return ece.item()

#currently unused
# def balanced_accuracy(outputs, targets, num_classes):
#     # outputs: [N, C] logits
#     # targets: [N]
#     preds = outputs.argmax(dim=1)
#     conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.long, device=targets.device)
#     for t, p in zip(targets.view(-1), preds.view(-1)):
#         conf_mat[t.long(), p.long()] += 1

#     # per-class recall = TP / (TP + FN)
#     tp_fn = conf_mat.sum(dim=1)  # 每行：该类的所有真实样本数
#     tp = conf_mat.diag()
#     # 为防止除以 0
#     valid = tp_fn > 0
#     recall = torch.zeros_like(tp, dtype=torch.float32)
#     recall[valid] = tp[valid].float() / tp_fn[valid].float()

#     ba = recall.mean().item()
#     return ba