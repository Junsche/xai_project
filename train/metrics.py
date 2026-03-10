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

    ece = 0.0
    bin_edges = torch.linspace(0, 1, steps=n_bins + 1, device=logits.device)

    for i in range(n_bins):
        mask = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue

        acc_bin = (preds[mask] == targets[mask]).float().mean()
        conf_bin = conf[mask].mean()
        ece += mask.float().mean() * torch.abs(acc_bin - conf_bin)

    return ece.item()
