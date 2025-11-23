import torch, torch.nn.functional as F, random
from .metrics import accuracy, expected_calibration_error, compute_balanced_acc

def _mixup(x, y, alpha):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], (y, y[idx], lam)

def _cutmix(x, y, alpha):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B, C, H, W = x.size()
    cx, cy = random.randrange(W), random.randrange(H)
    bw, bh = int(W*(1-lam)**0.5), int(H*(1-lam)**0.5)
    x1, y1 = max(cx-bw//2,0), max(cy-bh//2,0)
    x2, y2 = min(cx+bw//2,W), min(cy+bh//2,H)
    idx = torch.randperm(B, device=x.device)
    x2v = x.clone()
    x2v[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_eff = 1 - (x2-x1)*(y2-y1)/(W*H)
    return x2v, (y, y[idx], lam_eff)

def _criterion(logits, target):
    if isinstance(target, tuple):
        y1, y2, lam = target
        return lam*F.cross_entropy(logits, y1) + (1-lam)*F.cross_entropy(logits, y2)
    return F.cross_entropy(logits, target)

@torch.no_grad()
def evaluate(model, loader, device, do_ece: bool = True, do_bal_acc: bool = False):
    """
    Returns:
        acc, ece, mce, bal_acc, avg_loss
    """
    model.eval()
    tot_correct = 0
    tot_loss = 0.0
    tot_n = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()
            bs = x.size(0)

            tot_correct += correct
            tot_loss += loss.item() * bs
            tot_n += bs

            if do_ece or do_bal_acc:
                all_logits.append(logits.detach())
                all_labels.append(y.detach())

    acc = tot_correct / tot_n
    avg_loss = tot_loss / tot_n

    # default
    ece = None
    mce = None
    bal_acc = None

    if (do_ece or do_bal_acc) and len(all_logits) > 0:
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)

        # --- ECE + MCE ---
        if do_ece:
            probs = F.softmax(logits_cat, dim=1)
            confs, preds = probs.max(dim=1)
            correct = (preds == labels_cat).float()

            ece_tensor = torch.zeros(1, device=device)
            mce_tensor = torch.zeros(1, device=device)

            n_bins = 15
            bins = torch.linspace(0, 1, n_bins + 1, device=device)

            for i in range(n_bins):
                lo, hi = bins[i], bins[i+1]
                mask = (confs > lo) & (confs <= hi)
                m = mask.sum()

                if m == 0:
                    continue

                conf_bin = confs[mask].mean()
                acc_bin = correct[mask].mean()
                gap = torch.abs(conf_bin - acc_bin)

                ece_tensor += (m / len(confs)) * gap
                mce_tensor = torch.max(mce_tensor, gap)

            ece = ece_tensor.item()
            mce = mce_tensor.item()

        # --- Balanced Accuracy ---
        if do_bal_acc:
            _, preds = logits_cat.max(dim=1)
            from sklearn.metrics import balanced_accuracy_score
            bal_acc = balanced_accuracy_score(
                labels_cat.cpu().numpy(),
                preds.cpu().numpy()
            )

    return acc, ece, mce, bal_acc, avg_loss

def train_one_epoch(model, loader, optimizer, device, scaler=None, cfg_train=None):
    model.train()
    tot_acc = 0.0
    tot_loss = 0.0
    tot_n = 0

    use_mixup = cfg_train is not None and "mixup_alpha"  in cfg_train
    use_cm   = cfg_train is not None and "cutmix_alpha" in cfg_train

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_mixup:
            x, y = _mixup(x, y, cfg_train["mixup_alpha"])
        if use_cm:
            x, y = _cutmix(x, y, cfg_train["cutmix_alpha"])

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = _criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = _criterion(logits, y)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        # accuracy: 对 mixup/cutmix 取 y[0] 作为“原始标签”
        acc_target = y if not isinstance(y, tuple) else y[0]
        tot_acc += accuracy(logits, acc_target) * bs
        tot_loss += loss.item() * bs
        tot_n += bs

    avg_acc = tot_acc / tot_n
    avg_loss = tot_loss / tot_n
    return avg_acc, avg_loss