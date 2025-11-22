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
    EN:
      Evaluate model on a loader.
      - Always returns accuracy
      - Optionally returns ECE (do_ece)
      - Optionally returns balanced accuracy (do_bal_acc), for medical datasets.
    ZH：
      在给定的 loader 上评估模型：
      - 始终返回普通 accuracy
      - 可选计算 ECE（do_ece）
      - 可选计算 Balanced Accuracy（do_bal_acc，用于医学数据集）
    """
    model.eval()
    tot_acc = 0.0
    tot_ece = 0.0
    tot_n = 0

    # 如果要算 balanced accuracy，需要收集所有 logits 和 labels
    # If we need balanced accuracy, we must store all logits & labels.
    all_logits = [] if do_bal_acc else None
    all_labels = [] if do_bal_acc else None

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        bs = x.size(0)

        # 1) 普通 accuracy（跟你原来的代码一样：加权平均）
        #    Standard accuracy (same as before: weighted average).
        tot_acc += accuracy(logits, y) * bs

        # 2) ECE 按 batch 加权平均
        #    ECE as a weighted average over batches.
        if do_ece:
            tot_ece += expected_calibration_error(logits, y) * bs

        # 3) 如果启用 balanced accuracy，就先把 logits & labels 存起来
        #    If we want balanced accuracy, store logits & labels for later.
        if do_bal_acc:
            all_logits.append(logits.detach())
            all_labels.append(y.detach())

        tot_n += bs

    acc = tot_acc / tot_n
    ece = (tot_ece / tot_n) if do_ece else None

    # 4) 计算 Balanced Accuracy（只在医学数据集时开启）
    #    Compute balanced accuracy only when requested (e.g., medical datasets).
    bal_acc = None
    if do_bal_acc:
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        _, preds = logits_cat.max(dim=1)
        bal_acc = compute_balanced_acc(labels_cat, preds)

    return acc, ece, bal_acc

def train_one_epoch(model, loader, optimizer, device, scaler=None, cfg_train=None):
    model.train(); tot_acc=tot_n=0
    use_mixup = cfg_train is not None and "mixup_alpha"  in cfg_train
    use_cm   = cfg_train is not None and "cutmix_alpha" in cfg_train
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_mixup: x, y = _mixup(x, y, cfg_train["mixup_alpha"])
        if use_cm:    x, y = _cutmix(x, y, cfg_train["cutmix_alpha"])
        if scaler:
            with torch.cuda.amp.autocast():
                lg = model(x); loss = _criterion(lg, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            lg = model(x); loss = _criterion(lg, y)
            loss.backward(); optimizer.step()
        tot_acc += accuracy(lg, y if not isinstance(y,tuple) else y[0]) * x.size(0)
        tot_n += x.size(0)
    return tot_acc/tot_n