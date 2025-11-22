# 用途：把 CIFAR-C 的 .npy 文件按 corruption & severity 切片并生成 DataLoader。
import os, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
CORRUPTIONS = [
    "brightness","contrast","defocus_blur","elastic_transform","fog","frost",
    "gaussian_blur","gaussian_noise","glass_blur","impulse_noise",
    "jpeg_compression","motion_blur","pixelate","saturate","shot_noise","snow",
    "spatter","speckle_noise","zoom_blur"
]
# 上面含 19 种；CIFAR-C 经典集合常用 15 种，你可根据你下载的目录微调列表。

def _load_c_corruption(root, name):
    x = np.load(os.path.join(root, f"{name}.npy"))  # [50000,32,32,3] * 5 severities 拼接
    y = np.load(os.path.join(root, "labels.npy"))   # [50000]
    return x, y

def make_cifar_c_loaders(root, batch_size=128, num_workers=4):
    loaders = {}
    for cname in os.listdir(root):
        if not cname.endswith(".npy") or cname=="labels.npy": continue
        cname = cname.replace(".npy","")
        x_all, y_all = _load_c_corruption(root, cname)
        severities = []
        for s in range(5):
            xs = x_all[s*10000:(s+1)*10000]
            ys = y_all[s*10000:(s+1)*10000]
            # NHWC -> NCHW
            xs = torch.from_numpy(xs.transpose(0,3,1,2)).float()/255.0
            ys = torch.from_numpy(ys).long()
            ds = TensorDataset(xs, ys)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            severities.append(dl)
        loaders[cname] = severities
    return loaders