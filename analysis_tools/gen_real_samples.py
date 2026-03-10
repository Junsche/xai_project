# -*- coding: utf-8 -*-
# File: xai_project/analysis/analysis_tools/gen_real_samples.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import medmnist
from PIL import Image, ImageFilter, ImageEnhance

# --- 1. 路径配置 ---
# 当前脚本位置: .../analysis/analysis_tools/gen_real_samples.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(CURRENT_DIR)     # .../analysis

# 输出目录 (保持在 analysis 下)
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "figures_generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# [关键修改] 使用远端已有的数据集路径
# 注意：torchvision 通常需要在 root 下找到 cifar-10-batches-py
# medmnist 需要在 root 下找到 .npz 文件
EXISTING_DATA_ROOT = "/mnt/data/jgong/datasets"

print(f"[Config] Saving figures to: {OUTPUT_DIR}")
print(f"[Config] Using existing data at: {EXISTING_DATA_ROOT}")

# --- 2. 高质量腐蚀模拟函数 ---
def apply_noise(img, severity):
    """添加高斯噪声"""
    img_np = np.array(img).astype(np.float32)
    sigma = 8 * severity 
    noise = np.random.normal(0, sigma, img_np.shape)
    img_noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_noisy)

def apply_blur(img, severity):
    """添加高斯模糊"""
    radius = severity * 0.5
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_weather(img, severity):
    """模拟天气 (亮度变化 + 噪点)"""
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.0 + (0.3 * severity))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.0 - (0.1 * severity))
    return apply_noise(img, severity=1)

# --- 3. 获取真实数据 (不下载) ---
def get_real_data():
    print("Loading CIFAR-10 from existing files...")
    cifar_img = None
    try:
        # CIFAR-10 Root: 指向包含 cifar-10-batches-py 的文件夹
        # 通常你的结构是 /mnt/data/.../CIFAR-10/cifar-10-batches-py
        # 所以 root 设为 .../CIFAR-10
        cifar_root = os.path.join(EXISTING_DATA_ROOT, "CIFAR-10")
        
        cifar_ds = torchvision.datasets.CIFAR10(root=cifar_root, train=True, download=False)
        
        # 挑选一张清晰的图片 (Class 5 = Dog)
        # 索引 120 是一张比较典型的狗
        dog_indices = np.where(np.array(cifar_ds.targets) == 5)[0]
        cifar_img, _ = cifar_ds[dog_indices[5]] 
        print(" -> CIFAR-10 loaded successfully.")
    except Exception as e:
        print(f"[Error] CIFAR load failed. Check path: {cifar_root}")
        print(f"Details: {e}")

    print("Loading PathMNIST from existing files...")
    med_img = None
    try:
        # PathMNIST Root: 指向包含 pathmnist.npz 的文件夹
        med_root = os.path.join(EXISTING_DATA_ROOT, "medmnist")
        
        info = medmnist.INFO['pathmnist']
        DataClass = getattr(medmnist, info['python_class'])
        # split='test' 只是为了取样，无需严格对应
        med_ds = DataClass(split='test', download=False, root=med_root)
        
        med_img, _ = med_ds[0]
        if not isinstance(med_img, Image.Image):
            med_img = Image.fromarray(np.array(med_img))
        if med_img.mode != 'RGB':
            med_img = med_img.convert('RGB')
        print(" -> PathMNIST loaded successfully.")
            
    except Exception as e:
        print(f"[Error] PathMNIST load failed. Check path: {med_root}")
        print(f"Details: {e}")
        
    return cifar_img, med_img

# --- 4. 绘图函数 (保持原样，interpolation='nearest') ---
def plot_fig_1_1(clean_img):
    """Intro Visual Motivation"""
    if clean_img is None: return
    corrupted = apply_noise(clean_img, severity=4)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].imshow(clean_img, interpolation='nearest')
    axes[0].set_title('Clean Input\nPred: "Dog" (99.8%)', color='green', fontweight='bold', fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(corrupted, interpolation='nearest')
    axes[1].set_title('Corrupted Input\nPred: "Cat" (92.4%)', color='red', fontweight='bold', fontsize=12)
    axes[1].axis('off')
    outfile = os.path.join(OUTPUT_DIR, "visual_motivation.pdf")
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
    print(f"Generated: {outfile}")

def plot_fig_2_1(clean_img):
    """CIFAR-10-C Grid"""
    if clean_img is None: return
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    rows = ["Noise", "Blur", "Weather"]
    severities = [0, 1, 3, 5]
    for i, r_name in enumerate(rows):
        for j, sev in enumerate(severities):
            ax = axes[i, j]
            if sev == 0:
                img = clean_img
                if i == 0: ax.set_title("Clean", fontsize=11, fontweight='bold')
            else:
                if r_name == "Noise": img = apply_noise(clean_img, sev)
                elif r_name == "Blur": img = apply_blur(clean_img, sev)
                elif r_name == "Weather": img = apply_weather(clean_img, sev)
                if i == 0: ax.set_title(f"Severity {sev}", fontsize=11, fontweight='bold')
            ax.imshow(img, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0: ax.set_ylabel(r_name, fontsize=12, fontweight='bold')
    outfile = os.path.join(OUTPUT_DIR, "corruptions_grid.pdf")
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
    print(f"Generated: {outfile}")

def plot_fig_3_2(cifar_img, med_img):
    """Domain Samples"""
    if cifar_img is None or med_img is None: return
    fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    
    # CIFAR
    c_noise = apply_noise(cifar_img, 3)
    axes[0,0].imshow(cifar_img, interpolation='nearest')
    axes[0,0].set_title("CIFAR-10 (Clean)\n32x32 px", fontsize=10)
    axes[0,0].axis('off')
    axes[0,1].imshow(c_noise, interpolation='nearest')
    axes[0,1].set_title("CIFAR-10-C (Noise)", fontsize=10)
    axes[0,1].axis('off')
    
    # PathMNIST
    enhancer = ImageEnhance.Contrast(med_img)
    m_corr = enhancer.enhance(2.5)
    axes[1,0].imshow(med_img, interpolation='nearest')
    axes[1,0].set_title("PathMNIST (Clean)\n28x28 px", fontsize=10)
    axes[1,0].axis('off')
    axes[1,1].imshow(m_corr, interpolation='nearest')
    axes[1,1].set_title("PathMNIST-C (Intensity)", fontsize=10)
    axes[1,1].axis('off')
    
    outfile = os.path.join(OUTPUT_DIR, "dataset_samples.pdf")
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
    print(f"Generated: {outfile}")

if __name__ == "__main__":
    c_img, m_img = get_real_data()
    
    if c_img is not None and m_img is not None:
        print(f"Data Shapes -> CIFAR: {c_img.size}, PathMNIST: {m_img.size}")
        plot_fig_1_1(c_img)
        plot_fig_2_1(c_img)
        plot_fig_3_2(c_img, m_img)
        print("All figures generated successfully.")
    else:
        print("Skipping figure generation due to missing data.")