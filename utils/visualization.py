import torch
import matplotlib
matplotlib.use('Agg')  # ← AVANT tout autre import matplotlib
import matplotlib.pyplot as plt

import numpy as np

def save_image_grid(imgs, path, nrow=4):
    """
    imgs : tensor (N, 3, H, W) en [-1, 1]
    """
    imgs = imgs.cpu().detach()
    imgs = (imgs + 1) / 2  # [-1,1] -> [0,1]
    imgs = imgs.clamp(0, 1)

    N = imgs.size(0)
    ncols = nrow
    nrows = (N + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i in range(N):
        img = imgs[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')

    # cacher les axes vides
    for i in range(N, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f'💾 Image sauvegardée : {path}')


def show_real_vs_fake(real_imgs, fake_imgs, n=8):
    """
    Affiche n images réelles vs générées côte à côte
    """
    real_imgs = (real_imgs[:n].cpu() + 1) / 2
    fake_imgs = (fake_imgs[:n].cpu() + 1) / 2

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        axes[0, i].imshow(real_imgs[i].permute(1, 2, 0).clamp(0,1))
        axes[0, i].axis('off')
        axes[1, i].imshow(fake_imgs[i].permute(1, 2, 0).clamp(0,1))
        axes[1, i].axis('off')

    axes[0, 0].set_title('Réel',  fontsize=12)
    axes[1, 0].set_title('Généré', fontsize=12)
    plt.tight_layout()
    plt.show()

