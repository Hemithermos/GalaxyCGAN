import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm

from models.generator import Generator
from models.discriminator import Discriminator
from training.losses import wasserstein_loss_D, wasserstein_loss_G, gradient_penalty

# ── Config ──────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Z_DIM       = 100
NUM_CLASSES = 4
EMBED_DIM   = 50
BATCH_SIZE  = 64
LR          = 1e-4          # plus faible pour WGAN
BETAS       = (0.0, 0.9)    # recommandé pour WGAN-GP
EPOCHS      = 200
LAMBDA_GP   = 10
N_CRITIC    = 5             # D s'entraîne 5x plus que G
SAVE_EVERY  = 10
IMG_EVERY   = 5

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('outputs/images', exist_ok=True)

# ── Données ─────────────────────────────────────────────
def get_dataloader():
    images = np.load('data/processed/train_images.npy')
    labels = np.load('data/processed/train_labels.npy')
    images = torch.tensor(images).permute(0, 3, 1, 2).float()
    labels = torch.tensor(labels).long()
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# ── Init poids ──────────────────────────────────────────
def weights_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

# ── Main ────────────────────────────────────────────────
def train():
    print(f'Device : {DEVICE}')
    loader = get_dataloader()

    G = Generator(Z_DIM, NUM_CLASSES, EMBED_DIM).to(DEVICE)
    D = Discriminator(NUM_CLASSES).to(DEVICE)
    G.apply(weights_init)
    D.apply(weights_init)

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    fixed_z      = torch.randn(16, Z_DIM).to(DEVICE)
    fixed_labels = torch.tensor([0, 1, 2, 3] * 4).to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        G.train(); D.train()
        loss_D_total = loss_G_total = 0
        g_updates = 0

        for i, (real_imgs, real_labels) in enumerate(tqdm(loader, desc=f'Epoch {epoch}/{EPOCHS}', leave=False)):
            real_imgs   = real_imgs.to(DEVICE)
            real_labels = real_labels.to(DEVICE)
            B = real_imgs.size(0)

            # ── Train D (N_CRITIC fois) ──
            for _ in range(N_CRITIC):
                z           = torch.randn(B, Z_DIM).to(DEVICE)
                fake_labels = torch.randint(0, NUM_CLASSES, (B,)).to(DEVICE)
                fake_imgs   = G(z, fake_labels).detach()

                real_scores = D(real_imgs, real_labels)
                fake_scores = D(fake_imgs, fake_labels)
                gp          = gradient_penalty(D, real_imgs, fake_imgs, fake_labels, DEVICE)
                loss_D      = wasserstein_loss_D(real_scores, fake_scores) + LAMBDA_GP * gp

                opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            loss_D_total += loss_D.item()

            # ── Train G (1 fois par batch) ──
            z           = torch.randn(B, Z_DIM).to(DEVICE)
            fake_labels = torch.randint(0, NUM_CLASSES, (B,)).to(DEVICE)
            fake_imgs   = G(z, fake_labels)
            fake_scores = D(fake_imgs, fake_labels)
            loss_G      = wasserstein_loss_G(fake_scores)

            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
            loss_G_total += loss_G.item()
            g_updates    += 1

        n = len(loader)
        print(f'Epoch {epoch:03d} | Loss D: {loss_D_total/n:.4f} | Loss G: {loss_G_total/max(g_updates,1):.4f}')

        if epoch % IMG_EVERY == 0:
            from utils.visualization import save_image_grid
            G.eval()
            with torch.no_grad():
                imgs = G(fixed_z, fixed_labels)
            save_image_grid(imgs, f'outputs/images/epoch_{epoch:03d}.png')

        if epoch % SAVE_EVERY == 0:
            torch.save({'G': G.state_dict(), 'D': D.state_dict(),
                        'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict(),
                        'epoch': epoch},
                       f'checkpoints/ckpt_epoch_{epoch:03d}.pt')

if __name__ == '__main__':
    train()

