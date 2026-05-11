import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from models.generator import Generator

DEVICE = 'cpu'
Z_DIM = 100
NUM_CLASSES = 4
EMBED_DIM = 50

import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Dataloader
images = np.load('data/processed/train_images.npy')
labels = np.load('data/processed/train_labels.npy')

images = torch.tensor(images).float()
labels = torch.tensor(labels).long()
images = torch.tensor(images).float()
# Si shape est (N, H, W, C) → convertir en (N, C, H, W)
if images.shape[-1] == 3:
    images = images.permute(0, 3, 1, 2)

dataset = TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# Modèle
G = Generator(Z_DIM, NUM_CLASSES, EMBED_DIM).to(DEVICE)
ckpt = torch.load('checkpoints/ckpt_epoch_100.pt', map_location=DEVICE)
G.load_state_dict(ckpt['G'])
G.eval()

# Images réelles
real_imgs, _ = next(iter(dataloader))

# Images générées
z = torch.randn(4, Z_DIM).to(DEVICE)
labels = torch.tensor([0, 1, 2, 3]).to(DEVICE)
with torch.no_grad():
    fake_imgs = G(z, labels)

# Plot
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(4):
    axes[0, i].imshow(real_imgs[i].permute(1,2,0).clamp(0,1))
    axes[0, i].axis('off')
    axes[1, i].imshow(fake_imgs[i].permute(1,2,0).clamp(0,1))
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Réel',   fontsize=12)
axes[1, 0].set_ylabel('Généré', fontsize=12)

plt.savefig('outputs/real_vs_fake.png', dpi=150, bbox_inches='tight')
print("Sauvegardé : outputs/real_vs_fake.png")

