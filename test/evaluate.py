import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image
from models.generator import Generator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Z_DIM = 100
NUM_CLASSES = 4
EMBED_DIM = 50
N_SAMPLES = 1000  # par classe 4000 total

# Dossiers
real_dir  = Path('outputs/eval/real')
fake_dir  = Path('outputs/eval/fake')
real_dir.mkdir(parents=True, exist_ok=True)
fake_dir.mkdir(parents=True, exist_ok=True)

# Charge le générateur
G = Generator(Z_DIM, NUM_CLASSES, EMBED_DIM).to(DEVICE)
ckpt = torch.load('checkpoints/ckpt_epoch_100.pt', map_location=DEVICE)
G.load_state_dict(ckpt['G'])
G.eval()

# Sauvegarde les vraies images
print("Sauvegarde des vraies images...")
images = np.load('data/processed/train_images.npy')  # (N, H, W, C)
for i in range(min(len(images), N_SAMPLES * NUM_CLASSES)):
    img = Image.fromarray((images[i] * 255).astype(np.uint8) if images[i].max() <= 1.0 
                           else images[i].astype(np.uint8))
    img.save(real_dir / f'{i:05d}.png')

print(f"→ {i+1} vraies images sauvegardées")

# Génère les fausses images
print("Génération des fausses images...")
count = 0
with torch.no_grad():
    for cls in range(NUM_CLASSES):
        for batch_start in range(0, N_SAMPLES, 64):
            batch_size = min(64, N_SAMPLES - batch_start)
            z = torch.randn(batch_size, Z_DIM).to(DEVICE)
            labels = torch.full((batch_size,), cls, dtype=torch.long).to(DEVICE)
            imgs = G(z, labels).cpu()  # (B, C, H, W) in [-1,1] or [0,1]
            imgs = imgs.clamp(0, 1)
            for img in imgs:
                save_image(img, fake_dir / f'{count:05d}.png')
                count += 1

print(f"→ {count} fausses images sauvegardées")

# Calcul FID
print("\nCalcul du FID...")
from cleanfid import fid
score = fid.compute_fid(str(real_dir), str(fake_dir))
print(f"\n{'='*40}")
print(f"  FID Score : {score:.2f}")
print(f"{'='*40}")
print("(Plus bas = meilleur | <50 bien | <20 excellent)")
print("Real images stats:", images.min(), images.max(), images.dtype)

G_out = G(torch.randn(4, Z_DIM).to(DEVICE), torch.tensor([0,1,2,3]).to(DEVICE))
print("Fake images stats:", G_out.min().item(), G_out.max().item())

