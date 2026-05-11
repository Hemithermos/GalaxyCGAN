import torch
import matplotlib.pyplot as plt
from models.generator import Generator

# Config identique à l'entraînement
Z_DIM = 100
NUM_CLASSES = 4
EMBED_DIM = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le générateur
G = Generator(Z_DIM, NUM_CLASSES, EMBED_DIM).to(device)
ckpt = torch.load("checkpoints/ckpt_epoch_300.pt", map_location=device)
G.load_state_dict(ckpt['G'])
G.eval()

# Générer 64 galaxies (16 par classe)
with torch.no_grad():
    z = torch.randn(64, Z_DIM, device=device)
    labels = torch.tensor([0]*16 + [1]*16 + [2]*16 + [3]*16, device=device)
    fake_imgs = G(z, labels)

# Afficher
fake_imgs = (fake_imgs * 0.5 + 0.5).clamp(0, 1).cpu()

class_names = ["Elliptique", "Spirale", "Spirale barrée", "Irrégulière"]

fig, axes = plt.subplots(8, 8, figsize=(16, 16))
for i, ax in enumerate(axes.flatten()):
    img = fake_imgs[i].permute(1, 2, 0).numpy()
    ax.imshow(img)
    ax.axis("off")
    if i % 16 == 0:
        ax.set_title(class_names[i // 16], fontsize=9)

plt.tight_layout()
plt.savefig("outputs/generated_galaxies.png", dpi=150)
print("✅ Sauvegardé : outputs/generated_galaxies.png")

