
# test/evaluate_progression.py
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from models.generator import Generator

Z_DIM       = 100
NUM_CLASSES = 4
EMBED_DIM   = 50
EPOCHS      = list(range(10, 300, 10))  # 17 epochs
N_SAMPLES   = 4  # une image par classe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Bruit fixe — même input pour toutes les epochs
fixed_z      = torch.randn(N_SAMPLES, Z_DIM, device=device)
fixed_labels = torch.tensor([0, 1, 2, 3], device=device)

class_names = ["Elliptique", "Spirale", "Sp. barrée", "Irrégulière"]


# Collecter toutes les images
all_imgs = {}  # epoch -> tensor (4, C, H, W)

for i in EPOCHS:
    ckpt_path = f"checkpoints/ckpt_epoch_{i:03d}.pt"
    if not Path(ckpt_path).exists():
        print(f"✗ {ckpt_path} introuvable, skip")
        continue

    ckpt = torch.load(ckpt_path, map_location=device)
    G = Generator(Z_DIM, NUM_CLASSES, EMBED_DIM).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    with torch.no_grad():
        fake = G(fixed_z, fixed_labels)

    fake = (fake * 0.5 + 0.5).clamp(0, 1).cpu()
    all_imgs[i] = fake
    print(f"✓ epoch {i:03d}")

# ── Frise : lignes = classes, colonnes = epochs ──────────
epochs_done = sorted(all_imgs.keys())
n_epochs    = len(epochs_done)
n_classes   = N_SAMPLES

fig, axes = plt.subplots(
    n_classes, n_epochs,
    figsize=(n_epochs * 1.6, n_classes * 1.8)
)

for col, epoch in enumerate(epochs_done):
    for row in range(n_classes):
        ax = axes[row, col]
        img = all_imgs[epoch][row].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis("off")

        if row == 0:
            ax.set_title(f"ep {epoch}", fontsize=7, pad=2)
        if col == 0:
            ax.set_ylabel(class_names[row], fontsize=8, rotation=90, labelpad=4)

plt.suptitle("Progression de l'entraînement — même input", fontsize=12, y=1.01)
plt.tight_layout()

out = "outputs/progression_frise.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nFrise sauvegardée dans {out}")

