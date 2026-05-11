# data/download_dataset.py

import os
import argparse
import numpy as np
from pathlib import Path
from galaxy_mnist import GalaxyMNIST, GalaxyMNISTHighrez

# ============================================================
# LABELS
# ============================================================

LABEL_NAMES = {
    0: "smooth_round",
    1: "smooth_cigar",
    2: "edge_on_disk",
    3: "unbarred_spiral"
}

# ============================================================
# DOWNLOAD
# ============================================================

def download_galaxy_mnist(
    root: str = "data/raw/galaxy_mnist",
    highrez: bool = False,
    test_size: float = 0.2
):
    """
    Télécharge GalaxyMNIST
    
    Args:
        root      : Dossier de téléchargement
        highrez   : False → 64x64 | True → 224x224
        test_size : Proportion du test set (default 0.2)
    
    Returns:
        train_images, train_labels, test_images, test_labels
    """

    Path(root).mkdir(parents=True, exist_ok=True)

    size = 224 if highrez else 64

    print("=" * 50)
    print(f"   Téléchargement GalaxyMNIST {'HighRez 224x224' if highrez else '64x64'}")
    print("=" * 50)

    # ---- Téléchargement ----
    print(f"\n Download → {root}")
    dataset = GalaxyMNIST(
        root=root,
        download=True,
        train=True
    )

    # ---- Split custom ----
    print(f"\n  Split stratifié → train:{1-test_size:.0%} / test:{test_size:.0%}")
    (train_images, train_labels), (test_images, test_labels) = \
        dataset.load_custom_data(test_size=test_size, stratify=True)

    # ---- Infos ----
    print(f"\n Dataset chargé :")
    print(f"   Images size  : {size}x{size} px")
    print(f"   Train        : {len(train_images)} images")
    print(f"   Test         : {len(test_images)} images")
    print(f"   Shape        : {train_images.shape}  (uint8)")
    print(f"   Classes      :")
    for label_id, label_name in LABEL_NAMES.items():
        n_train = (train_labels == label_id).sum()
        n_test  = (test_labels  == label_id).sum()
        print(f"     [{label_id}] {label_name:<20} → train: {n_train} | test: {n_test}")

    return train_images, train_labels, test_images, test_labels


# ============================================================
# VÉRIFICATION
# ============================================================

def verify_dataset(train_images, train_labels, test_images, test_labels):
    """
    Vérifie que le dataset est bien formé
    """
    print("\n Vérification...")

    assert train_images.dtype == np.uint8, "dtype doit être uint8"
    assert train_images.shape[-1] == 3,    "images doivent être RGB"
    assert len(train_images) == len(train_labels), "mismatch images/labels"
    assert len(test_images)  == len(test_labels),  "mismatch images/labels"
    assert set(np.unique(train_labels)).issubset(set(LABEL_NAMES.keys())), \
        "labels inconnus détectés"

    print(f"    dtype     : {train_images.dtype}")
    print(f"    shape     : {train_images.shape}")
    print(f"    min/max   : {train_images.min()} / {train_images.max()}")
    print(f"    labels    : {np.unique(train_labels)}")
    print("    Dataset valide !\n")


# ============================================================
# SAVE (numpy)
# ============================================================

def save_dataset(
    train_images, train_labels,
    test_images,  test_labels,
    output_dir: str = "data/processed"
):
    """
    Sauvegarde le dataset en .npy pour un chargement rapide ensuite
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Sauvegarde dans {output_dir}...")

    np.save(output_path / "train_images.npy", train_images)
    np.save(output_path / "train_labels.npy", train_labels)
    np.save(output_path / "test_images.npy",  test_images)
    np.save(output_path / "test_labels.npy",  test_labels)

    # Taille des fichiers
    for f in output_path.glob("*.npy"):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"{f.name:<25} {size_mb:.1f} MB")

    print("Sauvegarde terminée !")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Télécharge et prépare GalaxyMNIST"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/raw/galaxy_mnist",
        help="Dossier de téléchargement"
    )
    parser.add_argument(
        "--highrez",
        action="store_true",
        help="Utiliser 224x224 au lieu de 64x64"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion du test set (default: 0.2)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Dossier de sauvegarde des .npy"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Ne pas sauvegarder en .npy"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Download
    train_images, train_labels, test_images, test_labels = download_galaxy_mnist(
        root=args.root,
        highrez=args.highrez,
        test_size=args.test_size
    )

    # 2. Vérification
    verify_dataset(train_images, train_labels, test_images, test_labels)

    # 3. Sauvegarde
    if not args.no_save:
        save_dataset(
            train_images, train_labels,
            test_images,  test_labels,
            output_dir=args.output_dir
        )



if __name__ == "__main__":
    main()

