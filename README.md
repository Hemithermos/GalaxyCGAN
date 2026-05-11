# Galaxy GAN

Génération de galaxies par classe avec un GAN conditionnel (cGAN).

## Classes
- 0 : Elliptique
- 1 : Spirale
- 2 : Spirale barrée
- 3 : Irrégulière


## Structure

galaxy-gan/
├── models/         # Generator & Discriminator
├── train/          # Script d'entraînement
├── test/           # Génération d'images
├── data/           # Dataset (non versionné)
└── checkpoints/    # Modèles sauvegardés (non versionnés)


## Usage
```bash
python -m train.train
python -m test.generate
```
