# data/preprocess.py
import numpy as np
from PIL import Image
from pathlib import Path
from galaxy_datasets import galaxy_mnist
from tqdm import tqdm

def load_and_preprocess(catalog, img_size=64):
    images = []
    labels = []
    
    for _, row in tqdm(catalog.iterrows(), total=len(catalog)):
        img = Image.open(row['file_loc']).convert('RGB')
        img = img.resize((img_size, img_size))
        img = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
        images.append(img)
        labels.append(row['label'])
    
    return np.array(images), np.array(labels)

if __name__ == '__main__':
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)

    print('Chargement train...')
    train_catalog, _ = galaxy_mnist(root='data/raw', train=True, download=False)
    print('Chargement test...')
    test_catalog, _  = galaxy_mnist(root='data/raw', train=False, download=False)


    print('Preprocessing train...')
    train_images, train_labels = load_and_preprocess(train_catalog)
    
    print('Preprocessing test...')
    test_images, test_labels = load_and_preprocess(test_catalog)

    np.save(output_dir / 'train_images.npy', train_images)
    np.save(output_dir / 'train_labels.npy', train_labels)
    np.save(output_dir / 'test_images.npy', test_images)
    np.save(output_dir / 'test_labels.npy', test_labels)

    print(f'Train: {train_images.shape}, labels: {np.unique(train_labels)}')
    print(f'Test:  {test_images.shape}')
    print(f'Pixel range: [{train_images.min():.2f}, {train_images.max():.2f}]')

