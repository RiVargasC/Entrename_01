import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import cv2

class ImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_img_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_img_path = os.path.join(self.hr_dir, self.hr_images[idx])

        # Leer imágenes GeoTIFF con rasterio
        with rasterio.open(lr_img_path) as lr_dataset:
            lr_img = lr_dataset.read()
            lr_img = np.moveaxis(lr_img, 0, -1)  # Reordenar canales a (H, W, C)

        with rasterio.open(hr_img_path) as hr_dataset:
            hr_img = hr_dataset.read()
            hr_img = np.moveaxis(hr_img, 0, -1)  # Reordenar canales a (H, W, C)

        # Redimensionar las imágenes a las dimensiones esperadas
        lr_img = cv2.resize(lr_img, (64, 64), interpolation=cv2.INTER_AREA)
        hr_img = cv2.resize(hr_img, (128, 128), interpolation=cv2.INTER_AREA)

        # Convertir a tensores directamente, ya están normalizadas
        lr_img = torch.tensor(lr_img, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        hr_img = torch.tensor(hr_img, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)

        return lr_img, hr_img

def get_data_loaders(lr_dir, hr_dir, batch_size=16, num_workers=4):
    dataset = ImageDataset(lr_dir, hr_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader
