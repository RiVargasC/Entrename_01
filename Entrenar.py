import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from tqdm import tqdm
from RRDBNet_arch import RRDBNet
from utils import get_data_loaders
from google.colab import drive

# Montar Google Drive
drive.mount('/content/drive')

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hiperparámetros
lr_dir = 'LR'
hr_dir = 'HR'
model_save_dir = '/content/drive/My Drive/CarpetaDeModelos'  # Guardar en Google Drive
num_epochs = 50
batch_size = 16
learning_rate = 1e-4
num_workers = 4

# Crear la carpeta de modelos si no existe
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# Obtener dataloaders
train_loader = get_data_loaders(lr_dir, hr_dir, batch_size=batch_size, num_workers=num_workers)

# Definir el modelo
model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
model = model.to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Función para encontrar el último checkpoint guardado
def find_last_checkpoint(model_save_dir):
    checkpoints = [f for f in os.listdir(model_save_dir) if f.startswith('RRDB_ESRGAN_x3_epoch_')]
    if not checkpoints:
        return None, 0
    checkpoints.sort()
    last_checkpoint = checkpoints[-1]
    last_epoch = int(last_checkpoint.split('_')[-1].split('.')[0])
    return os.path.join(model_save_dir, last_checkpoint), last_epoch

# Cargar el último checkpoint si existe
last_checkpoint_path, start_epoch = find_last_checkpoint(model_save_dir)
if last_checkpoint_path:
    model.load_state_dict(torch.load(last_checkpoint_path))
    print(f'Model loaded from {last_checkpoint_path}')

# Bucle de entrenamiento
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0
    for lr_imgs, hr_imgs in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        # Forward pass
        sr_imgs = model(lr_imgs)
        sr_imgs = F.interpolate(sr_imgs, size=hr_imgs.shape[2:], mode='bicubic', align_corners=False)
        loss = criterion(sr_imgs, hr_imgs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Guardar el modelo después de cada época
    current_checkpoint_path = os.path.join(model_save_dir, f'RRDB_ESRGAN_x2_epoch_{epoch+1:03d}.pth')
    torch.save(model.state_dict(), current_checkpoint_path)
    
    # Borrar el checkpoint anterior si no es el primer epoch
    if epoch > start_epoch:
        previous_checkpoint_path = os.path.join(model_save_dir, f'RRDB_ESRGAN_x2_epoch_{epoch:03d}.pth')
        if os.path.exists(previous_checkpoint_path):
            os.remove(previous_checkpoint_path)

print('Entrenamiento completado y modelo guardado.')
