import torch
import rasterio
from rasterio.plot import reshape_as_image, reshape_as_raster
from RRDBNet_arch import RRDBNet
import numpy as np

# Ruta del modelo entrenado
model_checkpoint_path = "models/RRDB_ESRGAN_x2_epoch_200.pth"

# Definir el modelo RRDBNet con los mismos parámetros que se usaron durante el entrenamiento
model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# Función para leer y procesar la imagen GeoTIFF
def read_geotiff(path):
    with rasterio.open(path) as src:
        image = src.read(out_dtype="float32")  # Leer la imagen como float32
        profile = src.profile  # Guardar el perfil para escritura posterior
        image = reshape_as_image(image)  # Convertir a formato [H, W, C]
    return image, profile

# Función para aplicar el modelo a la imagen
def super_resolve(image, model, scale_factor=2):
    # Normalizar la imagen entre [0, 1] (si ya está normalizada, este paso puede omitirse)
    image = np.clip(image, 0.0, 1.0)
    
    # Convertir a tensor y ajustar dimensiones para PyTorch [B, C, H, W]
    input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)

    # Pasar por el modelo
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Convertir de tensor a numpy y reordenar a [H, W, C]
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(output_image, 0.0, 1.0)  # Asegurar que los valores estén en [0, 1]

# Función para guardar la imagen en formato GeoTIFF
def save_geotiff(path, image, profile, scale_factor=2):
    # Ajustar el perfil para la nueva resolución
    profile.update({
        "height": image.shape[0],
        "width": image.shape[1],
        "transform": profile["transform"] * profile["transform"].scale(1 / scale_factor, 1 / scale_factor)
    })
    # Convertir a formato [Bands, H, W] para rasterio
    image = reshape_as_raster(image)
    
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(image)

# Ruta de entrada y salida
input_tiff = "E:/Entrenado_30_a_15/LR_0_0_001/20230118_LANDSAT_8_30m.tif"
output_tiff = "E:/Entrenado_30_a_15/HR_0_0_001/20230118_LANDSAT_8_15m.tif"

# Procesar la imagen
image, profile = read_geotiff(input_tiff)
super_resolved_image = super_resolve(image, model, scale_factor=2)
save_geotiff(output_tiff, super_resolved_image, profile)

print(f"Imagen superresuelta guardada en {output_tiff}")
