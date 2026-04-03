import os
import rasterio
from config import MAPAS_RAW_DIR
import numpy as np

nombre = f"mapbiomas-peru-collection-30-amazoniaperu-1985.tif"
ruta = os.path.join(MAPAS_RAW_DIR, nombre)

"""
Lee el raster original MapBiomas y extrae metadatos clave.
"""
with rasterio.open(ruta) as src:
    img = src.read(1)
    meta = src.meta.copy()
    bounds = src.bounds
    nodata = src.nodata

# contar todos los valores tal cual
valores, conteos = np.unique(img, return_counts=True)

total = conteos.sum()

print("Total de píxeles:", total)
print()

for v, c in zip(valores, conteos):
    proporcion = c / total
    print(f"Valor {v}:")
    print(f"  píxeles = {c}")
    print(f"  proporción = {proporcion:.6f}")
    print(f"  porcentaje = {proporcion*100:.2f}%")