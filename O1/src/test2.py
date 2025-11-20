import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# 1. Leer raster
ruta_tif = r"D:\VisualCode\Tesis\O1\Dataset\peru_collection3_integration_v1-classification_2022.tif"
with rasterio.open(ruta_tif) as src:
    img = src.read(1)

# 2. Leer .clr (clase, R, G, B, A?, ?)
ruta_clr = r"D:\VisualCode\Tesis\O1\Dataset\PERU_2024_Nivel3_c3.clr"
data = np.loadtxt(ruta_clr, dtype=int)

classes = data[:, 0]       # códigos 0..72
rgb = data[:, 1:4]         # columnas R G B

# 3. Construir colormap discreto
max_class = classes.max()
colors = np.zeros((max_class + 1, 3))     # una fila por código de clase
colors[classes] = rgb / 255.0             # normalizar a 0–1

cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(max_class + 2) - 0.5, cmap.N)

# 4. Visualizar (puedes usar un recorte si es muy grande)
h, w = img.shape
recorte = img[h//4: 3*h//8, w//4: 3*w//8]  # zona central

plt.figure(figsize=(6, 8))
plt.imshow(recorte, cmap=cmap, norm=norm)
cbar = plt.colorbar()
cbar.set_ticks(classes)          # marcas en las clases
plt.title("Recorte MapBiomas con paleta oficial")
plt.show()
