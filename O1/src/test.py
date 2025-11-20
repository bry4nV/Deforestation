import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

ruta = r"D:\VisualCode\Tesis\O1\Output\MapasReclasificados\bosque_nobosque_amazonia_2020.tif"

with rasterio.open(ruta) as src:
    img = src.read(1)

# Recorte central
h, w = img.shape
recorte = img[h//4 : 3*h//8, 0 : w//3]

# Colores por clase:
# 0 → gris (no bosque)
# 1 → verde (bosque)
# 255 → amarillo suave (no determinado)
colors = ['gray', 'green', '#ffff99']
values = [0, 1, 255]

cmap = ListedColormap(colors)
norm = BoundaryNorm([0, 1, 255, 256], cmap.N)

plt.figure(figsize=(7,7))
im = plt.imshow(recorte, cmap=cmap, norm=norm)
plt.title("Recorte central (0=No bosque, 1=Bosque, 255=No determinado)")

# Barra de color con etiquetas
cbar = plt.colorbar(im, ticks=values)
cbar.ax.set_yticklabels([
    "No bosque (0)",
    "Bosque (1)",
    "No determinado (255)"
])

plt.show()
