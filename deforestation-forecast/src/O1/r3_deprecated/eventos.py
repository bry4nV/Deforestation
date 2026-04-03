import os
import numpy as np
import rasterio


def calcular_perdida_anual(raster_t, raster_t1, output_path):
    """
    Genera raster de pérdida anual:
    1 = pérdida
    0 = no pérdida
    255 = nodata
    """

    with rasterio.open(raster_t) as src_t, rasterio.open(raster_t1) as src_t1:

        img_t = src_t.read(1)
        img_t1 = src_t1.read(1)

        meta = src_t.meta.copy()

        # Inicializar salida
        perdida = np.full(img_t.shape, 255, dtype="uint8")

        # máscara válida
        validoo = (img_t != 255) & (img_t1 != 255)

        # pérdida: bosque -> no bosque
        mascara_perdida = (img_t == 1) & (img_t1 == 0) & valido

        # no pérdida
        mascara_no_perdida = valido & (~mascara_perdida)

        perdida[mascara_perdida] = 1
        perdida[mascara_no_perdida] = 0

        meta.update(dtype="uint8", nodata=255)

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(perdida, 1)

    print(f"[OK] Pérdida generada: {output_path}")