import os
import numpy as np
import rasterio

from config import (
    DATA_DIR,
    MAPAS_RECLAS_DIR,
    ANIOS,

    CLASES_VALIDAS,
    CLASES_BOSQUE,
    CLASE_NOBSERVADO,
    NODATA_OUT
)

# =====================================================
# ETAPA 1: CARGA Y VERIFICACIÓN
# =====================================================

def etapa1_cargar_y_verificar(ruta_archivo):
    """
    Lee el raster original MapBiomas y extrae metadatos clave.
    """
    with rasterio.open(ruta_archivo) as src:
        img = src.read(1)
        meta = src.meta.copy()
        bounds = src.bounds
        nodata = src.nodata

    info = {
        "archivo": os.path.basename(ruta_archivo),
        "crs": meta["crs"],
        "res_x": meta["transform"][0],
        "res_y": meta["transform"][4],
        "width": meta["width"],
        "height": meta["height"],
        "bounds": bounds,
        "nodata_original": nodata,
        "dtype": meta["dtype"],
    }

    return img, meta, info, nodata


# =====================================================
# ETAPA 2: VALIDACIÓN Y DEPURACIÓN DE CLASES
# =====================================================

def etapa2_validar_y_depurar_clases(img):
    """
    Depura el raster, conservando solo clases válidas y asignando NoData (255)
    a valores no observados o no permitidos.
    """
    print("[E2] Clases antes:", np.unique(img))

    img_norm = img.copy()

    # A) No observado oficial (27)
    img_norm[img_norm == CLASE_NOBSERVADO] = NODATA_OUT

    # B) Todo lo que NO sea clase válida → NODATA_OUT
    mascara_invalidos = ~np.isin(img_norm, list(CLASES_VALIDAS) + [NODATA_OUT])
    img_norm[mascara_invalidos] = NODATA_OUT

    print("[E2] Clases después:", np.unique(img_norm))

    return img_norm


# =====================================================
# ETAPA 3: BOSQUE / NO BOSQUE
# =====================================================

def etapa3_reclasificar(img_norm):
    """
    Produce un raster binario:
        1 = bosque (clases CLASES_BOSQUE)
        0 = no bosque
      255 = NoData
    """

    # Máscara de bosque
    mascara_bosque = np.isin(img_norm, list(CLASES_BOSQUE))

    bosque_bin = np.full(img_norm.shape, NODATA_OUT, dtype="uint8")
    bosque_bin[mascara_bosque] = 1

    # No bosque (clase válida pero no está en CLASES_BOSQUE)
    mascara_no_bosque = (img_norm != NODATA_OUT) & (~mascara_bosque)
    bosque_bin[mascara_no_bosque] = 0

    valores, cuentas = np.unique(bosque_bin, return_counts=True)
    print(f"[E3] Distribución binaria final: {dict(zip(valores.tolist(), cuentas.tolist()))}")

    return bosque_bin

# =====================================================
# ETAPA 4: EXPORTACIÓN
# =====================================================

def etapa4_exportar(bosque_bin, meta, ruta_salida):
    """
    Guarda el raster binario como GeoTIFF con nodata=255
    """
    meta_out = meta.copy()
    meta_out.update(dtype="uint8", count=1, nodata=NODATA_OUT, compress="lzw")

    with rasterio.open(ruta_salida, "w", **meta_out) as dst:
        dst.write(bosque_bin, 1)


# =====================================================
# ORQUESTADOR
# =====================================================

def ejecutar_pipeline_anio(anio):

    nombre = f"mapbiomas-peru-collection-30-amazoniaperu-{anio}.tif"
    ruta = os.path.join(DATA_DIR, nombre)

    if not os.path.exists(ruta):
        print(f"[WARN] Archivo no encontrado: {ruta}")
        return None
    else:
        print(f"Archivo de análisis: {nombre}")

    # Etapa 1
    img, meta, info, nodata_original = etapa1_cargar_y_verificar(ruta)

    # Etapa 2
    img_norm = etapa2_normalizar(img)

    # Etapa 3
    bosque_bin = etapa3_reclasificar(img_norm)

    # Etapa 4
    salida = os.path.join(MAPAS_RECLAS_DIR, f"bosque_nobosque_amazonia_{anio}.tif")
    etapa4_exportar(bosque_bin, meta, salida)

    print(f"[OK] Año {anio} procesado → {salida}")

    return info