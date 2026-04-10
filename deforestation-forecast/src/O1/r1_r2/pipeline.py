import os
import numpy as np
import rasterio

from O1.config import (
    MAPAS_RECLAS_DIR,
    MAPAS_RAW_DIR,

    CLASES_VALIDAS,
    CLASES_BOSQUE,
    CLASE_NOBSERVADO,
    NODATA
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
        "transform": meta["transform"],
        "res_x": meta["transform"][0],
        "res_y": meta["transform"][4],
        "width": meta["width"],
        "height": meta["height"],
        "count": meta["count"],
        "bounds": bounds,
        "nodata_original": nodata,
        "dtype": meta["dtype"],
        "clases_unicas": np.unique(img).tolist(),
        "file_size_mb": round(os.path.getsize(ruta_archivo) / (1024 * 1024), 2)
    }

    return img, meta, info


# =====================================================
# ETAPA 2: VALIDACIÓN Y DEPURACIÓN DE CLASES
# =====================================================

def etapa2_validar_y_depurar_clases(img):
    """
    Depura el raster, conservando solo clases válidas y asignando NoData (255)
    a valores no observados o no permitidos.
    """
    print("[E2] Clases antes:", np.unique(img))

    img_dep = img.copy()

    # No observado oficial
    img_dep[img_dep == CLASE_NOBSERVADO] = NODATA

    # Todo lo que NO sea clase válida
    mascara_invalidos = ~np.isin(img_dep, list(CLASES_VALIDAS) + [NODATA])
    img_dep[mascara_invalidos] = NODATA

    print("[E2] Clases después:", np.unique(img_dep))

    return img_dep


# =====================================================
# ETAPA 3: BOSQUE / NO BOSQUE
# =====================================================

def etapa3_reclasificar(img_dep):
    """
    Produce un raster binario:
        1 = bosque (clases CLASES_BOSQUE)
        0 = no bosque
      255 = NoData
    """

    # Máscara de bosque
    mascara_bosque = np.isin(img_dep, list(CLASES_BOSQUE))

    bosque_bin = np.full(img_dep.shape, NODATA, dtype="uint8")
    bosque_bin[mascara_bosque] = 1

    # No bosque
    mascara_no_bosque = (img_dep != NODATA) & (~mascara_bosque)
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
    meta_out.update(dtype="uint8", count=1, nodata=NODATA, compress="lzw")

    with rasterio.open(ruta_salida, "w", **meta_out) as dst:
        dst.write(bosque_bin, 1)
        valores, cuentas = np.unique(bosque_bin, return_counts=True)

    dist = dict(zip(valores.tolist(), cuentas.tolist()))
    total = bosque_bin.size

    bosque_pix = dist.get(1, 0)
    nobosque_pix = dist.get(0, 0)
    nodata_pix = dist.get(NODATA, 0)

    pixel_area_km2 = 0.0009

    info = {
        "archivo_salida": os.path.basename(ruta_salida),
        "dtype_salida": meta_out["dtype"],
        "nodata_salida": meta_out["nodata"],
        "clases_unicas_salida": sorted(np.unique(bosque_bin).tolist()),
        "total_pixeles": int(total),
        "bosque_pix": int(bosque_pix),
        "nobosque_pix": int(nobosque_pix),
        "nodata_pix": int(nodata_pix),
        "bosque_pct": round(bosque_pix / total * 100, 6),
        "nobosque_pct": round(nobosque_pix / total * 100, 6),
        "nodata_pct": round(nodata_pix / total * 100, 6),
        "bosque_area_km2": round(bosque_pix * pixel_area_km2, 4),
        "nobosque_area_km2": round(nobosque_pix * pixel_area_km2, 4),
        "nodata_area_km2": round(nodata_pix * pixel_area_km2, 4),
    }

    return info

# =====================================================
# ORQUESTADOR
# =====================================================

def ejecutar_pipeline_anio(anio):

    nombre = f"mapbiomas-peru-collection-30-amazoniaperu-{anio}.tif"
    ruta = os.path.join(MAPAS_RAW_DIR, nombre)

    if not os.path.exists(ruta):
        print(f"[WARN] Archivo no encontrado: {ruta}")
        return None
    else:
        print(f"Archivo de análisis: {nombre}")

    # Etapa 1
    img, meta, info_raw = etapa1_cargar_y_verificar(ruta)

    # Etapa 2
    img_dep = etapa2_validar_y_depurar_clases(img)

    # Etapa 3
    bosque_bin = etapa3_reclasificar(img_dep)

    # Etapa 4
    salida = os.path.join(MAPAS_RECLAS_DIR, f"bosque_nobosque_amazonia_{anio}.tif")
    info_reclasificado = etapa4_exportar(bosque_bin, meta, salida)


    info = {
        "anio": anio,
        "raw": info_raw,
        "reclasificado": info_reclasificado
    }

    print(f"[OK] Año {anio} procesado → {salida}")

    return info