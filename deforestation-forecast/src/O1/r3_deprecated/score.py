import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter


def calcular_score(features, pesos):
    """
    Calcula score de potencial histórico de deforestación.
    
    S_i = w1·F_i + w2·R_i + w3·(1-P_i) + w4·E_i + w5·D_i
    
    Parameters:
    -----------
    features : dict
        Diccionario con features: freq, recencia, persist_bosque, exposicion, densidad_local
    pesos : dict
        Pesos para cada componente: w_frecuencia, w_recencia, w_persistencia, 
        w_exposicion, w_densidad
    
    Returns:
    --------
    np.ndarray : Score por píxel [0-1]
    """
    
    # Validar que tenemos todos los features
    required = ['freq', 'recencia', 'persist_bosque', 'exposicion', 'densidad_local']
    for key in required:
        if key not in features:
            raise ValueError(f"Feature faltante: {key}")
    
    # Normalizar cada feature al rango [0-1]
    freq_norm = normalizar_robusto(features['freq'])
    recencia_norm = features['recencia']  # ya está normalizado [0-1]
    
    # Invertir persistencia (alta persistencia = bajo score de cambio)
    persist_norm = normalizar_robusto(features['persist_bosque'])
    persist_inv = 1.0 - persist_norm
    
    exposicion_norm = normalizar_robusto(features['exposicion'])
    densidad_norm = normalizar_robusto(features['densidad_local'])
    
    # Calcular score ponderado
    score = (
        pesos['w_frecuencia'] * freq_norm +
        pesos['w_recencia'] * recencia_norm +
        pesos['w_persistencia'] * persist_inv +
        pesos['w_exposicion'] * exposicion_norm +
        pesos['w_densidad'] * densidad_norm
    )
    
    # Normalizar score final a [0-1]
    score = np.clip(score, 0, 1)
    
    # Mantener NaN donde había datos inválidos
    mask_valido = ~np.isnan(features['freq'])
    score[~mask_valido] = np.nan
    
    return score


def normalizar_robusto(arr, percentil_min=5, percentil_max=95):
    """
    Normaliza array usando percentiles robustos para evitar outliers.
    """
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return arr
    
    vmin = np.percentile(valid, percentil_min)
    vmax = np.percentile(valid, percentil_max)
    
    arr_norm = (arr - vmin) / (vmax - vmin + 1e-10)
    arr_norm = np.clip(arr_norm, 0, 1)
    
    return arr_norm


def suavizar_score(score, sigma=2.0):
    """
    Suaviza el score espacialmente usando filtro Gaussiano.
    
    Parameters:
    -----------
    score : np.ndarray
        Score original
    sigma : float
        Desviación estándar del kernel Gaussiano (default=2.0)
    
    Returns:
    --------
    np.ndarray : Score suavizado
    """
    # Manejar NaN: rellenar temporalmente con 0
    score_filled = np.nan_to_num(score, nan=0.0)
    
    # Crear máscara de valores válidos
    mask_valido = ~np.isnan(score)
    
    # Suavizar score y máscara por separado
    score_smooth = gaussian_filter(score_filled, sigma=sigma, mode='constant', cval=0.0)
    mask_smooth = gaussian_filter(mask_valido.astype(float), sigma=sigma, mode='constant', cval=0.0)
    
    # Normalizar para compensar el suavizado de la máscara
    score_smooth = np.divide(score_smooth, mask_smooth, 
                             out=np.zeros_like(score_smooth), 
                             where=mask_smooth > 0.1)
    
    # Restaurar NaN en píxeles originalmente inválidos
    score_smooth[~mask_valido] = np.nan
    
    return score_smooth


def guardar_score(score, transform, crs, output_path):
    """
    Guarda el score como raster GeoTIFF.
    """
    meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': np.nan,
        'width': score.shape[1],
        'height': score.shape[0],
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(score.astype('float32'), 1)
    
    print(f"[OK] Score guardado: {output_path}")


def estadisticas_score(score):
    """
    Calcula estadísticas descriptivas del score.
    """
    valid = score[~np.isnan(score)]
    
    if len(valid) == 0:
        print("[WARN] Score sin valores válidos")
        return
    
    print("\n" + "="*50)
    print("ESTADÍSTICAS DEL SCORE")
    print("="*50)
    print(f"  Píxeles válidos: {len(valid):,}")
    print(f"  Media:           {np.mean(valid):.4f}")
    print(f"  Mediana:         {np.median(valid):.4f}")
    print(f"  Desv. estándar:  {np.std(valid):.4f}")
    print(f"  Mínimo:          {np.min(valid):.4f}")
    print(f"  Máximo:          {np.max(valid):.4f}")
    print(f"\n  Percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        val = np.percentile(valid, p)
        print(f"    P{p:2d}: {val:.4f}")
    print("="*50 + "\n")
