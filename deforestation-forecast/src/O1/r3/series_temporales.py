"""
Extracción de series temporales de deforestación por zona.

Este módulo extrae la pérdida anual de bosque para cada zona identificada,
generando un panel zona-año que será usado para el modelado predictivo.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import os


def extraer_series_temporales_por_zona(ruta_mapa_zonas, rutas_mapas_bosque, anios, ruta_salida_csv):
    """
    Extrae series temporales de pérdida de bosque para cada zona.
    
    Calcula la pérdida comparando mapas consecutivos de bosque/no bosque:
    Pérdida en año t = bosque(t-1) → no bosque(t)
    
    Parameters:
    -----------
    ruta_mapa_zonas : str
        Ruta al mapa de zonas (zonas_cambios_conectividad_8.tif)
    rutas_mapas_bosque : list
        Lista de rutas a mapas de bosque/no bosque (1985-2024)
    anios : list
        Lista de años correspondientes a los mapas
    ruta_salida_csv : str
        Ruta donde guardar el CSV con las series temporales
    
    Returns:
    --------
    pd.DataFrame : Panel zona-año con pérdida anual
    """
    
    print("\n" + "="*70)
    print("EXTRACCIÓN DE SERIES TEMPORALES POR ZONA")
    print("="*70 + "\n")
    
    # ========================================
    # PASO 1: Cargar mapa de zonas
    # ========================================
    
    print("[INFO] Cargando mapa de zonas...")
    with rasterio.open(ruta_mapa_zonas) as src:
        mapa_zonas = src.read(1)
        transform = src.transform
        nodata = src.nodata
    
    # Identificar zonas únicas (excluyendo 0 y nodata)
    zonas_unicas = np.unique(mapa_zonas)
    zonas_unicas = zonas_unicas[(zonas_unicas != 0) & (zonas_unicas != nodata)]
    n_zonas = len(zonas_unicas)
    
    print(f"[OK] Zonas identificadas: {n_zonas:,}")
    print(f"[INFO] Dimensiones del mapa: {mapa_zonas.shape[1]} x {mapa_zonas.shape[0]} píxeles\n")
    
    # ========================================
    # PASO 2: Calcular área de píxel
    # ========================================
    
    # Para EPSG:4326, aproximación simple
    res_deg_x = abs(transform.a)
    res_deg_y = abs(transform.e)
    
    # Latitud representativa (centro del raster)
    h, w = mapa_zonas.shape
    _, lat = transform * (w // 2, h // 2)
    
    # Conversión grados → metros
    m_per_deg_lat = 110574.0
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat))
    
    # Resolución en metros
    res_m_x = res_deg_x * m_per_deg_lon
    res_m_y = res_deg_y * m_per_deg_lat
    
    # Área
    area_pixel_m2 = res_m_x * res_m_y
    area_pixel_km2 = area_pixel_m2 / 1e6
    
    print(f"[INFO] Área por píxel: {area_pixel_km2:.6f} km²\n")
    
    # ========================================
    # PASO 3: Extraer series temporales
    # ========================================
    
    print("[INFO] Extrayendo series temporales...")
    print(f"[INFO] Procesando {len(anios)-1} años de pérdida para {n_zonas:,} zonas\n")
    
    # Inicializar lista para almacenar resultados
    registros = []
    
    # Procesar cada par de años consecutivos
    for idx in range(len(anios) - 1):
        anio_anterior = anios[idx]
        anio_actual = anios[idx + 1]
        
        print(f"  [{idx+1}/{len(anios)-1}] Procesando pérdida {anio_anterior} → {anio_actual}...", end=" ")
        
        # Cargar mapas de bosque
        with rasterio.open(rutas_mapas_bosque[idx]) as src:
            mapa_anterior = src.read(1)
        
        with rasterio.open(rutas_mapas_bosque[idx + 1]) as src:
            mapa_actual = src.read(1)
        
        # Verificar dimensiones
        if mapa_anterior.shape != mapa_zonas.shape or mapa_actual.shape != mapa_zonas.shape:
            print(f"\n[ERROR] Dimensiones no coinciden")
            print(f"  Zonas: {mapa_zonas.shape}")
            print(f"  Mapa {anio_anterior}: {mapa_anterior.shape}")
            print(f"  Mapa {anio_actual}: {mapa_actual.shape}")
            continue
        
        # Calcular pérdida: bosque (1) en t-1 → no bosque (0) en t
        mascara_valida = (mapa_anterior != NODATA) & (mapa_actual != NODATA)
        mapa_perdida = ((mapa_anterior == 1) & (mapa_actual == 0) & mascara_valida).astype(np.uint8)
        
        # Extraer pérdida por zona
        for zona_id in zonas_unicas:
            # Máscara de la zona
            mascara_zona = (mapa_zonas == zona_id)
            
            # Contar píxeles con pérdida en esta zona
            n_pixeles_perdida = np.sum(mapa_perdida[mascara_zona])
            
            # Calcular área de pérdida
            area_perdida_km2 = n_pixeles_perdida * area_pixel_km2
            
            # Guardar registro (pérdida se asigna al año actual)
            registros.append({
                'zona_id': int(zona_id),
                'anio': int(anio_actual),
                'pixeles_perdida': int(n_pixeles_perdida),
                'perdida_km2': float(area_perdida_km2)
            })
        
        print(f"✓ ({len(zonas_unicas):,} zonas procesadas)")
    
    # ========================================
    # PASO 4: Crear DataFrame y exportar
    # ========================================
    
    print(f"\n[INFO] Creando panel zona-año...")
    df_panel = pd.DataFrame(registros)
    
    # Ordenar por zona y año
    df_panel = df_panel.sort_values(['zona_id', 'anio']).reset_index(drop=True)
    
    # Estadísticas del panel
    print(f"\n" + "="*70)
    print("ESTADÍSTICAS DEL PANEL")
    print("="*70)
    print(f"  Total registros:      {len(df_panel):,}")
    print(f"  Zonas únicas:         {df_panel['zona_id'].nunique():,}")
    print(f"  Años únicos:          {df_panel['anio'].nunique()}")
    print(f"  Rango años:           {df_panel['anio'].min()} - {df_panel['anio'].max()}")
    print(f"\n  Pérdida total:        {df_panel['perdida_km2'].sum():,.2f} km²")
    print(f"  Pérdida promedio/año: {df_panel.groupby('anio')['perdida_km2'].sum().mean():,.2f} km²")
    print(f"  Pérdida promedio/zona:{df_panel.groupby('zona_id')['perdida_km2'].sum().mean():,.2f} km²")
    print("="*70 + "\n")
    
    # Exportar a CSV
    df_panel.to_csv(ruta_salida_csv, index=False, float_format='%.6f')
    print(f"[OK] Panel exportado: {ruta_salida_csv}")
    print(f"     {len(df_panel):,} registros × {len(df_panel.columns)} columnas\n")
    
    return df_panel


def generar_estadisticas_series_temporales(df_panel, ruta_salida_stats):
    """
    Genera estadísticas descriptivas de las series temporales.
    
    Parameters:
    -----------
    df_panel : pd.DataFrame
        Panel zona-año con series temporales
    ruta_salida_stats : str
        Ruta donde guardar estadísticas
    """
    
    print("[INFO] Generando estadísticas de series temporales...\n")
    
    # Estadísticas por zona
    stats_por_zona = df_panel.groupby('zona_id').agg({
        'perdida_km2': ['sum', 'mean', 'std', 'min', 'max'],
        'pixeles_perdida': ['sum', 'mean']
    }).reset_index()
    
    # Aplanar nombres de columnas
    stats_por_zona.columns = [
        'zona_id', 
        'perdida_total_km2', 
        'perdida_media_anual_km2',
        'perdida_std_km2',
        'perdida_min_km2',
        'perdida_max_km2',
        'pixeles_perdida_total',
        'pixeles_perdida_media_anual'
    ]
    
    # Ordenar por pérdida total descendente
    stats_por_zona = stats_por_zona.sort_values('perdida_total_km2', ascending=False)
    
    # Exportar
    stats_por_zona.to_csv(ruta_salida_stats, index=False, float_format='%.6f')
    
    print(f"[OK] Estadísticas por zona exportadas: {ruta_salida_stats}")
    print(f"     {len(stats_por_zona)} zonas × {len(stats_por_zona.columns)} columnas\n")
    
    # Mostrar top 10 zonas con mayor pérdida
    print("="*70)
    print("TOP 10 ZONAS CON MAYOR PÉRDIDA ACUMULADA")
    print("="*70)
    print(stats_por_zona[['zona_id', 'perdida_total_km2', 'perdida_media_anual_km2']].head(10).to_string(index=False))
    print("="*70 + "\n")
    
    return stats_por_zona


def visualizar_series_temporales_muestra(df_panel, ruta_salida_grafico, n_zonas=10):
    """
    Visualiza series temporales de una muestra de zonas.
    
    Parameters:
    -----------
    df_panel : pd.DataFrame
        Panel zona-año con series temporales
    ruta_salida_grafico : str
        Ruta donde guardar el gráfico
    n_zonas : int
        Número de zonas a visualizar
    """
    import matplotlib.pyplot as plt
    
    print(f"[INFO] Generando visualización de {n_zonas} zonas...\n")
    
    # Seleccionar zonas con mayor pérdida total
    zonas_top = df_panel.groupby('zona_id')['perdida_km2'].sum().nlargest(n_zonas).index
    
    # Filtrar datos
    df_muestra = df_panel[df_panel['zona_id'].isin(zonas_top)]
    
    # Crear gráfico
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: Series temporales individuales
    ax1 = axes[0]
    for zona_id in zonas_top:
        df_zona = df_muestra[df_muestra['zona_id'] == zona_id]
        ax1.plot(df_zona['anio'], df_zona['perdida_km2'], 
                marker='o', markersize=3, label=f'Zona {zona_id}', alpha=0.7)
    
    ax1.set_xlabel('Año', fontsize=11)
    ax1.set_ylabel('Pérdida Anual (km²)', fontsize=11)
    ax1.set_title(f'Series Temporales - Top {n_zonas} Zonas con Mayor Pérdida', 
                  fontsize=13, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Pérdida total por año (todas las zonas)
    ax2 = axes[1]
    perdida_anual = df_panel.groupby('anio')['perdida_km2'].sum()
    ax2.bar(perdida_anual.index, perdida_anual.values, color='darkred', alpha=0.7)
    ax2.set_xlabel('Año', fontsize=11)
    ax2.set_ylabel('Pérdida Total (km²)', fontsize=11)
    ax2.set_title('Pérdida Total Anual - Todas las Zonas', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir línea de tendencia
    z = np.polyfit(perdida_anual.index, perdida_anual.values, 1)
    p = np.poly1d(z)
    ax2.plot(perdida_anual.index, p(perdida_anual.index), 
            "r--", alpha=0.8, linewidth=2, label='Tendencia')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(ruta_salida_grafico, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Gráfico guardado: {ruta_salida_grafico}\n")


def validar_series_temporales(df_panel):
    """
    Valida la integridad de las series temporales.
    
    Parameters:
    -----------
    df_panel : pd.DataFrame
        Panel zona-año con series temporales
    """
    
    print("[INFO] Validando series temporales...\n")
    
    # Verificar completitud
    n_zonas = df_panel['zona_id'].nunique()
    n_anios = df_panel['anio'].nunique()
    registros_esperados = n_zonas * n_anios
    registros_reales = len(df_panel)
    
    print(f"  Zonas únicas:           {n_zonas:,}")
    print(f"  Años únicos:            {n_anios}")
    print(f"  Registros esperados:    {registros_esperados:,}")
    print(f"  Registros reales:       {registros_reales:,}")
    
    if registros_reales == registros_esperados:
        print(f"  ✓ Panel completo (100%)\n")
    else:
        print(f"  ⚠ Panel incompleto ({registros_reales/registros_esperados*100:.1f}%)\n")
    
    # Verificar valores negativos o nulos
    valores_negativos = (df_panel['perdida_km2'] < 0).sum()
    valores_nulos = df_panel['perdida_km2'].isnull().sum()
    
    print(f"  Valores negativos:      {valores_negativos}")
    print(f"  Valores nulos:          {valores_nulos}")
    
    if valores_negativos == 0 and valores_nulos == 0:
        print(f"  ✓ Sin valores inválidos\n")
    else:
        print(f"  ⚠ Valores inválidos detectados\n")
    
    # Verificar zonas sin pérdida
    zonas_sin_perdida = df_panel.groupby('zona_id')['perdida_km2'].sum()
    zonas_sin_perdida = (zonas_sin_perdida == 0).sum()
    
    print(f"  Zonas sin pérdida:      {zonas_sin_perdida} ({zonas_sin_perdida/n_zonas*100:.1f}%)")
    
    if zonas_sin_perdida > 0:
        print(f"  ⚠ Algunas zonas no tienen pérdida registrada\n")
    else:
        print(f"  ✓ Todas las zonas tienen pérdida registrada\n")
    
    print("="*70 + "\n")
