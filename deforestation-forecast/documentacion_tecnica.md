# Documentación Técnica — Pronóstico de Deforestación en Distritos Amazónicos del Perú

**Descripción:** Pipeline completo para el pronóstico de cobertura boscosa en los 200 distritos amazónicos del Perú con mayor cambio histórico, abarcando desde la ingesta de rasters MapBiomas hasta la comparación de modelos de series temporales.

---

## Tabla de contenidos

1. [Visión general](#1-visión-general)
2. [Estructura del proyecto](#2-estructura-del-proyecto)
3. [O1 — Preprocesamiento y análisis de datos](#3-o1--preprocesamiento-y-análisis-de-datos)
   - [3.1 Configuración global (O1/config.py)](#31-configuración-global-o1configpy)
   - [3.2 R1/R2 — Delimitación a la Amazonía y reclasificación](#32-r1r2--delimitación-a-la-amazonía-y-reclasificación)
   - [3.3 R3 — Detección de cambios, zonificación y series temporales](#33-r3--detección-de-cambios-zonificación-y-series-temporales)
4. [O2 — Modelado de pronóstico](#4-o2--modelado-de-pronóstico)
   - [4.1 Configuración global (O2/config.py)](#41-configuración-global-o2configpy)
   - [4.2 Construcción de datasets (construir_dataset.py)](#42-construcción-de-datasets-construir_datasetpy)
   - [4.3 Análisis diagnóstico ARIMA (analisis_arima.py)](#43-análisis-diagnóstico-arima-analisis_arimapy)
   - [4.4 Modelo Persistencia — baseline (pipeline_persistencia.py)](#44-modelo-persistencia--baseline-pipeline_persistenciapy)
   - [4.5 Modelo ARIMA (pipeline_arima.py)](#45-modelo-arima-pipeline_arimapy)
   - [4.6 Modelo MLP (pipeline_mlp.py)](#46-modelo-mlp-pipeline_mlppy)
   - [4.7 Modelo LSTM (pipeline_lstm.py)](#47-modelo-lstm-pipeline_lstmpy)
   - [4.8 Comparación final (pipeline_comparacion.py)](#48-comparación-final-pipeline_comparacionpy)
   - [4.9 Orquestador principal (main.py)](#49-orquestador-principal-mainpy)
5. [Flujo de datos extremo a extremo](#5-flujo-de-datos-extremo-a-extremo)
6. [Convenciones y reproducibilidad](#6-convenciones-y-reproducibilidad)
7. [Inventario de archivos de salida](#7-inventario-de-archivos-de-salida)

---

## 1. Visión general

El proyecto está dividido en dos objetivos principales:

| Objetivo | Carpeta | Descripción |
|----------|---------|-------------|
| **O1** | `src/O1/` | Preprocesamiento geoespacial: recorte a Amazonía, reclasificación bosque/no-bosque, detección de cambios, selección de distritos y extracción de series temporales. |
| **O2** | `src/O2/` | Modelado: entrenamiento y evaluación de cuatro modelos de pronóstico (Persistencia, ARIMA, MLP, LSTM) y comparación final. |

**Datos de entrada:** Colección 3 de MapBiomas Perú (40 mapas anuales 1985–2024, resolución 30 m).  
**Variable objetivo:** Porcentaje de cobertura boscosa por distrito (`pct_bosque`).  
**Período de pronóstico:** 5 años (2020–2024).  
**Universo de distritos:** 200 distritos amazónicos con mayor cambio histórico de cobertura boscosa.

---

## 2. Estructura del proyecto

```
deforestation-forecast/
├── data/
│   ├── raw/                          # Datos originales (solo lectura)
│   │   ├── mapbiomas-peru/           # 40 rasters MapBiomas (YYYY)
│   │   ├── biomas-peru/              # Shapefile biomas del Perú
│   │   └── distritos-peru/           # Shapefile distritos políticos
│   ├── interim/
│   │   ├── O1/                       # Salidas intermedias O1
│   │   │   ├── mapas-amazonia/
│   │   │   ├── mapas-reclasificados/
│   │   │   ├── mapas-cambios/
│   │   │   ├── distritos-amazonia/
│   │   │   ├── metricas-distritos/
│   │   │   ├── distritos-alto-cambio/
│   │   │   └── series-temporales/
│   │   │       ├── entrenamiento/
│   │   │       └── generalizacion-espacial/
│   │   └── O2/
│   │       └── modelos/
│   │           ├── persistencia/
│   │           ├── arima/
│   │           ├── mlp/
│   │           ├── lstm/
│   │           └── comparacion/
│   └── outputs/                      # Productos finales
└── src/
    ├── O1/
    │   ├── config.py
    │   ├── r1_r2/
    │   │   ├── main.py
    │   │   ├── pipeline.py
    │   │   └── delimitacion_mapa_amazonas.py
    │   └── r3/
    │       ├── main.py
    │       ├── deteccion_cambios.py
    │       ├── delimitacion_distritos_amazonas.py
    │       ├── zonificacion_distrito.py
    │       ├── distritos_alto_cambio.py
    │       └── series_temporales.py
    └── O2/
        ├── config.py
        └── r4_r5/
            ├── main.py
            ├── construir_dataset.py
            ├── analisis_arima.py
            ├── pipeline_persistencia.py
            ├── pipeline_arima.py
            ├── pipeline_mlp.py
            ├── pipeline_lstm.py
            └── pipeline_comparacion.py
```

---

## 3. O1 — Preprocesamiento y análisis de datos

### 3.1 Configuración global (`O1/config.py`)

Define todas las rutas y constantes compartidas por los módulos O1.

#### Rutas principales

| Variable | Descripción |
|----------|-------------|
| `BASE_DIR` | Raíz del proyecto |
| `DATA_DIR`, `RAW_DIR` | Datos originales |
| `INTERIM_DIR` | Datos intermedios procesados |
| `MAPAS_RAW_DIR` | Rasters MapBiomas originales |
| `MAPAS_AMAZONIA_DIR` | Rasters recortados a Amazonía |
| `MAPAS_RECLAS_DIR` | Rasters binarios bosque/no-bosque |
| `DISTRITOS_AMAZONIA_DIR` | GeoPackage distritos amazónicos |
| `MAPAS_CAMBIOS_DIR` | Raster de cambios históricos |
| `METRICAS_DISTRITOS_DIR` | Métricas de cambio por distrito |
| `DISTRITOS_ALTO_CAMBIO_DIR` | Top 200 distritos seleccionados |
| `SERIES_ENTRENAMIENTO_DIR` | Series temporales para O2 |
| `SERIES_GENERALIZACION_DIR` | Series para validación espacial |

#### Clasificación MapBiomas (Colección 3 Perú)

```python
CLASES_BOSQUE   = {3, 4, 5, 6}          # Formaciones forestales
CLASES_VALIDAS  = {3,4,5,6,11,12,29,66,70,
                   13,15,18,35,40,72,9,21,
                   23,24,30,32,61,68,25,33,31,34}
CLASE_NOBSERVADO = 27                    # No observado → NODATA
NODATA           = 255                   # Marcador ausencia de dato
```

La reclasificación es binaria: píxeles en `CLASES_BOSQUE` → 1, el resto → 0, no observado → 255.

#### Proyecciones

| Constante | Valor | Uso |
|-----------|-------|-----|
| `CRS_PROYECTADO` | `EPSG:32718` | Cálculos de área (UTM zona 18S) |
| `CRS_GEOG` | `EPSG:4326` | Almacenamiento geográfico (WGS84) |

#### Parámetros de series temporales

```python
ANIOS              = list(range(1985, 2025))  # 40 años
TAMANIO_ENTRENAMIENTO = 0.9                   # 90 % de distritos para entrenamiento
SEMILLA_SPLIT      = 42                       # Reproducibilidad del split 90/10
```

---

### 3.2 R1/R2 — Delimitación a la Amazonía y reclasificación

**Punto de entrada:** `O1/r1_r2/main.py`

#### Flujo de ejecución

```
main.py
  ├── Si no existen distritos amazónicos:
  │     pipeline_delimitacion_amazonia()
  │       ├── identificar_distritos_amazonia_interseccion()
  │       └── recortar_mapas_amazonia()
  └── Para cada año en 1985–2024:
        ejecutar_pipeline_anio(anio)
          ├── etapa1_cargar_y_verificar()
          ├── etapa2_validar_y_depurar_clases()
          ├── etapa3_reclasificar()
          └── etapa4_exportar()
  └── Guarda metadatos acumulados en dos CSVs
```

---

#### `delimitacion_mapa_amazonas.py`

**`identificar_distritos_amazonia_interseccion(ruta_biomas_peru, ruta_distritos_peru, ruta_distritos_amazonia_delimitados)`**

Selecciona los distritos del Perú que tienen más del 50 % de su área dentro del bioma Amazonía.

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `ruta_biomas_peru` | `str` | Ruta al shapefile de biomas nacionales |
| `ruta_distritos_peru` | `str` | Ruta al shapefile de distritos |
| `ruta_distritos_amazonia_delimitados` | `str` | Ruta de salida (GeoPackage) |

Proceso:
1. Filtra el bioma `"[Amazonía]"`.
2. Reproyecta a `EPSG:32718` para medir áreas en m².
3. Calcula intersección distrito–bioma y retiene los distritos con `área_intersección / área_distrito ≥ 0.50`.
4. Convierte el resultado a `EPSG:4326` y guarda el GeoPackage.

Salida:
- `distritos_amazonia.gpkg` — geometrías + columnas de área.
- CSV auxiliar con porcentajes de intersección por geocode.

---

**`recortar_mapas_amazonia(gdf_amazonia, carpeta_mapas_raw, carpeta_mapas_amazonia)`**

Para cada uno de los 40 años recorta el raster MapBiomas original a la envolvente geométrica unificada de los distritos amazónicos.

Salida: `peru_amazonia_YYYY.tif` (40 rasters, mismo CRS que el original).

---

#### `pipeline.py`

Procesa un raster anual en cuatro etapas encadenadas:

**`etapa1_cargar_y_verificar(ruta_archivo)`** — Lectura y metadatos

Carga el raster con `rasterio` y extrae: CRS, resolución, dimensiones, clases únicas y tamaño en disco.

```python
retorna: (img: ndarray, meta: dict, info: dict)
```

---

**`etapa2_validar_y_depurar_clases(img)`** — Limpieza de clases

- Convierte píxeles `CLASE_NOBSERVADO` (27) a `NODATA` (255).
- Convierte cualquier clase fuera de `CLASES_VALIDAS` a `NODATA`.

```python
retorna: img_dep (ndarray) — igual forma, solo clases válidas o 255
```

---

**`etapa3_reclasificar(img_dep)`** — Binarización bosque/no-bosque

```python
bosque_bin[pixel] = 1    si pixel ∈ CLASES_BOSQUE
bosque_bin[pixel] = 0    si pixel válido y ∉ CLASES_BOSQUE
bosque_bin[pixel] = 255  si pixel == NODATA
```

```python
retorna: bosque_bin (ndarray, dtype=uint8)
```

---

**`etapa4_exportar(bosque_bin, meta, ruta_salida)`** — Exportación y estadísticas

Escribe el GeoTIFF comprimido (`LZW`) y calcula:

| Estadística | Descripción |
|-------------|-------------|
| `bosque_pix` | Píxeles con valor 1 |
| `nobosque_pix` | Píxeles con valor 0 |
| `nodata_pix` | Píxeles con valor 255 |
| `bosque_pct` | Porcentaje bosque sobre píxeles válidos |
| `bosque_area_km2` | Área boscosa estimada (km²) |

```python
retorna: info_reclasificado (dict) — estadísticas completas
```

---

**Salidas generadas por R1/R2**

| Archivo | Descripción |
|---------|-------------|
| `mapas-amazonia/peru_amazonia_YYYY.tif` | 40 rasters recortados a Amazonía |
| `mapas-reclasificados/bosque_nobosque_amazonia_YYYY.tif` | 40 rasters binarios (0/1/255) |
| `mapas-amazonia/metadatos_mapas_amazonia.csv` | Metadatos de los rasters originales |
| `mapas-reclasificados/metadatos_mapas_reclasificados_amazonia.csv` | Estadísticas de los rasters binarios |
| `distritos-amazonia/distritos_amazonia.gpkg` | Límites de los distritos amazónicos |

---

### 3.3 R3 — Detección de cambios, zonificación y series temporales

**Punto de entrada:** `O1/r3/main.py`

#### Flujo de ejecución

```
main.py
  ├── Paso 1 — Detección de cambios
  │     detectar_cambios_por_tiles()
  │     guardar_mapa_cambios()
  │     exportar_estadisticas_cambios()
  │
  ├── Paso 2 — Zonificación por distrito
  │     pipeline_zonificacion_distrito()
  │       ├── calcular_metricas_por_distrito()  (rasterstats)
  │       ├── guardar_mapa_cambios_distrito()
  │       ├── exportar_csv_distritos()
  │       └── exportar_csv_resumen()
  │
  ├── Paso 3 — Selección distritos alto cambio
  │     pipeline_seleccion_distritos_alto_cambio()
  │
  └── Paso 4 — Series temporales
        pipeline_extraer_series_temporales()
          ├── split_aleatorio()  →  180 entrenamiento / 20 generalización
          ├── extraer_series(gdf_train)
          └── extraer_series(gdf_test)
```

---

#### `deteccion_cambios.py`

**`detectar_cambios_por_tiles(rutas_mapas_reclasificados, tamanio_tile=5000)`**

Procesa los 40 rasters binarios en *tiles* de 5 000 × 5 000 píxeles para evitar problemas de memoria.

Para cada tile:
1. Carga el stack temporal (shape: 40 × alto_tile × ancho_tile).
2. Llama a `detectar_cambios_tile()`.
3. Escribe el resultado en el mapa global de cambios.

**`detectar_cambios_tile(conjunto_tiles)`**

```python
# Para cada par de años consecutivos detecta cambio de categoría
cambio[y, x] = 1  si ∃ t: tile[t, y, x] ≠ tile[t+1, y, x]  y ninguno es NODATA
cambio[y, x] = 0  si la serie es constante y sin NODATA
cambio[y, x] = 255  si algún año tiene NODATA
```

```
entrada:  conjunto_tiles (ndarray, shape: n_años × alto × ancho)
salida:   mapa_cambios (ndarray, shape: alto × ancho, dtype=uint8)
```

**`exportar_estadisticas_cambios(ruta_mapa_cambios, ruta_estadisticas_cambios)`**

Genera CSV con recuentos y porcentajes de píxeles con/sin cambio sobre el mapa completo.

---

#### `zonificacion_distrito.py`

**`calcular_metricas_por_distrito(ruta_mapa_cambios, gdf_distritos)`**

Usa `rasterstats.zonal_stats()` en modo categórico para contar píxeles `0` y `1` dentro de cada polígono distrital. Calcula:

```
porcentaje_cambio = pixeles_cambiados / pixeles_validos × 100
```

**`pipeline_zonificacion_distrito(...)`**

Orquesta: cálculo de métricas → merge con geometrías → exportación GeoPackage + dos CSVs (detalle por distrito y resumen global).

---

#### `distritos_alto_cambio.py`

**`pipeline_seleccion_distritos_alto_cambio(ruta_mapa_cambios_distrito, ruta_distritos_alto_cambio)`**

Ordena los distritos por `porcentaje_cambio` descendente y retiene los **200 primeros**. Exporta GeoPackage y CSV.

---

#### `series_temporales.py`

**`extraer_series(gdf, rutas_mapas_reclasificados)`**

Para cada año (40 iteraciones) extrae con `zonal_stats()` el recuento de píxeles bosque (1) y no-bosque (0) por distrito y calcula `pct_bosque`.

```
salida:  DataFrame — (n_distritos × 40) filas
         columnas: geocode, departamento, distrito, anio,
                   pix_total, pix_bosque, pix_no_bosque,
                   pct_bosque, pct_no_bosque
```

**`split_aleatorio(gdf)`**

Divide los 200 distritos en 90 % entrenamiento (180 distritos) y 10 % generalización espacial (20 distritos) con semilla fija `SEMILLA_SPLIT = 42`.

**`pipeline_extraer_series_temporales(...)`**

Ejecuta `split_aleatorio` y luego `extraer_series` para cada subconjunto, guardando dos CSVs independientes.

---

**Salidas generadas por R3**

| Archivo | Descripción |
|---------|-------------|
| `mapas-cambios/mapa_cambios_1985_2024.tif` | Raster binario de cambio histórico (0/1/255) |
| `mapas-cambios/estadisticas_cambios.csv` | Resumen global de cambios |
| `metricas-distritos/mapa_cambios_distrito_1985_2024.gpkg` | Geometrías + métricas por distrito |
| `metricas-distritos/metricas_distritos.csv` | Tabla de métricas por distrito |
| `distritos-alto-cambio/distritos_alto_cambio.gpkg` | Top 200 distritos seleccionados |
| `distritos-alto-cambio/distritos_alto_cambio.csv` | Versión tabular sin geometría |
| `series-temporales/entrenamiento/distritos_entrenamiento.csv` | 180 distritos × 40 años |
| `series-temporales/generalizacion-espacial/distritos_generalizacion_espacial.csv` | 20 distritos × 40 años |

---

## 4. O2 — Modelado de pronóstico

### 4.1 Configuración global (`O2/config.py`)

#### Rutas de salida

| Variable | Ruta relativa bajo `interim/O2/modelos/` |
|----------|------------------------------------------|
| `PERSISTENCIA_DIR` | `persistencia/` |
| `ARIMA_DIR` | `arima/` |
| `ANALISIS_ARIMA_DIR` | `arima/analisis_arima/` |
| `MLP_DIR` | `mlp/` |
| `LSTM_DIR` | `lstm/` |
| `COMPARACION_DIR` | `comparacion/` |

Todos los directorios se crean automáticamente al importar el módulo.

#### Parámetros temporales

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `TAMANIO_ENTRENAMIENTO` | `35` | Años de histórico usados para ajustar modelos (1985–2019) |
| `HORIZONTE` | `5` | Años de pronóstico (2020–2024) |
| `ANIO_INICIO` | `1985` | Año base para etiquetas en gráficos |
| `SEMILLA` | `42` | Semilla global de aleatoriedad |

#### Hiperparámetros ARIMA

```python
ARIMA_P_VALUES      = [0, 1, 2]
ARIMA_D_VALUES      = [1]
ARIMA_Q_VALUES      = [0, 1, 2]
ARIMA_WINDOW_VALUES = [3, 4, 5, 10, 15, 20, 25, 30, 35]
```

Combinaciones totales: 3 × 1 × 3 × 9 = **81 configuraciones**, evaluadas con walk-forward para los 180 distritos.

#### Hiperparámetros MLP

```python
MLP_HIDDEN_SIZES_VALUES = [[32, 16], [64, 32], [128, 64, 32]]
MLP_DROPOUT_VALUES      = [0.0, 0.1]
MLP_EPOCHS_VALUES       = [50]
MLP_LR_VALUES           = [0.001, 0.0005]
MLP_BATCH_SIZE_VALUES   = [8, 16]
DL_WINDOW_VALUES        = [3, 4, 5, 10]
```

Combinaciones totales (por ventana): 3 × 2 × 1 × 2 × 2 = 24; con 4 ventanas: **96 configuraciones**.

#### Hiperparámetros LSTM

```python
LSTM_HIDDEN_SIZE_VALUES = [32, 64, 128]
LSTM_NUM_LAYERS_VALUES  = [1]
LSTM_DROPOUT_VALUES     = [0.0, 0.1]
LSTM_EPOCHS_VALUES      = [50]
LSTM_LR_VALUES          = [0.001, 0.0005]
LSTM_BATCH_SIZE_VALUES  = [8, 16]
```

Combinaciones totales: 3 × 1 × 2 × 1 × 2 × 2 × 4 ventanas = **96 configuraciones**.

---

### 4.2 Construcción de datasets (`construir_dataset.py`)

Este módulo transforma el CSV de series temporales en las estructuras de datos que consumen los modelos.

---

**`cargar_series(ruta_series)`**

Lee el CSV de entrenamiento y pivotea a formato matricial.

```
entrada:  CSV con columnas {geocode, departamento, distrito, anio, pct_bosque}
salida:
  series           ndarray (180, 40)  — pct_bosque[distrito, año]
  df_distritos_info  DataFrame (180, 3)  — geocode, departamento, distrito
```

El índice geográfico se preserva: `series[i, :]` corresponde a `df_distritos_info.iloc[i]`.

---

**`construir_dataset_estadistico(series, train_size=35, horizon=5)`**

Divide la matriz de series en período de ajuste y período de evaluación para los modelos estadísticos.

```
entrada:  series (180, 40)
salida:
  X_train_stat  ndarray (180, 35, 1)  — histórico 1985–2019
  y_train_stat  ndarray (180, 5)      — horizonte 2020–2024
```

La dimensión extra al final de `X_train_stat` es requerida por las funciones ARIMA y Persistencia.

---

**`crear_ventanas_split(series, window_size, tamanio_entrenamiento)`**

Genera ventanas deslizantes para modelos de aprendizaje profundo.

```
Para cada distrito i:
  Para cada t en [0, n_anios - window_size):
    X_window = series[i, t : t + window_size]
    y_target = series[i, t + window_size]
    Si t + window_size < 35  →  train
    Si t + window_size >= 35 →  test
```

```
salida (con window_size=4, 180 distritos):
  X_train  (≈6 120, 4, 1)    muestras de entrenamiento
  y_train  (≈6 120, 1)
  X_test   (≈900, 4, 1)      muestras de test
  y_test   (≈900, 1)
```

**Nota:** Con `window_size ≥ tamanio_entrenamiento` no se generan muestras de entrenamiento y la ventana se omite.

---

**`construir_dataset_dl(series, window_sizes, tamanio_entrenamiento)`**

Aplica `crear_ventanas_split` para cada ventana en `DL_WINDOW_VALUES` y convierte los arrays a `torch.float32`.

```
salida:  dict[window_size → {
           "train": (X_tensor, y_tensor),
           "test":  (X_tensor, y_tensor)
         }]
```

---

### 4.3 Análisis diagnóstico ARIMA (`analisis_arima.py`)

Módulo de exploración previa al ajuste ARIMA. Genera gráficos para orientar la selección de los órdenes `(p, d, q)`.

**`generar_analisis_arima(ruta_estadisticas, ruta_series, ruta_analisis_arima)`**

Flujo:
1. Selecciona tres series representativas: distrito de mayor variabilidad, de menor variabilidad y mediana global.
2. Traza las series originales.
3. Aplica diferenciación de orden 1 y traza las series diferenciadas.
4. Genera gráficos ACF y PACF (hasta 10 lags) para cada serie.

Salida: 11 imágenes PNG en `ANALISIS_ARIMA_DIR/`.

---

### 4.4 Modelo Persistencia — baseline (`pipeline_persistencia.py`)

**`pipeline_persistencia(X_train, y_train, df_distritos_info, ruta_modelo_persistencia)`**

Implementa el modelo de persistencia con evaluación walk-forward: en cada paso del horizonte utiliza el último valor observado como pronóstico y luego incorpora el valor real para el siguiente paso.

```
Para cada distrito i:
    history = X_train[i, :, 0]          # 35 valores reales
    Para t en {0, 1, 2, 3, 4}:
        ŷ[t] = history[-1]              # Último valor conocido
        history.append(y_train[i, t])   # Actualiza con el real
```

**Cálculo de métricas por departamento**

Las métricas se calculan sobre el conjunto **pooled** de residuos de todos los distritos del departamento, garantizando comparabilidad con ARIMA:

```python
y_t = y_train[mask]        # (n_dist_dep, 5)
y_p = y_pred_total[mask]   # (n_dist_dep, 5)
RMSE_dep = sqrt(mean((y_t - y_p)²))   # sobre todos los elementos
MAE_dep  = mean(|y_t - y_p|)
```

**Salidas**

| Archivo | Contenido |
|---------|-----------|
| `persistencia_resultados.csv` | RMSE y MAE por distrito |
| `persistencia_resultados_departamento.csv` | RMSE y MAE pooled por departamento |
| `persistencia_resultados_global.csv` | Métrica escalar global |
| `persistencia_resultados_ypred.npy` | Array (180, 5) de predicciones |

**Return**

```python
{"modelo": "Persistencia_WF", "rmse": float, "mae": float, "y_pred": ndarray(180,5)}
```

---

### 4.5 Modelo ARIMA (`pipeline_arima.py`)

#### `metricas_por_departamento(df_distritos_info, y_true, y_pred)`

Función auxiliar de referencia. Calcula el RMSE y MAE pooled por departamento — la misma lógica se replica en MLP y LSTM.

```python
Para dep en departamentos únicos:
    mask = departamento == dep
    RMSE_dep = sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    MAE_dep  = mean_absolute_error(y_true[mask], y_pred[mask])
```

---

#### `evaluar_arima(X_train, y_train, df_distritos_info, window_size, order, exportar, ruta)`

Evaluación walk-forward de una configuración `(window_size, p, d, q)`.

```
Para cada distrito i:
    history = X_train[i, :, 0]           # Histórico completo
    Para t en {0, 1, 2, 3, 4}:
        ventana = history[-window_size:]
        Ajusta ARIMA(p,d,q) sobre ventana
        ŷ[t] = model.forecast()[0]
        history.append(y_train[i, t])    # Incorpora real
        Si falla ARIMA → ŷ[t] = ventana[-1]  (fallback persistencia)
```

Con `exportar=True` escribe tres CSVs (distritos, departamentos, global) en `ruta`.

**Return**

```python
{
  "modelo":         "ARIMA_WF_w{window_size}",
  "rmse":           float,       # Global sobre 180×5 residuos
  "mae":            float,
  "y_pred":         ndarray(180,5),
  "df_metricas":    DataFrame,   # Por distrito
  "df_dep":         DataFrame,   # Por departamento (pooled)
  "rmse_distritos": ndarray(180) # Para boxplot
}
```

---

#### `grid_search(...)`

Evalúa las 81 combinaciones `(p, d, q, window)` y selecciona la que minimiza el RMSE global.

Salidas por ventana: `arima_w{w}_grid.csv`.  
Resumen de mejores: `arima_mejores_por_ventana.csv`.

---

#### `boxplot_ventanas(X_train, y_train, df_distritos_info, df_mejores, ruta_base)`

Para la mejor configuración de cada ventana, genera un boxplot que muestra la distribución del RMSE entre los 180 distritos.

---

#### `pipeline_arima(...)`

Orquesta en orden: grid search → boxplot → análisis departamental → exportación de la mejor configuración global.

Además de los CSVs, guarda las predicciones de la mejor configuración:

| Archivo | Contenido |
|---------|-----------|
| `arima_w{w}_grid.csv` | Todas las combinaciones por ventana |
| `arima_mejores_por_ventana.csv` | Mejor `(p,d,q)` por ventana |
| `arima_best_w{w}_p{p}_d{d}_q{q}.csv` | Métricas de la mejor config global |
| `arima_best_w{w}_p{p}_d{d}_q{q}_departamento.csv` | Por departamento |
| `arima_best_w{w}_p{p}_d{d}_q{q}_global.csv` | Métrica escalar |
| `arima_best_ypred.npy` | Array (180, 5) de predicciones |
| `arima_boxplot_ventanas.png` | Distribución RMSE por ventana |

---

### 4.6 Modelo MLP (`pipeline_mlp.py`)

#### Arquitectura

```
Clase MLP(nn.Module):
    Para cada h en hidden_sizes:
        Linear(prev → h) → ReLU → Dropout(p)
    Linear(h_final → 1)

Ejemplo con hidden_sizes=[64, 32]:
    Input(window_size) → Linear(w→64) → ReLU → Dropout
                       → Linear(64→32) → ReLU → Dropout
                       → Linear(32→1) → ŷ
```

La entrada se aplana: tensor `(batch, window, 1)` → `(batch, window)`.

---

#### `entrenar(X_train_t, y_train_t, hidden_sizes, dropout, epochs, batch_size, lr, patience=10)`

- Pérdida: `MSELoss`.
- Optimizador: `Adam`.
- Early stopping: si el train loss no mejora en `patience=10` épocas consecutivas, se detiene y restaura los pesos del mejor epoch.

```python
retorna: (model, train_losses)
```

---

#### `evaluar_geografico(model, series, df_distritos_info, window_size, tamanio_entrenamiento)`

Evaluación walk-forward sobre la geografía completa (análoga a ARIMA):

```
Para cada distrito i:
    history = series[i, :35]
    Para t en {0, 1, 2, 3, 4}:
        x = tensor(history[-window_size:]).unsqueeze(0)
        ŷ[t] = model(x).item()
        history.append(y_real[i, t])
```

Las métricas por departamento se calculan con el mismo método **pooled** que ARIMA:

```python
y_t, y_p = y_true_total[mask], y_pred_total[mask]
RMSE_dep = sqrt(mean((y_t - y_p)²))
```

```python
retorna: (df_distrito, df_departamento, rmse_global, mae_global, y_pred_wf)
```

---

#### `pipeline_mlp(...)`

1. **Grid search** (96 configuraciones): entrena cada combinación y evalúa en test (ventanas deslizantes).
2. **Mejor modelo**: la configuración con menor `rmse_test`.
3. **Evaluación walk-forward geográfica**: sobre los 180 distritos completos.
4. **Exportaciones**:

| Archivo | Contenido |
|---------|-----------|
| `mlp_resultados.csv` | RMSE/MAE train+test de las 96 configs |
| `mlp_mejor.pth` | State dict + configuración del mejor modelo |
| `mlp_mejor_curva.png` | Evolución de la pérdida por época |
| `mlp_mejores_por_ventana.csv` | Mejor config por tamaño de ventana |
| `mlp_mejor_distrito.csv` | RMSE/MAE walk-forward por distrito |
| `mlp_mejor_departamento.csv` | RMSE/MAE pooled por departamento |
| `mlp_global.csv` | Métrica escalar walk-forward |
| `mlp_mejor_ypred.npy` | Array (180, 5) de predicciones walk-forward |

**Return**

```python
{"modelo": str, "rmse": float, "mae": float, "y_pred": ndarray(180,5)}
```

---

### 4.7 Modelo LSTM (`pipeline_lstm.py`)

#### Arquitectura

```
Clase LSTM(nn.Module):
    nn.LSTM(input_size=1, hidden_size, num_layers, batch_first=True)
        → estado final [batch, hidden_size]
    Dropout(p)
    Linear(hidden_size → 1)
```

A diferencia del MLP, la entrada **no se aplana**: el tensor mantiene la dimensión de secuencia `(batch, window_size, 1)`, lo que permite que el LSTM capture dependencias temporales implícitas.

---

#### Diferencias respecto al MLP

| Aspecto | MLP | LSTM |
|---------|-----|------|
| Forma de entrada | `(batch, window)` — aplanado | `(batch, window, 1)` — secuencial |
| Capas recurrentes | No | Sí (`num_layers`) |
| Hiperparámetro adicional | `hidden_sizes` (lista) | `hidden_size` (escalar) + `num_layers` |

Todos los demás aspectos —entrenamiento, early stopping, evaluación walk-forward, métricas por departamento, exportaciones— son análogos al MLP.

---

**Salidas** (análogas a MLP, con prefijo `lstm_`):

| Archivo | Contenido |
|---------|-----------|
| `lstm_resultados.csv` | RMSE/MAE train+test de las 96 configs |
| `lstm_mejor.pth` | Pesos del mejor modelo |
| `lstm_mejor_curva.png` | Curva de aprendizaje |
| `lstm_mejores_por_ventana.csv` | Mejor config por ventana |
| `lstm_mejor_distrito.csv` | Métricas walk-forward por distrito |
| `lstm_mejor_departamento.csv` | Métricas pooled por departamento |
| `lstm_global.csv` | Métrica escalar |
| `lstm_mejor_ypred.npy` | Array (180, 5) de predicciones |

---

### 4.8 Comparación final (`pipeline_comparacion.py`)

#### `exportar_comparacion(resultados, ruta_csv)`

Tabulación de métricas de los cuatro modelos, ordenada por RMSE ascendente.

```python
entrada:  lista de dicts — [{"modelo", "rmse", "mae"}, ...]
salida:   DataFrame guardado en comparacion_modelos.csv
```

---

#### `graficar_predicciones(resultados, series, df_distritos_info, tamanio_entrenamiento, comparacion_dir, n=5, contexto_anios=5, anio_inicio=1985)`

**Criterio de ranking:** Para cada distrito se calcula el MAE de cada modelo y luego se promedia entre todos los modelos con `y_pred` disponible. Esto da un ranking de *consenso* que no favorece a ningún modelo en particular.

```
mean_MAE[i] = mean over models { mean(|y_true[i] - y_pred_modelo[i]|) }
```

- **Top 5 menores** `mean_MAE` → grupo `mejores`.
- **Top 5 mayores** `mean_MAE` → grupo `peores`.

**Contenido de cada gráfico (10 en total):**

```
Eje X: años del período de análisis
  - contexto: [anio_inicio + 30 ... anio_inicio + 34]  (5 años de entrenamiento)
  - test:     [anio_inicio + 35 ... anio_inicio + 39]  (5 años pronóstico)

Línea negra sólida     → valores reales en el contexto
Línea negra punteada   → valores reales en el período de test
Línea vertical gris    → separación entrenamiento/test
Líneas de color        → predicciones de cada modelo (MAE individual en leyenda)
Título                 → departamento, distrito, geocode, MAE promedio entre modelos
```

Archivos generados: `mejores_01_{geocode}.png` ... `mejores_05_{geocode}.png` y `peores_01_{geocode}.png` ... `peores_05_{geocode}.png`.

---

#### `pipeline_comparacion(resultados, series, df_distritos_info, tamanio_entrenamiento, comparacion_dir, anio_inicio)`

Orquesta las dos funciones anteriores y anuncia el modelo con menor RMSE global.

**Salidas en `comparacion/`:**

| Archivo | Contenido |
|---------|-----------|
| `comparacion_modelos.csv` | Ranking de los 4 modelos por RMSE |
| `mejores_01_*.png` … `mejores_05_*.png` | Gráficos de los 5 distritos mejor predichos |
| `peores_01_*.png` … `peores_05_*.png` | Gráficos de los 5 distritos peor predichos |

---

### 4.9 Orquestador principal (`main.py`)

El orquestador implementa **idempotencia por caché**: si los archivos de salida de un modelo ya existen, los carga desde disco en lugar de re-ejecutar el entrenamiento. En ese caso también intenta cargar el archivo `.npy` de predicciones para que la comparación final cuente con las predicciones de todos los modelos.

#### Función auxiliar `_cargar_ypred(ruta_npy)`

Carga un archivo `.npy` si existe; en caso contrario emite un aviso y devuelve `None` (el modelo se incluye en el ranking CSV pero no en las gráficas de comparación).

#### Flujo completo

```
main()
 │
 ├── PASO 1 — Dataset
 │     cargar_series()
 │     construir_dataset_estadistico()
 │     construir_dataset_dl()
 │
 ├── PASO 2 — Modelos estadísticos
 │     [Persistencia]
 │       Si existe _global.csv → carga (+ .npy si existe)
 │       Si no               → pipeline_persistencia()
 │
 │     [Diagnóstico ARIMA]
 │       Si existen PNGs en analisis_arima/ → omite
 │       Si no                              → generar_analisis_arima()
 │
 │     [ARIMA]
 │       Si existe _mejores_por_ventana.csv → carga mejor fila (+ .npy)
 │       Si no                             → pipeline_arima()
 │
 ├── PASO 3 — Modelos Deep Learning
 │     [MLP]
 │       Si existe _resultados.csv → carga primera fila (+ .npy)
 │       Si no                    → pipeline_mlp()
 │
 │     [LSTM]
 │       Si existe _resultados.csv → carga primera fila (+ .npy)
 │       Si no                    → pipeline_lstm()
 │
 └── PASO 4 — Comparación
       pipeline_comparacion(resultados, series, ...)
```

Todos los modelos devuelven al menos `{"modelo", "rmse", "mae", "y_pred"}` al objeto `resultados` que se pasa a la comparación.

---

## 5. Flujo de datos extremo a extremo

```
[RAW] MapBiomas (40 rasters) + Biomas + Distritos
          │
          ▼
   ┌──────────────────────────────────────────────┐
   │ O1 / R1-R2  Delimitación & Reclasificación   │
   │  - Recorte a Amazonía (40 rasters)           │
   │  - Binario bosque/no-bosque (40 rasters)     │
   └──────────────────┬───────────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────────┐
   │ O1 / R3  Cambios + Zonificación + Series     │
   │  - Mapa de cambio histórico (1 raster)       │
   │  - Top 200 distritos amazónicos              │
   │  - Series temporales pct_bosque (180×40)     │
   └──────────────────┬───────────────────────────┘
                      │
                      ▼
           distritos_entrenamiento.csv
           (180 distritos × 40 años)
                      │
                      ▼
   ┌──────────────────────────────────────────────┐
   │ O2  construir_dataset                        │
   │  series    (180, 40)                         │
   │  X_stat    (180, 35, 1) — train              │
   │  y_stat    (180, 5)     — test               │
   │  dataset_dl (dict por window_size)            │
   └──────────┬─────────────┬────────────────┬────┘
              │             │                │
              ▼             ▼                ▼
      Persistencia        ARIMA          MLP / LSTM
      walk-forward     grid_search      grid_search
      (180 × 5)       (81 configs)     (96 configs)
              │             │                │
              └──────┬──────┘                │
                     └──────────┬────────────┘
                                │
                    evaluar_geografico (walk-forward)
                    y_pred (180, 5) × 4 modelos
                                │
                                ▼
                    pipeline_comparacion
                     comparacion_modelos.csv
                     10 PNG (5 mejores + 5 peores)
```

---

## 6. Convenciones y reproducibilidad

### Semillas de aleatoriedad

| Módulo | Semilla | Uso |
|--------|---------|-----|
| `O1/config.py` | `SEMILLA_SPLIT = 42` | Split 90/10 de distritos |
| `O2/config.py` | `SEMILLA = 42` | Inicialización modelos PyTorch |

La función `fijar_semilla()` en `pipeline_mlp.py` y `pipeline_lstm.py` fija `random`, `numpy`, `torch.manual_seed` y desactiva el benchmark de cuDNN para garantizar determinismo.

### Protocolo de evaluación

Todos los modelos utilizan **evaluación walk-forward multistep**:

1. El modelo recibe como entrada únicamente el histórico hasta el año anterior al primer punto de pronóstico.
2. Después de cada predicción, el valor **real** del año correspondiente se incorpora al histórico.
3. Se repite durante los 5 años del horizonte.

Esto garantiza que las métricas sean comparables entre modelos estadísticos y de aprendizaje profundo.

### Métricas por departamento

Las métricas departamentales se calculan de forma **pooled** (no como promedio de métricas de distritos), igual que la métrica global:

```python
RMSE_dep = sqrt( mean( (y_true[distritos_dep] - y_pred[distritos_dep])² ) )
```

Esto corresponde a la implementación de referencia en `metricas_por_departamento()` de `pipeline_arima.py` y se replica en Persistencia, MLP y LSTM.

### Caché de resultados

`main.py` detecta la existencia de archivos de salida antes de ejecutar cada pipeline. Esto permite:
- Reanudar una ejecución interrumpida sin re-entrenar desde cero.
- Re-generar únicamente las comparaciones finales sin tocar los modelos.

---

## 7. Inventario de archivos de salida

### O1

```
data/interim/O1/
├── mapas-amazonia/
│   ├── peru_amazonia_YYYY.tif            × 40
│   └── metadatos_mapas_amazonia.csv
├── mapas-reclasificados/
│   ├── bosque_nobosque_amazonia_YYYY.tif × 40
│   └── metadatos_mapas_reclasificados_amazonia.csv
├── distritos-amazonia/
│   └── distritos_amazonia.gpkg
├── mapas-cambios/
│   ├── mapa_cambios_1985_2024.tif
│   └── estadisticas_cambios.csv
├── metricas-distritos/
│   ├── mapa_cambios_distrito_1985_2024.gpkg
│   ├── metricas_distritos.csv
│   └── metricas_distritos_resumen.csv
├── distritos-alto-cambio/
│   ├── distritos_alto_cambio.gpkg
│   └── distritos_alto_cambio.csv
└── series-temporales/
    ├── entrenamiento/
    │   ├── distritos_entrenamiento.csv         (180 × 40 registros)
    │   └── estadisticas_distritos_entrenamiento.csv
    └── generalizacion-espacial/
        └── distritos_generalizacion_espacial.csv (20 × 40 registros)
```

### O2

```
data/interim/O2/modelos/
├── persistencia/
│   ├── persistencia_resultados.csv
│   ├── persistencia_resultados_departamento.csv
│   ├── persistencia_resultados_global.csv
│   └── persistencia_resultados_ypred.npy
├── arima/
│   ├── analisis_arima/                    ← 11 PNG exploratorios
│   ├── arima_w{w}_grid.csv                × 9 ventanas
│   ├── arima_mejores_por_ventana.csv
│   ├── arima_best_w{w}_p{p}_d{d}_q{q}.csv
│   ├── arima_best_w{w}_p{p}_d{d}_q{q}_departamento.csv
│   ├── arima_best_w{w}_p{p}_d{d}_q{q}_global.csv
│   ├── arima_best_ypred.npy
│   └── arima_boxplot_ventanas.png
├── mlp/
│   ├── mlp_resultados.csv
│   ├── mlp_mejor.pth
│   ├── mlp_mejor_curva.png
│   ├── mlp_mejores_por_ventana.csv
│   ├── mlp_mejor_distrito.csv
│   ├── mlp_mejor_departamento.csv
│   ├── mlp_global.csv
│   └── mlp_mejor_ypred.npy
├── lstm/
│   ├── lstm_resultados.csv
│   ├── lstm_mejor.pth
│   ├── lstm_mejor_curva.png
│   ├── lstm_mejores_por_ventana.csv
│   ├── lstm_mejor_distrito.csv
│   ├── lstm_mejor_departamento.csv
│   ├── lstm_global.csv
│   └── lstm_mejor_ypred.npy
└── comparacion/
    ├── comparacion_modelos.csv
    ├── mejores_01_{geocode}.png
    ├── mejores_02_{geocode}.png
    ├── mejores_03_{geocode}.png
    ├── mejores_04_{geocode}.png
    ├── mejores_05_{geocode}.png
    ├── peores_01_{geocode}.png
    ├── peores_02_{geocode}.png
    ├── peores_03_{geocode}.png
    ├── peores_04_{geocode}.png
    └── peores_05_{geocode}.png
```
