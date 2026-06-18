# O1 — Documentación Técnica

Referencia única del módulo de preparación de datos para el pipeline de pronóstico de deforestación amazónica peruana. Integra arquitectura, diseño, análisis técnico y decisiones de implementación.

---

## Tabla de contenidos

1. [Visión general](#1-visión-general)
2. [Estructura del proyecto](#2-estructura-del-proyecto)
3. [Flujo de procesamiento](#3-flujo-de-procesamiento)
4. [Módulos y scripts](#4-módulos-y-scripts)
5. [Datos de entrada y salida](#5-datos-de-entrada-y-salida)
6. [Referencia de configuración](#6-referencia-de-configuración)
7. [Decisiones de diseño](#7-decisiones-de-diseño)
8. [Convenciones internas](#8-convenciones-internas)
9. [Reproducibilidad y trazabilidad](#9-reproducibilidad-y-trazabilidad)
10. [Dependencias entre módulos](#10-dependencias-entre-módulos)
11. [Análisis técnico y mejoras aplicadas](#11-análisis-técnico-y-mejoras-aplicadas)

---

## 1. Visión general

O1 transforma datos satelitales crudos (MapBiomas Perú, colección 3, 1985–2024) en un panel tabular estructurado listo para el entrenamiento de modelos predictivos. El módulo resuelve dos problemas fundamentales: la complejidad del dato geoespacial (rasters multi-clase, multi-año, multi-escala) y la necesidad de reducir dimensionalidad sin perder la señal de deforestación.

La arquitectura está dividida en dos etapas lineales e independientes:

- **R1–R2** — Preparación y clasificación binaria del dato raster.
- **R3** — Detección de cambios, zonificación espacial y extracción de series temporales.

**Producto final:** panel en formato largo con 8 000 observaciones (200 distritos × 40 años), dividido en 7 200 filas de entrenamiento (180 distritos) y 800 de generalización espacial (20 distritos).

---

## 2. Estructura del proyecto

### Árbol de código fuente

```
src/O1/
├── __init__.py                          — marca el paquete Python
├── config.py                            — configuración centralizada: rutas, constantes, logging
├── utils.py                             — utilidades compartidas: guardar_csv(), log_config()
├── O1_DOCUMENTATION.md                  — este documento (fuente única de verdad)
│
├── r1_r2/                               — Etapa 1-2: recorte y reclasificación binaria
│   ├── __init__.py
│   ├── main.py                          — entrypoint R1-R2
│   ├── pipeline.py                      — procesamiento anual (4 etapas)
│   └── delimitacion_mapa_amazonas.py    — delimitación amazónica y recorte de rasters
│
├── r3/                                  — Etapa 3: cambios, zonificación y series temporales
│   ├── __init__.py
│   ├── main.py                          — entrypoint R3 (def main())
│   ├── deteccion_cambios.py             — comparación temporal por tiles
│   ├── zonificacion_distrito.py         — estadísticas zonales por distrito
│   ├── distritos_alto_cambio.py         — selección top-N distritos
│   └── series_temporales.py            — extracción de panel + split 90/10
│
└── test/                                — validaciones puntuales
    ├── test_reclasificado.py            — visualización de rasters reclasificados
    ├── test_crs.py                      — validación de sistemas de referencia
    └── _deprecated_reproyeccion.py      — descartado (no usar)
```

### Árbol de datos

```
data/
├── raw/
│   ├── mapbiomas-peru/                  — 40 rasters crudos MapBiomas C3 (1985-2024)
│   │   └── peru_collection3_integration_v1-classification_YYYY.tif
│   ├── biomas-peru/
│   │   └── BIOMES_v1.shp               — polígono Amazonía peruana
│   └── distritos-peru/
│       └── POLITICAL_LEVEL_4_v1.shp    — 1 874 distritos administrativos
│
└── interim/
    └── O1/
        ├── mapas-amazonia/              — rasters crudos recortados a bbox amazónica
        │   ├── peru_amazonia_YYYY.tif   (40 archivos)
        │   └── metadatos_mapas_amazonia.csv
        ├── mapas-reclasificados/        — rasters binarios bosque/no-bosque
        │   ├── bosque_nobosque_amazonia_YYYY.tif  (40 archivos, uint8)
        │   └── metadatos_mapas_reclasificados_amazonia.csv
        ├── mapas-cambios/
        │   ├── mapa_cambios_1985_2024.tif
        │   └── estadisticas_cambios.csv
        ├── distritos-amazonia/
        │   ├── distritos_amazonia.gpkg
        │   └── distritos_amazonia.csv
        ├── metricas-distritos/
        │   ├── mapa_cambios_distrito_1985_2024.gpkg
        │   ├── estadisticas_cambios_distritos.csv
        │   └── estadisticas_cambios_distritos_resumen.csv
        ├── distritos-alto-cambio/
        │   ├── distritos_alto_cambio.gpkg
        │   └── distritos_alto_cambio.csv
        └── series-temporales/
            ├── entrenamiento/
            │   ├── distritos_entrenamiento.csv          — 7 200 filas (180 × 40)
            │   ├── distritos_entrenamiento.gpkg
            │   └── estadisticas_distritos_entrenamiento.csv
            └── generalizacion-espacial/
                ├── distritos_generalizacion_espacial.csv  — 800 filas (20 × 40)
                ├── distritos_generalizacion_espacial.gpkg
                └── estadisticas_distritos_generalizacion_espacial.csv
```

---

## 3. Flujo de procesamiento

```
ENTRADAS (data/raw/)
  40 rasters MapBiomas C3  +  shapefile Amazonía  +  shapefile distritos
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ R1-R2 / delimitacion_mapa_amazonas.py                   │
│                                                         │
│  1. Identifica distritos con >50% de área en Amazonía   │
│     → distritos_amazonia.gpkg                           │
│  2. Recorta 40 rasters crudos a bbox amazónica          │
│     → peru_amazonia_YYYY.tif                            │
└────────────────────┬────────────────────────────────────┘
                     │
        ▼
┌─────────────────────────────────────────────────────────┐
│ R1-R2 / pipeline.py  (×40 años, idempotente)            │
│                                                         │
│  E1: Carga raster + extrae metadatos                    │
│  E2: Depura clases inválidas → NODATA (255)             │
│  E3: Reclasifica bosque=1 / no-bosque=0 / nodata=255    │
│  E4: Exporta GeoTIFF uint8 LZW + metadatos CSV         │
│     → bosque_nobosque_amazonia_YYYY.tif                 │
└────────────────────┬────────────────────────────────────┘
                     │
        ▼
┌─────────────────────────────────────────────────────────┐
│ R3 PASO 1 / deteccion_cambios.py  (idempotente)         │
│                                                         │
│  Stack 3D (tiempo × y × x) procesado por tiles         │
│  Detecta píxeles con ANY(t₁ ≠ t₂) en serie temporal    │
│     → mapa_cambios_1985_2024.tif  {0, 1, 255}          │
│     → estadisticas_cambios.csv                          │
└────────────────────┬────────────────────────────────────┘
                     │
        ▼
┌─────────────────────────────────────────────────────────┐
│ R3 PASO 2 / zonificacion_distrito.py  (idempotente)     │
│                                                         │
│  zonal_stats() sobre mapa de cambios por distrito       │
│  Calcula píxeles válidos, cambiados, % cambio           │
│     → mapa_cambios_distrito_1985_2024.gpkg              │
│     → estadisticas_cambios_distritos.csv + _resumen     │
└────────────────────┬────────────────────────────────────┘
                     │
        ▼
┌─────────────────────────────────────────────────────────┐
│ R3 PASO 3 / distritos_alto_cambio.py  (idempotente)     │
│                                                         │
│  Ranking descendente por porcentaje de cambio           │
│  Selecciona top-200 distritos                           │
│     → distritos_alto_cambio.gpkg                        │
│     → distritos_alto_cambio.csv                         │
└────────────────────┬────────────────────────────────────┘
                     │
        ▼
┌─────────────────────────────────────────────────────────┐
│ R3 PASO 4 / series_temporales.py  (idempotente)         │
│                                                         │
│  Split aleatorio 90/10 por distrito (semilla=42)        │
│  zonal_stats() sobre 40 rasters binarios × 200 distr.  │
│  Columnas: geocode, departamento, distrito, anio,       │
│            pix_total, pix_bosque, pix_no_bosque,        │
│            pct_bosque, pct_no_bosque                    │
│                                                         │
│     → distritos_entrenamiento.csv    (180 × 40 = 7 200) │
│     → distritos_generalizacion_espacial.csv (20 × 40)   │
└─────────────────────────────────────────────────────────┘
                     │
        ▼
SALIDA para O2 (modelos predictivos)
```

Ver sección 10.

---

## 4. Módulos y scripts

### `config.py`
Centraliza todas las rutas, constantes y parámetros del pipeline. Configura `logging.basicConfig` al importarse, lo que garantiza formato uniforme en todos los módulos.

Crea los directorios de salida de forma idempotente (`os.makedirs(exist_ok=True)`) al importar el módulo.

### `utils.py`
Utilidades compartidas entre R1-R2 y R3.

| Función | Descripción |
|---|---|
| `guardar_csv(gdf, ruta)` | Elimina columna `geometry` y guarda como CSV |
| `log_config()` | Registra vía `logging.info` todos los parámetros activos al inicio de ejecución |

### `r1_r2/main.py`
Entrypoint de la etapa R1-R2. Ejecuta bajo `if __name__ == "__main__"`. Al iniciar llama a `log_config()`.

Flujo:
1. Si no existen `distritos_amazonia.gpkg` ni rasters en `mapas-amazonia/`, ejecuta `pipeline_delimitacion_amazonia()`
2. Itera sobre `ANIOS` llamando a `ejecutar_pipeline_anio(anio)`
3. Consolida metadatos en dos CSVs al finalizar

### `r1_r2/delimitacion_mapa_amazonas.py`
**Función `identificar_distritos_amazonia_interseccion()`**
- Lee shapefile de biomas y distritos
- Verifica CRS coincidente antes de operar
- Reproyecta a `EPSG:32718` para cálculo de áreas (métrico)
- Filtra distritos con `porcentaje_amazonia > UMBRAL_AMAZONIA`
- Reproyecta resultado a `EPSG:4326` para almacenamiento
- Guarda GeoPackage + CSV

**Función `recortar_mapas_amazonia()`**
- Recorta cada raster crudo a la geometría unificada de Amazonía
- Salida: `peru_amazonia_YYYY.tif`
- Idempotente: skip si el archivo ya existe

### `r1_r2/pipeline.py`
Procesamiento de un año individual en 4 etapas:

| Etapa | Función | Descripción |
|---|---|---|
| E1 | `etapa1_cargar_y_verificar()` | Carga raster, extrae metadatos (CRS, resolución, clases únicas, tamaño) |
| E2 | `etapa2_validar_y_depurar_clases()` | Elimina clase 27 (no-observado) y clases fuera de `CLASES_VALIDAS` → 255 |
| E3 | `etapa3_reclasificar()` | Produce raster binario: bosque=1, no-bosque=0, nodata=255 |
| E4 | `etapa4_exportar()` | Guarda GeoTIFF uint8 LZW; calcula áreas en km² con `PIXEL_AREA_KM2` |

### `r3/main.py`
Entrypoint R3 con estructura `def main()`. Al iniciar llama a `log_config()`.

Cuatro pasos, todos idempotentes mediante `if os.path.exists(ruta_salida)`. Ver sección 10 para el orden de ejecución y outputs por paso.

### `r3/deteccion_cambios.py`
Detecta qué píxeles cambiaron de valor (bosque ↔ no-bosque) en algún momento de la serie 1985–2024.

**Estrategia por tiles:**
- Dimensiones imagen: ~50 000 × 50 000 px
- Tiles de `TAMANIO_TILE × TAMANIO_TILE` px (default 5 000)
- Para cada tile: stack 3D `(tiempo, y, x)` con `np.any(t[:-1] != t[1:], axis=0)`
- Libera memoria después de cada tile

**Semántica de salida:** `{0: sin cambio, 1: cambió, 255: nodata}`

### `r3/zonificacion_distrito.py`
Calcula cuántos píxeles cambiaron dentro de cada distrito amazónico.

Usa `rasterstats.zonal_stats()` con `categorical=True` sobre el mapa de cambios. Produce:
- GeoPackage con métricas por distrito
- CSV detallado (columnas renombradas para legibilidad)
- CSV resumen (min/max/promedio global)

### `r3/distritos_alto_cambio.py`
Ordena todos los distritos amazónicos por `porcentaje_cambio` descendente y selecciona los primeros `N_DISTRITOS_ALTO_CAMBIO`. Guarda GeoPackage + CSV.

### `r3/series_temporales.py`
Extrae la serie temporal de cobertura forestal para cada distrito seleccionado.

**`split_aleatorio(gdf)`** — divide los 200 distritos en 90% train / 10% test por distrito completo; ver sección 7.4.

**`extraer_series(gdf, rutas_mapas_reclasificados)`** — para cada año:
- `zonal_stats()` con `categorical=True` sobre raster binario
- Cuenta `pix_bosque` (valor 1) y `pix_no_bosque` (valor 0)
- Calcula `pct_bosque = pix_bosque / (pix_bosque + pix_no_bosque)`

---

## 5. Datos de entrada y salida

### Entradas (data/raw/)

| Recurso | Formato | Descripción |
|---|---|---|
| `mapbiomas-peru/peru_collection3_*_YYYY.tif` | GeoTIFF int32 | 40 rasters anuales MapBiomas C3, valores 0–72 |
| `biomas-peru/BIOMES_v1.shp` | Shapefile | Polígonos de biomas peruanos; se filtra `NAME == "[Amazonía]"` |
| `distritos-peru/POLITICAL_LEVEL_4_v1.shp` | Shapefile | 1 874 distritos administrativos de Perú |

### Salidas intermedias (data/interim/O1/)

| Archivo | Formato | Dimensión | Generado por |
|---|---|---|---|
| `mapas-amazonia/peru_amazonia_YYYY.tif` | GeoTIFF | 40 archivos | R1-R2 delimitacion |
| `mapas-reclasificados/bosque_nobosque_amazonia_YYYY.tif` | GeoTIFF uint8 | 40 archivos | R1-R2 pipeline |
| `distritos-amazonia/distritos_amazonia.gpkg` | GeoPackage | ~300 distritos | R1-R2 delimitacion |
| `mapas-cambios/mapa_cambios_1985_2024.tif` | GeoTIFF uint8 | 1 archivo | R3 paso 1 |
| `metricas-distritos/mapa_cambios_distrito_*.gpkg` | GeoPackage | ~300 distritos | R3 paso 2 |
| `distritos-alto-cambio/distritos_alto_cambio.gpkg` | GeoPackage | 200 distritos | R3 paso 3 |

### Salidas finales (para O2)

| Archivo | Filas | Columnas clave |
|---|---|---|
| `series-temporales/entrenamiento/distritos_entrenamiento.csv` | 7 200 | `geocode`, `departamento`, `distrito`, `anio`, `pix_total`, `pix_bosque`, `pix_no_bosque`, `pct_bosque`, `pct_no_bosque` |
| `series-temporales/generalizacion-espacial/distritos_generalizacion_espacial.csv` | 800 | ídem |

---

## 6. Referencia de configuración

Todas las constantes residen en `config.py`. Modificar aquí afecta todo el pipeline automáticamente.

### Rango temporal

| Constante | Valor | Descripción |
|---|---|---|
| `ANIOS` | `range(1985, 2025)` | 40 años de serie temporal |

### Clases MapBiomas (colección 3)

| Constante | Valor | Descripción |
|---|---|---|
| `CLASES_BOSQUE` | `{3, 4, 5, 6}` | Clases que se clasifican como bosque (ATBD MapBiomas Perú C3) |
| `CLASES_VALIDAS` | ~28 clases | Todas las clases con representación ecológica válida en Amazonía Perú |
| `CLASE_NOBSERVADO` | `27` | Clase "no observado" de MapBiomas; se convierte a NODATA |
| `NODATA` | `255` | Valor NoData canónico; compatible con uint8 y convención MapBiomas |

### Sistemas de referencia

Ver sección 7.5.

### Parámetros del pipeline

| Constante | Valor | Descripción |
|---|---|---|
| `UMBRAL_AMAZONIA` | `0.50` | Fracción mínima del área distrital dentro de Amazonía |
| `N_DISTRITOS_ALTO_CAMBIO` | `200` | Top-N distritos seleccionados por % de cambio |
| `PIXEL_AREA_KM2` | `0.0009` | Área de un píxel (30 m × 30 m = 0,0009 km²) |
| `TAMANIO_TILE` | `5000` | Lado del tile en píxeles para procesamiento por bloques |
| `TAMANIO_ENTRENAMIENTO` | `0.9` | Proporción del split de entrenamiento (90%) |
| `SEMILLA_SPLIT` | `42` | Semilla fija para reproducibilidad del split |

### Logging

Ver sección 4 (`utils.py` y entrypoints de R1-R2 y R3).

---

## 7. Decisiones de diseño

### 7.1 Reclasificación binaria (R1–R2)

**Decisión: Reducir a bosque / no-bosque**

MapBiomas ofrece ~30 clases de uso de suelo. La decisión de colapsar esas clases en una distinción binaria —bosque (1) vs. no-bosque (0)— responde a tres razones:

1. **Enfoque en la variable objetivo.** El fenómeno a predecir es pérdida de cobertura forestal, no la transición entre subclases de no-bosque. La granularidad adicional no aporta señal predictiva para ese objetivo.

2. **Consistencia temporal.** Las categorías de MapBiomas han variado entre versiones de la colección. La binarización aisla el pipeline de esos cambios de nomenclatura, garantizando que la comparación 1985–2024 sea metodológicamente coherente.

3. **Reducción de complejidad downstream.**

---

**Decisión: NoData = 255 (uint8)**

El valor 255 en un raster de 8 bits cumple un rol estructural: marca píxeles no observados (nubes, sombras, bordes) sin introducir ambigüedad. Esta elección es coherente con la convención de MapBiomas, evita conflictos con los valores semánticos 0 y 1, y permite almacenar los rasters en el tipo más compacto disponible.

---

**Decisión: Depuración de clases inválidas antes de reclasificar**

Antes de aplicar la regla bosque/no-bosque se eliminan clases sin representación ecológica válida en la Amazonía peruana (ej. clase 27, "no observado"). Si estas clases pasaran a la reclasificación, contaminarían las métricas de área forestal de años con alta cobertura de nube.

---

### 7.2 Detección de cambios (R3.1)

**Decisión: Comparación de años consecutivos**

El mapa de cambios se construye comparando cada par de años adyacentes, no el raster inicial contra el final. Esto permite detectar tanto deforestación como regeneración (no-bosque → bosque), y preserva la señal de pulsos episódicos de deforestación que se habrían suavizado en una comparación directa 1985 vs. 2024.

---

**Decisión: Procesamiento por teselas (TAMANIO_TILE × TAMANIO_TILE píxeles)**

Cargar el stack temporal completo (~50 000 × 50 000 × 40 capas) en memoria es inviable en hardware convencional. El procesamiento por teselas independientes resuelve esa restricción sin alterar el resultado, porque la detección de cambio es una operación píxel a píxel sin dependencias espaciales entre vecinos.

---

### 7.3 Unidad de análisis: el distrito administrativo (R3.2–R3.3)

**Decisión: Agregar a nivel distrital, no mantener resolución de píxel**

El distrito administrativo es la unidad de observación del pipeline. Cada distrito acumula los píxeles de bosque y no-bosque dentro de su geometría, produciendo una serie temporal de cobertura por entidad geográfica estable.

Esta elección tiene dos consecuencias metodológicas directas:

- La unidad de análisis coincide con la escala a la que operan las decisiones de política forestal en Perú, lo que da relevancia aplicada al pronóstico producido por los modelos downstream.

---

**Decisión: Estadísticas zonales en lugar de intersección vectorial**

Para calcular cuántos píxeles cambiaron dentro de cada distrito se usa estadística zonal sobre el raster (rasterstats), en lugar de intersectar geometrías vectoriales. Las razones son:

- Las operaciones raster son órdenes de magnitud más rápidas que la intersección polígono-polígono sobre 200 distritos y 40 años.
- Las operaciones vectoriales generan artefactos geométricos (slivers) en los bordes de distrito que introducen ruido sin valor analítico.
- El raster es el mismo para todas las consultas temporales; el costo de apertura se paga una sola vez por año.

---

**Decisión: Selección de los N distritos de mayor cambio (`N_DISTRITOS_ALTO_CAMBIO`)**

El corpus de entrenamiento se restringe a los distritos con mayor densidad de cambio forestal:

1. **Concentrar señal.** Los distritos con cambio mínimo no aportan información predictiva útil; incluirlos diluye el conjunto de entrenamiento y perjudica los modelos.
2. **Separación espacial del conjunto de prueba.** Los distritos no seleccionados (de menor cambio) forman un conjunto de generalización espacial que permite evaluar si el modelo generaliza a zonas que no vio durante el entrenamiento —una métrica más exigente que una partición aleatoria.

---

### 7.4 Estructura del panel de salida (R3.4)

**Decisión: Formato largo (long format), una fila por (distrito, año)**

La salida final es un panel en formato largo: cada combinación distrito–año ocupa una fila, con columnas de píxeles de bosque, no-bosque y porcentaje de cobertura.

---

**Decisión: Partición 90% / 10% (entrenamiento / generalización espacial)**

La partición espacial utiliza el 10% de los distritos seleccionados como conjunto de generalización. Esta proporción es más conservadora que el 70/30 convencional porque el conjunto de 200 × 40 filas (8 000 observaciones) ya es reducido; un 20% de prueba recortaría demasiado el entrenamiento. La semilla fija (`SEMILLA_SPLIT = 42`) garantiza que la partición sea reproducible en todas las ejecuciones.

---

### 7.5 Sistema de coordenadas

**Decisión: Doble CRS (EPSG:32718 + EPSG:4326)**

El pipeline mantiene dos sistemas de referencia con roles distintos:

| CRS | Uso | Razón |
|---|---|---|
| EPSG:32718 (UTM Zona 18S) | Cálculo de áreas, filtro de umbral `UMBRAL_AMAZONIA` | Proyección métrica; error <2% en latitudes peruanas |
| EPSG:4326 (WGS84) | Almacenamiento de geometrías, operaciones con rasters | Formato nativo de los shapefiles fuente |

La alternativa de proyectar todo a un único CRS fue descartada porque reproyectar rasters grandes introduce errores de remuestreo y aumenta el tiempo de procesamiento sin beneficio en la métrica final (porcentaje, no metros cuadrados absolutos).

---

### 7.6 Idempotencia y checkpoints

**Decisión: Cada etapa verifica si su salida ya existe**

Antes de ejecutar cualquier paso costoso (detección de cambios, zonificación, extracción de series), el pipeline comprueba si el archivo de salida existe. Si existe, omite el paso. Esta propiedad tiene consecuencias prácticas importantes:

- Permite reanudar una ejecución interrumpida sin recomputar desde cero.
- Facilita el desarrollo iterativo: se puede modificar R3.4 sin volver a ejecutar R3.1.
- Hace el pipeline auditable: los archivos intermedios pueden inspeccionarse en cualquier momento.

El costo es que cambios en parámetros upstream no invalidan automáticamente los outputs downstream; la responsabilidad de borrar outputs obsoletos recae en el usuario.

---

### 7.7 Trazabilidad de metadatos

Cada etapa genera un CSV de metadatos paralelo al output principal: distribución de clases, recuentos de píxeles, área en km², estadísticas de cambio por distrito. Estos archivos sirven tres propósitos:

1. **Auditoría de calidad** durante el desarrollo, sin necesidad de abrir los rasters.
2. **Trazabilidad** para la tesis: permiten reportar estadísticas descriptivas sin depender de una re-ejecución del pipeline.
3. **Detección temprana de errores**: una distribución de clases anómala en el CSV de metadatos es una señal de alerta antes de que el error se propague a las series temporales.

---

## 8. Convenciones internas

### Nomenclatura de archivos

| Tipo | Patrón | Ejemplo |
|---|---|---|
| Raster crudo recortado | `peru_amazonia_YYYY.tif` | `peru_amazonia_1985.tif` |
| Raster binario | `bosque_nobosque_amazonia_YYYY.tif` | `bosque_nobosque_amazonia_2024.tif` |
| Mapa de cambios | `mapa_cambios_AAAA_BBBB.tif` | `mapa_cambios_1985_2024.tif` |
| Vectorial con métricas | `mapa_cambios_distrito_AAAA_BBBB.gpkg` | — |
| CSV de estadísticas | `estadisticas_{entidad}.csv` | — |
| CSV de resumen | `estadisticas_{entidad}_resumen.csv` | — |

### Estándares de datos

| Aspecto | Estándar |
|---|---|
| dtype rasters | `uint8` (valores posibles: 0, 1, 255) |
| Compresión rasters | LZW (reversible, buena tasa) |
| NoData | `255` |
| CRS almacenamiento vectorial | EPSG:4326 |
| Encoding vectorial | UTF-8 |
| Driver vectorial | GPKG (GeoPackage) |
| Formato tabular | CSV (formato largo, una fila por entidad-año) |
| Columnas tabular | snake_case en español (`pct_bosque`, `pix_total`) |

### Principios de diseño

| Principio | Manifestación en O1 |
|---|---|
| Escalabilidad de memoria | Procesamiento por tiles en R3.1 |
| Alineación con el objetivo | Binarización bosque/no-bosque; unidad distrital |
| Reproducibilidad | Semilla fija, CRS explícitos, NoData canónico, `log_config()` |
| Separación de responsabilidades | R1–R2 limpia; R3 agrega; O2 construye características |
| Resiliencia operativa | Checkpoints idempotentes en cada paso |
| Trazabilidad | Metadatos CSV paralelos a cada output |
| Configuración centralizada | Todos los parámetros en `config.py`; cero hardcoding |

---

## 9. Reproducibilidad y trazabilidad

### Garantías de reproducibilidad

- **Semilla fija:** ver secciones 6 y 7.4.
- **Configuración centralizada:** modificar `config.py` es el único punto de cambio; todos los módulos lo importan.
- **CRS explícitos:** ver sección 7.5.
- **Log de configuración:** `log_config()` imprime todos los parámetros activos al inicio de cada ejecución.
- **Idempotencia:** ver sección 7.6.
- **Metadatos CSV por etapa:** ver sección 7.7.

### Limitaciones conocidas

- **El split no verifica el orden del GDF:** si `distritos_alto_cambio.gpkg` se regenera con un orden diferente, la semilla produce particiones distintas. Mitigation: no re-ejecutar R3 pasos 3–4 si el paso 2 no cambió.
- **Sin checksums en outputs:** no hay validación cruzada entre el CSV de metadatos y el raster correspondiente.
- **Cambios en `config.py` no invalidan outputs existentes:** si se modifica `UMBRAL_AMAZONIA` o `N_DISTRITOS_ALTO_CAMBIO`, hay que borrar manualmente los outputs afectados y re-ejecutar.

---

## 10. Dependencias entre módulos

### Grafo de importaciones

```
config.py
    ├── r1_r2/delimitacion_mapa_amazonas.py  (ANIOS, CRS_PROYECTADO, UMBRAL_AMAZONIA)
    ├── r1_r2/pipeline.py                    (MAPAS_RECLAS_DIR, CLASES_*, NODATA, PIXEL_AREA_KM2)
    ├── r3/deteccion_cambios.py              (NODATA)
    ├── r3/zonificacion_distrito.py          (NODATA)
    ├── r3/distritos_alto_cambio.py          (N_DISTRITOS_ALTO_CAMBIO)
    └── r3/series_temporales.py             (NODATA, ANIOS, SEMILLA_SPLIT, TAMANIO_ENTRENAMIENTO)

utils.py
    ├── (importa de config.py)
    ├── r1_r2/delimitacion_mapa_amazonas.py  (guardar_csv)
    ├── r1_r2/main.py                        (log_config)
    └── r3/main.py                           (log_config)
```

### Orden de ejecución requerido

```
r1_r2/delimitacion_mapa_amazonas.py   → produce distritos_amazonia.gpkg
         ↓
r1_r2/pipeline.py (×40 años)          → produce bosque_nobosque_amazonia_YYYY.tif
         ↓
r3/deteccion_cambios.py               → consume los 40 rasters binarios
         ↓
r3/zonificacion_distrito.py           → consume mapa_cambios + distritos_amazonia.gpkg
         ↓
r3/distritos_alto_cambio.py           → consume mapa_cambios_distrito.gpkg
         ↓
r3/series_temporales.py               → consume distritos_alto_cambio.gpkg + 40 rasters
```

---

## 11. Análisis técnico y mejoras aplicadas

Esta sección documenta los hallazgos del análisis técnico realizado sobre el módulo y las mejoras que se aplicaron como resultado.

### 11.1 Mejoras aplicadas

#### Centralización de parámetros en `config.py`

**Problema:** Cuatro valores con impacto metodológico estaban hardcodeados en los scripts:
- `0.50` (umbral Amazonía) en `delimitacion_mapa_amazonas.py`
- `200` (top-N distritos) en `distritos_alto_cambio.py`
- `0.0009` (área de píxel) en `pipeline.py`
- `5000` (tamaño de tile) en `r3/main.py`

**Solución:** Todos movidos a `config.py` como `UMBRAL_AMAZONIA`, `N_DISTRITOS_ALTO_CAMBIO`, `PIXEL_AREA_KM2`, `TAMANIO_TILE`.

---

#### Creación de `utils.py` y eliminación de duplicación

**Problema:** La función `guardar_csv()` estaba definida de forma idéntica en `r1_r2/delimitacion_mapa_amazonas.py` y en `r3/delimitacion_distritos_amazonas.py`.

**Solución:** Creado `utils.py` con la función compartida. Ambos módulos ahora la importan desde allí.

---

#### Configuración de logging y trazabilidad de ejecución

`logging.basicConfig` en `config.py` + `log_config()` en `utils.py` llamada al inicio de ambos entrypoints; ver sección 4.

---

#### Eliminación de código muerto

**Problema:** `r3/delimitacion_distritos_amazonas.py` era un módulo importado en `r3/main.py` pero cuya función nunca se llamaba. Además, su lógica era una versión reducida (sin umbral del 50%, sin reproyección) de `r1_r2/delimitacion_mapa_amazonas.py`.

**Solución:** Archivo eliminado. Import correspondiente removido de `r3/main.py`.

---

#### Idempotencia completa en todos los pasos de R3

Añadida guardia `if os.path.exists(ruta_distritos_alto_cambio)` al Paso 3 de `r3/main.py`; ver sección 7.6 para la justificación del patrón.

---

#### Eliminación de imports muertos en `r3/main.py`

**Problema:** Tras eliminar el módulo de código muerto, quedaron `BIOMAS_PERU_DIR` y `DISTRITOS_PERU_DIR` importados de `config.py` pero sin uso.

**Solución:** Removidos del bloque de imports.

---

#### Eliminación de `return` vacío en `r3/main.py`

**Problema:** `return` sin valor al final de `main()`, sin propósito.

**Solución:** Línea eliminada.

---

#### Eliminación de cálculo redundante de `np.unique` en `pipeline.py`

**Problema:** `np.unique(bosque_bin, return_counts=True)` se llamaba dentro del bloque `with rasterio.open(...)` (resultado inmediatamente descartado) y luego `np.unique(bosque_bin)` de nuevo fuera del bloque para construir `info`.

**Solución:** La llamada única se movió a fuera del bloque `with`; `valores` se reutiliza para `clases_unicas_salida` en lugar de una tercera llamada.

---

### 11.2 Problemas identificados pendientes de resolución

| # | Problema | Impacto | Módulo |
|---|---|---|---|
| 1 | `ejecutar_pipeline_anio` no verifica si el raster de salida ya existe | Re-procesa 40 rasters en cada ejecución aunque ya estén generados | `r1_r2/pipeline.py` |
| 2 | Split no estable si el GDF se carga en orden diferente | Riesgo metodológico de reproducibilidad | `r3/series_temporales.py` |
| 3 | Inconsistencia de estructura de entrypoints | `r1_r2/main.py` usa código directo bajo `if __name__`; `r3/main.py` usa `def main()` | `r1_r2/main.py` |
| 4 | CSVs de salida mezclan nombres raw con nombres de display | `distritos_alto_cambio.py` y `zonificacion_distrito.py` usan `"Departamento"`, `"% Cambio"` vs. snake_case de `series_temporales.py` | `r3/distritos_alto_cambio.py`, `r3/zonificacion_distrito.py` |
