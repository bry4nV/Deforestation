# Pipeline R3 - Detección de Cambios y Zonificación

## 📋 Descripción General

Pipeline completo para identificar y zonificar áreas de deforestación en la Amazonía peruana (1985-2024).

**Pasos:**
1. **Detección de cambios**: Identifica píxeles que cambiaron entre bosque y no bosque
2. **Zonificación**: Agrupa píxeles contiguos en zonas usando componentes conectados

---

## 🚀 Ejecución Rápida

### Pipeline Completo (Detección + Zonificación)

```bash
cd src
python -m O1.r3.main
```

**Outputs generados:**
- `data/interim/O1/mapas-cambios/mapa_cambios_1985_2024.tif` - Mapa binario de cambios
- `data/interim/O1/mapas-cambios/estadisticas_cambios.txt` - Estadísticas de cambios
- `data/interim/O1/zonas/zonas_conectividad8.tif` - Mapa de zonas identificadas
- `data/interim/O1/zonas/estadisticas_zonas.csv` - Métricas por zona
- `data/interim/O1/zonas/distribucion_areas_zonas.png` - Histograma de áreas

---

## 📦 Módulos del Pipeline

### 1. Detección de Cambios (`deteccion_cambios.py`)

Identifica píxeles que transitaron entre bosque y no bosque en algún momento.

**Características:**
- Procesamiento por tiles (5000×5000 px) para eficiencia de memoria
- Detecta cambios bidireccionales: bosque→no bosque Y no bosque→bosque
- Output: mapa binario (1=cambió, 0=estable, 255=nodata)

**Uso standalone:**
```python
from O1.r3.deteccion_cambios import detectar_cambios_por_tiles

mapa, transform, crs, stats = detectar_cambios_por_tiles(
    raster_paths=["path1.tif", "path2.tif", ...],
    tile_size=5000
)
```

---

### 2. Zonificación (`zonificacion.py`)

Agrupa píxeles contiguos en zonas mediante **componentes conectados** (conectividad 8).

**Características:**
- Usa `scipy.ndimage.label` con estructura 3×3 (incluye diagonales)
- Filtra zonas por área mínima/máxima
- Calcula métricas: área, centroide, bounding box
- Exporta raster, CSV y visualizaciones

**Parámetros:**
- `area_min_km2`: 50 km² (default) - zonas más pequeñas se descartan
- `area_max_km2`: 2000 km² (default) - zonas mayores se marcan con advertencia

**Uso standalone:**
```python
from O1.r3.zonificacion import pipeline_zonificacion

resumen = pipeline_zonificacion(
    mapa_cambios_path="path/to/mapa_cambios.tif",
    output_dir="output/",
    area_min_km2=50,
    area_max_km2=2000
)
```

---

## 🧪 Testing

### Test rápido de zonificación

```bash
cd src
python -m O1.r3.test_zonificacion
```

### Validar outputs

```python
import rasterio
import pandas as pd
import numpy as np

# Verificar mapa de cambios
with rasterio.open("data/interim/O1/mapas-cambios/mapa_cambios_1985_2024.tif") as src:
    cambios = src.read(1)
    print(f"Píxeles con cambio: {np.sum(cambios == 1):,}")

# Verificar zonas
with rasterio.open("data/interim/O1/zonas/zonas_conectividad8.tif") as src:
    zonas = src.read(1)
    n_zonas = len(np.unique(zonas[zonas > 0]))
    print(f"Zonas identificadas: {n_zonas:,}")

# Verificar CSV
df = pd.read_csv("data/interim/O1/zonas/estadisticas_zonas.csv")
print(f"\nTop 5 zonas más grandes:")
print(df[['zona_id', 'area_km2', 'n_pixels']].head())
```

---

## 📊 Estructura de Outputs

### Mapa de Cambios (`mapa_cambios_1985_2024.tif`)
- **Tipo:** GeoTIFF uint8
- **Valores:** 
  - `1` = Píxel cambió (bosque ↔ no bosque)
  - `0` = Píxel estable
  - `255` = NoData
- **CRS:** EPSG:4326

### Mapa de Zonas (`zonas_conectividad8.tif`)
- **Tipo:** GeoTIFF int32
- **Valores:** 
  - `0` = Sin zona
  - `1, 2, 3, ...` = IDs de zonas
- **CRS:** EPSG:4326

### Estadísticas de Zonas (`estadisticas_zonas.csv`)

| Campo             | Descripción                              |
|-------------------|------------------------------------------|
| zona_id           | ID único de la zona                      |
| area_km2          | Área en km²                              |
| n_pixels          | Número de píxeles                        |
| centroid_lat      | Latitud del centroide                    |
| centroid_lon      | Longitud del centroide                   |
| bbox_minr         | Bounding box (fila mínima)               |
| bbox_minc         | Bounding box (columna mínima)            |
| bbox_maxr         | Bounding box (fila máxima)               |
| bbox_maxc         | Bounding box (columna máxima)            |
| es_muy_grande     | True si excede area_max_km2              |

---

## ⚙️ Configuración

Parámetros definidos en `O1/config.py`:

```python
# Directorios
MAPAS_CAMBIOS_DIR = "data/interim/O1/mapas-cambios/"
ZONAS_DIR = "data/interim/O1/zonas/"

# Años de análisis
ANIOS = list(range(1985, 2025))  # 1985-2024
```

---

## 🔍 Algoritmo: Connected Components

### ¿Por qué conectividad 8?

La deforestación se expande en **todas las direcciones**, incluyendo diagonales. Con conectividad 4, parches diagonalmente continuos se fragmentarían artificialmente.

**Ejemplo:**

```
Conectividad 4 (cruz):          Conectividad 8 (3×3):
  X X                               X X
 X ? X      → 2 zonas             X ? X    → 1 zona
  X X                               X X
```

### Funcionamiento

1. **Pass 1 - Etiquetar:**
   - Recorre píxeles de izquierda→derecha, arriba→abajo
   - Revisa vecinos ya visitados
   - Asigna etiquetas y registra equivalencias

2. **Pass 2 - Resolver equivalencias:**
   - Unifica etiquetas equivalentes usando Union-Find

3. **Filtrado:**
   - Descarta zonas < area_min_km2
   - Marca zonas > area_max_km2

---

## 📈 Próximos Pasos (Post-Zonificación)

1. **Visualización en QGIS**
   ```
   - Cargar zonas_conectividad8.tif
   - Simbología: Valores únicos
   - Superponer con mapa de cambios
   ```

2. **Análisis de Series Temporales**
   ```python
   # Para cada zona, extraer pérdida anual 1985-2024
   # Construir panel: [zona_id, año, perdida_km2]
   ```

3. **Preparar Dataset de Modelado**
   ```python
   # Panel largo para deep learning
   # Train/val/test splits
   # Normalización de features
   ```

---

## 🛠️ Dependencias

```bash
pip install numpy rasterio scipy scikit-image pandas matplotlib
```

---

## 📁 Estructura de Archivos

```
O1/r3/
├── main.py                      # Pipeline completo (detección + zonificación)
├── deteccion_cambios.py         # Módulo de detección de cambios
├── zonificacion.py              # Módulo de zonificación
├── test_zonificacion.py         # Test rápido
├── README.md                    # Este archivo
├── README_ZONIFICACION.md       # Documentación detallada de zonificación
└── _deprecated_reproyeccion.py  # Archivos del enfoque anterior (deprecados)
```

---

## ❓ FAQ

**¿Cuánta memoria RAM necesito?**
- Detección de cambios: ~2-4 GB (procesamiento por tiles)
- Zonificación: ~8-16 GB (carga mapa completo en RAM)

**¿Puedo ajustar los umbrales de área?**
- Sí, modifica `area_min_km2` y `area_max_km2` en `pipeline_zonificacion()`

**¿Qué pasa si el mapa de cambios no existe?**
- El pipeline detecta automáticamente y genera el mapa primero

**¿Cómo se calculan las áreas en EPSG:4326?**
- Aproximación: 1° ≈ 111 km
- Suficiente para latitudes de Perú (-5° a -15°)
- Para precisión métrica, reproyectar a UTM

**¿Por qué no usar el enfoque de features complejos?**
- Requisitos de memoria prohibitivos (81.6 GB)
- Enfoque simplificado es más eficiente y suficiente para zonificación
- Permite enfocarse en modelado predictivo de series temporales

---

## 📝 Notas Técnicas

- **Memoria:** Detección usa tiles; zonificación carga mapa completo
- **Tiempo:** ~10-30 min para Amazonía completa (depende de hardware)
- **Precisión de áreas:** EPSG:4326 introduce ~2-5% de error vs UTM
- **Escalabilidad:** Para rasters >100k×100k, considerar zonificación por tiles

---

## 📚 Referencias

- **Connected Components:** Rosenfeld & Pfaltz (1966)
- **Librerías:** `scipy.ndimage.label`, `skimage.measure.regionprops`
- **Datos:** MapBiomas Perú (1985-2024)

---

**Autor:** Bryan Valdiviezo  
**Proyecto:** Pronóstico de Deforestación - Amazonía Peruana  
**Fase:** R3 - Zonificación Espacial  
**Última actualización:** Abril 2024
