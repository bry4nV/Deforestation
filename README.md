# 🌳 Pronóstico de Deforestación en la Amazonía Peruana

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow.svg)]()

**Proyecto de tesis:** Sistema de pronóstico anual de deforestación en la Amazonía peruana mediante deep learning y zonificación espacial basada en series temporales (1985-2024).

---

## 📋 Descripción General

Este proyecto desarrolla una metodología para **predecir la deforestación anual** en la Amazonía peruana utilizando:

- **Datos históricos:** Mapas anuales de cobertura bosque/no bosque (1985-2024) de MapBiomas
- **Zonificación espacial:** Identificación de áreas dinámicas mediante componentes conectados
- **Series temporales:** Panel zona-año para análisis predictivo
- **Deep Learning:** Modelos de series temporales (LSTM, Transformers, N-BEATS)

**Objetivo:** Predecir la pérdida de cobertura forestal (km²) por zona para el año t+1, proporcionando herramientas para políticas de conservación y monitoreo temprano.

---

## 🎯 Objetivos del Proyecto

### Objetivo General
Desarrollar un sistema de pronóstico de deforestación anual en la Amazonía peruana mediante modelos de deep learning entrenados sobre series temporales de pérdida de cobertura forestal por zona.

### Objetivos Específicos
1. ✅ **[Completado]** Procesar y reclasificar mapas anuales de cobertura forestal (1985-2024)
2. ✅ **[Completado]** Implementar zonificación espacial mediante componentes conectados
3. 🔄 **[En desarrollo]** Extraer series temporales de pérdida por zona
4. ⏳ **[Pendiente]** Entrenar modelos de deep learning para pronóstico
5. ⏳ **[Pendiente]** Evaluar y comparar arquitecturas de modelos
6. ⏳ **[Pendiente]** Desplegar sistema de predicción operacional

---

## 🏗️ Estructura del Proyecto

```
Tesis/
│
├── deforestation-forecast/         # Código fuente principal
│   ├── data/                       # Datos del proyecto (no incluido en repo)
│   │   ├── raw/                    # Datos originales de MapBiomas
│   │   └── interim/                # Datos procesados intermedios
│   │       └── O1/
│   │           ├── mapas-reclas/   # Mapas reclasificados bosque/no bosque
│   │           ├── mapas-cambios/  # Mapas de detección de cambios
│   │           └── zonas/          # Mapas de zonificación y estadísticas
│   │
│   ├── src/                        # Código fuente
│   │   ├── O1/                     # Objetivo 1: Zonificación espacial
│   │   │   ├── config.py           # Configuración general
│   │   │   ├── r1_r2/              # R1/R2: Procesamiento y reclasificación
│   │   │   └── r3/                 # ✅ R3: Detección de cambios y zonificación
│   │   │       ├── main.py         # Pipeline principal
│   │   │       ├── deteccion_cambios.py
│   │   │       ├── zonificacion.py
│   │   │       └── README.md       # Documentación detallada R3
│   │   │
│   │   ├── O2/                     # ⏳ Objetivo 2: Series temporales
│   │   ├── O3/                     # ⏳ Objetivo 3: Modelado predictivo
│   │   └── O4/                     # ⏳ Objetivo 4: Evaluación y despliegue
│   │
│   ├── outputs/                    # Resultados finales (gráficos, reportes)
│   ├── requirements.txt            # Dependencias de Python
│   └── venv/                       # Entorno virtual (no en repo)
│
└── README.md                       # Este archivo
```

---

## 📊 Estado de Implementación

### ✅ Fase 1: Procesamiento de Datos (R1/R2)
**Estado:** Completado

- Descarga y procesamiento de mapas MapBiomas (1985-2024)
- Reclasificación a formato binario bosque/no bosque
- Alineamiento espacial y temporal de rasters
- Validación de integridad de datos

**Outputs:**
- 40 mapas anuales reclasificados (1985-2024)
- Sistema de coordenadas: EPSG:4326
- Resolución: ~30m por píxel

> **📁 Resultados R1/R2:**  
> _(Agregar link a carpeta compartida con mapas reclasificados)_  
> - [ ] Mapas bosque/no bosque por año  
> - [ ] Estadísticas de cobertura  
> - [ ] Mapas de validación  

---

### ✅ Fase 2: Detección de Cambios y Zonificación (R3)
**Estado:** Completado

**Componentes implementados:**

#### 1. Detección de Cambios
- Identificación de píxeles con transiciones bosque ↔ no bosque
- Procesamiento por tiles para eficiencia de memoria (5000×5000 px)
- Generación de mapa binario de cambios (1985-2024)

**Características técnicas:**
- Algoritmo: Comparación temporal píxel a píxel
- Memoria requerida: 2-4 GB RAM
- Tiempo de ejecución: ~5-15 minutos

#### 2. Zonificación Espacial
- Agrupación de píxeles contiguos mediante **componentes conectados**
- Conectividad 8 (incluye diagonales) para capturar patrones reales
- Filtrado por área mínima (50 km²) y máxima (2000 km²)
- Cálculo de métricas: área, centroide, bounding box

**Algoritmo:** Two-Pass Connected Components Labeling  
**Librería:** `scipy.ndimage.label` + `skimage.measure.regionprops`

**Ejecución:**
```bash
cd deforestation-forecast/src
python -m O1.r3.main
```

> **📁 Resultados R3:**  
> _(Agregar links a carpeta compartida con resultados de zonificación)_  
> - [ ] Mapa de cambios 1985-2024  
> - [ ] Mapa de zonas identificadas  
> - [ ] Estadísticas de zonas (CSV)  
> - [ ] Histogramas y visualizaciones  

**Documentación detallada:** [`deforestation-forecast/src/O1/r3/README.md`](deforestation-forecast/src/O1/r3/README.md)

---

### 🔄 Fase 3: Series Temporales por Zona (O2)
**Estado:** En desarrollo

**Objetivos:**
- Extraer serie temporal de pérdida anual por zona (1985-2024)
- Construir panel largo: `[zona_id, año, perdida_km2, ...]`
- Calcular features adicionales (tendencias, estacionalidad)
- Preparar datasets train/validation/test

**Estructura esperada del panel:**
```
zona_id | año  | perdida_km2 | perdida_pct | acumulada_km2 | ...
--------|------|-------------|-------------|---------------|----
   1    | 1985 |    12.34    |    0.45     |    12.34      | ...
   1    | 1986 |    15.67    |    0.58     |    28.01      | ...
```

**Pendiente:**
- [ ] Extracción de series temporales por zona
- [ ] Cálculo de métricas derivadas
- [ ] Análisis exploratorio de datos (EDA)
- [ ] División train/val/test estratificada

> **📁 Resultados O2:**  
> _(Espacio reservado para panel zona-año y análisis)_  
> - [ ] Panel zona-año (CSV/Parquet)  
> - [ ] Gráficos de series temporales  
> - [ ] Análisis de tendencias  

---

### ⏳ Fase 4: Modelado Predictivo (O3)
**Estado:** Pendiente

**Modelos a implementar:**
- LSTM (Long Short-Term Memory)
- Transformers con atención temporal
- N-BEATS (Neural Basis Expansion Analysis)
- TFT (Temporal Fusion Transformers)

**Experimentos planeados:**
1. Baseline: Modelos estadísticos (ARIMA, Prophet)
2. Deep Learning: Arquitecturas modernas
3. Ensemble: Combinación de modelos

**Métricas de evaluación:**
- RMSE, MAE, MAPE
- R² por zona
- Validación espacial y temporal

**Pendiente:**
- [ ] Diseño de arquitecturas
- [ ] Pipeline de entrenamiento
- [ ] Optimización de hiperparámetros
- [ ] Evaluación comparativa

> **📁 Resultados O3:**  
> _(Espacio reservado para modelos y evaluaciones)_  
> - [ ] Pesos de modelos entrenados  
> - [ ] Métricas de evaluación  
> - [ ] Gráficos predicciones vs reales  

---

### ⏳ Fase 5: Despliegue (O4)
**Estado:** Pendiente

**Componentes planeados:**
- [ ] API REST para predicciones
- [ ] Dashboard interactivo
- [ ] Sistema de alertas tempranas
- [ ] Documentación de usuario

> **📁 Resultados O4:**  
> _(Espacio reservado para sistema desplegado)_  
> - [ ] Dashboard de predicciones  
> - [ ] Mapas de riesgo proyectados  

---

## 🚀 Instalación y Uso

### Requisitos Previos
- Python 3.8+
- 8-16 GB RAM
- ~50 GB espacio en disco

### Instalación

```bash
git clone https://github.com/tu-usuario/tesis-deforestacion.git
cd tesis-deforestacion/deforestation-forecast
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Ejecución Pipeline R3

```bash
cd deforestation-forecast/src
python -m O1.r3.main
```

---

## 📦 Dependencias Principales

```
numpy >= 2.2.0
rasterio >= 1.4.0
scipy >= 1.11.0
scikit-image >= 0.22.0
pandas >= 2.3.0
matplotlib >= 3.10.0
```

---

## 🔬 Metodología

### Zonificación Espacial
- **Enfoque:** Componentes conectados sobre mapa de cambios
- **Conectividad:** 8 vecinos (incluye diagonales)
- **Filtros:** Área 50-2000 km²

### Series Temporales (En desarrollo)
- **Panel:** Zona-año (40 años)
- **Target:** Pérdida anual en km²

### Modelado (Pendiente)
- **Arquitecturas:** LSTM, Transformers, N-BEATS
- **Estrategia:** Modelo global para todas las zonas

---

## 🗺️ Datos

**Fuente:** [MapBiomas Perú](https://peru.mapbiomas.org/)  
**Período:** 1985-2024  
**Resolución:** ~30m  
**CRS:** EPSG:4326

---

## 📚 Referencias

- MapBiomas Project. (2024). *Annual Land Use Land Cover Maps of Peru*.
- Rosenfeld & Pfaltz (1966). Sequential operations in digital picture processing.

---

## 👤 Autor

**Bryan Valdiviezo & Edwin Villanueva**  
_[Universidad]_ | Tesis de [Grado]

---

## 📌 Roadmap

- [x] ✅ Procesamiento de datos (R1/R2)
- [x] ✅ Detección de cambios (R3)
- [x] ✅ Zonificación espacial (R3)
- [ ] 🔄 Series temporales (O2)
- [ ] ⏳ Modelado predictivo (O3)
- [ ] ⏳ Evaluación (O3)
- [ ] ⏳ Despliegue (O4)

---

**Estado:** 🟡 En desarrollo  
**Última actualización:** Abril 2024
