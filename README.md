# 🌳 Pronóstico de Deforestación en la Amazonía Peruana

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow.svg)]()

**Proyecto de tesis:** Aplicación de deep learning para el pronóstico anual de zonas de riesgo de deforestación basada en series temporales (1985-2024).

---

## 📋 Descripción General

Este proyecto desarrolla una metodología para **predecir la deforestación anual** en la Amazonía peruana utilizando:

- **Datos históricos:** Mapas anuales de cobertura bosque/no bosque (1985-2024) de MapBiomas
- **Zonificación espacial:** Identificación de áreas dinámicas de cambio de bosque
- **Series temporales:** Panel zona-año para análisis predictivo
- **Variables locales:** Integración de factores socioeconómicos, demográficos, infraestructura, etc.
- **Deep Learning:** Modelos de series temporales (LSTM, ...)
- **Capacidad de generalización:** Evaluación en distintas zonas de estudio

**Objetivo:** Predecir la pérdida de cobertura forestal (km²) por zona para el año t+1, proporcionando herramientas para políticas de conservación y monitoreo temprano.

---

## 🎯 Objetivos del Proyecto

### Objetivo General
Desarrollar un sistema de pronóstico de deforestación anual en la Amazonía peruana mediante modelos de deep learning entrenados sobre series temporales de pérdida de cobertura forestal por zona.

### Objetivos Específicos
1. 🔄 **[En desarrollo]** Identificar y delimitar zonas de estudio potenciales a partir de MapBiomas Perú y construir series históricas anuales de pérdida de bosque por zona.
2. ⏳ **[Pendiente]** Diseñar y evaluar un modelo de pronóstico a partir de las series históricas de deforestación por zona, para estimar la pérdida de bosque en km² en horizonte anual.
3. ⏳ **[Pendiente]** Integrar un conjunto de variables locales en el modelo de pronóstico y evaluar su aporte en las estimaciones anuales por zona.
4. ⏳ **[Pendiente]** Evaluar la capacidad de generalización espacial de los pronósticos de deforestación en distintas zonas de estudio.

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
│   │   ├── O1/                     # Objetivo 1: Identificación de zonas de estudio de deforestación
│   │   │   ├── config.py           # Configuración general
│   │   │   ├── r1_r2/              # R1/R2: Procesamiento y reclasificación
│   │   │   └── r3/                 # R3: Detección de cambios y zonificación espacial
│   │   │
│   │   ├── O2/                     # ⏳ Objetivo 2: Modelo pronóstico temporal de pérdida anual de bosque
│   │   ├── O3/                     # ⏳ Objetivo 3: Modelado pronóstico temporal extendido con variables locales
│   │   └── O4/                     # ⏳ Objetivo 4: Evaluación de capacidad de generalización
│   │
│   ├── outputs/                    # Resultados finales (gráficos, reportes)
│   ├── requirements.txt            # Dependencias de Python
│   └── venv/                       # Entorno virtual (no en repo)
│
└── README.md                       # Este archivo
```

---

## 📊 Estado de Implementación

### ⏳ Objetivo 1 (O1): Identificar zonas de estudio potenciales de deforestación

---

### 📌 R1: Pipeline de Reprocesamiento
**Estado:** ✅ Completado

Pipeline automatizado para procesar mapas MapBiomas Perú (1985-2024): descarga, validación espacial y preparación para reclasificación.

**Código:** [`deforestation-forecast/src/O1/r1_r2/`](deforestation-forecast/src/O1/r1_r2/)

**Ejecución:**
```bash
cd deforestation-forecast/src
python -m O1.r1_r2.main
```

> **📂 Entregables:**  
> - [ ] Código fuente (GitHub)
> - [ ] Diagrama metodológico
>   Incluido en el documento de tesis

---

### 📌 R2: Mapas Bosque/No Bosque
**Estado:** ✅ Completado

40 mapas anuales reclasificados (1985-2024) en formato binario.

**Criterio de reclasificación:**
- Bosque (1): Formación Forestal, Manglar, Bosque Inundable
- No Bosque (0): Resto de clases
- NoData (255)

**Especificaciones:** GeoTIFF, EPSG:4326, ~30m/píxel

> **📂 Entregables:**  
> - [ ] Mapas raster bosque/no bosque (40 años, 1985-2024)
>   → [Descargar de Google Drive](https://drive.google.com/drive/folders/1vsw7WqRPHYCx2Khfrn27XH1i-QYw6Go6?usp=sharing)
>
> - [ ] Documento de criterios de reclasificación
>   → Incluido en documento de tesis

**Ubicación:** `deforestation-forecast/data/interim/O1/mapas-reclasificados/`

---

### 📌 R3: Zonas de Estudio y Series Históricas
**Estado:** 🔄 En desarrollo

Delimitación de zonas espaciales con dinámica de deforestación y extracción de sus series temporales de pérdida anual (1985-2024).

**Metodología:**
- Detección de cambios bosque ↔ no bosque
- Zonificación por componentes conectados (conectividad 8)
- Filtrado por área: 
- Extracción de series temporales por zona

**Ejecución:**
```bash
cd deforestation-forecast/src
python -m O1.r3.main
```

**Estructura del panel zona-año:**
```
zona_id | año | pixeles_perdida | perdida_km2
```

> **📂 Entregables:**
> - [ ] Mapa raster de zonas (.tif)
> - [ ] CSV con estadísticas de las series históricas de las zonas (área, centroide, etc.)
> - [ ] Documento de criterios de delimitación  

**Ubicación:** `deforestation-forecast/data/interim/O1/zonas/`  
**Documentación:** [`deforestation-forecast/src/O1/r3/README.md`](deforestation-forecast/src/O1/r3/README.md)

---

### ⏳ Objetivo 2 (O2): Modelo pronóstico temporal de pérdida anual de bosque
**Estado:** Pendiente

---

### ⏳ Objetivo 3 (O3): Modelado pronóstico temporal extendido con variables locales
**Estado:** Pendiente

---

### ⏳ Objetivo 4 (O4): Evaluación de capacidad de generalización
**Estado:** Pendiente

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

## 🗺️ Datos

**Fuente:** [MapBiomas Perú](https://peru.mapbiomas.org/)  
**Período:** 1985-2024  
**Resolución:** ~30m  
**CRS:** EPSG:4326

---

## 👤 Autor

**Bryan Valdiviezo**  
Pontificia Universidad Católica del Perú | Tesis de pregrado en Ingeniería Informática

---

**Estado:** 🟡 En desarrollo  
**Última actualización:** Abril 2024
