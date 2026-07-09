# 🌳 Pronóstico de Deforestación en la Amazonía Peruana

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow.svg)]()

**Proyecto de tesis:** Aplicación de modelos de aprendizaje profundo para el pronóstico anual de deforestación en la Amazonía peruana a partir de series temporales de cobertura boscosa y variables territoriales.

---

## 📋 Descripción General

Este proyecto desarrolla una metodología para **pronosticar la deforestación anual** en la Amazonía peruana utilizando:

- **Datos históricos:** Mapas anuales de cobertura bosque/no bosque (1985-2024) de MapBiomas
- **Zonificación espacial:** Identificación de distritos amazónicos con dinámica relevante de cambio de bosque
- **Series temporales:** Panel distrito-año para análisis predictivo
- **Variables locales:** Integración de factores agropecuarios, infraestructura vial, hidrografía, áreas naturales protegidas y topografía
- **Deep Learning:** Modelos de series temporales y comparación entre enfoques base y multivariables
- **Capacidad de generalización:** Evaluación en distritos no utilizados durante el entrenamiento

**Objetivo:** Estimar la cobertura boscosa futura y la deforestación anual esperada, proporcionando evidencia reproducible para apoyar el monitoreo temprano y la toma de decisiones territoriales.

---

## 🎯 Objetivos del Proyecto

### Objetivo General
Desarrollar un sistema de pronóstico de deforestación anual en la Amazonía peruana mediante modelos de aprendizaje profundo entrenados sobre series temporales distritales de cobertura forestal.

### Objetivos Específicos
1. ✅ **[Completado]** Identificar y delimitar distritos amazónicos de interés a partir de MapBiomas Perú y construir series históricas anuales de cobertura boscosa.
2. ✅ **[Implementado]** Diseñar y evaluar modelos base de pronóstico temporal para estimar la cobertura boscosa futura.
3. ✅ **[Implementado]** Integrar variables locales al modelo de pronóstico y evaluar su aporte frente al enfoque base.
4. ✅ **[Implementado]** Evaluar la capacidad de generalización espacial de los pronósticos en distritos no utilizados durante el entrenamiento.

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
│   │   ├── O2/                     # Objetivo 2: Modelos base de pronóstico temporal
│   │   ├── O3/                     # Objetivo 3: Variables locales y modelos multivariables
│   │   └── O4/                     # Objetivo 4: Evaluación de capacidad de generalización
│   │
│   ├── outputs/                    # Artefactos consolidados de salida
│   │   ├── figures/                # Figuras y gráficos generados
│   │   ├── logs/                   # Logs consolidados de ejecución
│   │   └── debug/                  # Salidas auxiliares de depuración
│   │
│   ├── scripts/                    # Scripts auxiliares no centrales
│   │   └── exploracion/            # Análisis exploratorios puntuales
│   │
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
>   → Incluido en documento de tesis

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
**Estado:** ✅ Completado

Delimitación de zonas espaciales con dinámica de cambio de cobertura forestal y extracción de sus series temporales de pérdida anual (1986-2024).

**Metodología:**
- Detección de cambios bosque ↔ no bosque (variación bidireccional)
- Zonificación por componentes conectados (conectividad 8)
- Filtrado por área mínima (1000 píxeles)
- Extracción de series temporales de pérdida (bosque → no bosque) por zona

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
> - [x] Mapa raster de zonas de cambio (.tif)
> - [x] CSV con estadísticas de las series históricas de las zonas (área, centroide, etc.)
> - [x] **Panel zona-año con pérdida anual de bosque (1986-2024)**
> - [x] **Estadísticas de series temporales por zona**
> - [x] **Visualizaciones de series temporales**
> - [ ] Documento de criterios de delimitación  

**Ubicación:** `deforestation-forecast/data/interim/O1/zonas/`  
**Documentación:** [`deforestation-forecast/src/O1/r3/README.md`](deforestation-forecast/src/O1/r3/README.md)

---

### 📌 Objetivo 2 (O2): Modelos base de pronóstico temporal
**Estado:** ✅ Implementado

Entrenamiento y evaluación de modelos base a partir de la serie histórica de cobertura boscosa. Incluye modelos de referencia y modelos de aprendizaje profundo.

**Código:** [`deforestation-forecast/src/O2/r4_r5/`](deforestation-forecast/src/O2/r4_r5/)

**Ejecución:**
```bash
cd deforestation-forecast/src
python -m O2.r4_r5.main
```

> **📂 Entregables:**
> - [x] Resultados por modelo: persistencia, ARIMA, MLP, LSTM y CNN
> - [x] Métricas globales, distritales y departamentales
> - [x] Comparación de modelos y selección de configuraciones finales

**Ubicación:** `deforestation-forecast/data/interim/O2/modelos/`

---

### 📌 Objetivo 3 (O3): Variables locales y modelos multivariables
**Estado:** ✅ Implementado

Construcción de variables territoriales locales e integración con el panel de cobertura boscosa. Este objetivo permite comparar el enfoque base de O2 con modelos multivariables.

**Variables integradas:**
- Cobertura agropecuaria
- Áreas naturales protegidas
- Densidad de carreteras
- Densidad de ríos
- Elevación media
- Pendiente media

**Código:** [`deforestation-forecast/src/O3/`](deforestation-forecast/src/O3/)

**Ejecución:**
```bash
cd deforestation-forecast/src
python -m O3.r8_r9_r10.main
python -m O3.eda_panel
python -m O3.r11.main
```

> **📂 Entregables:**
> - [x] Variables locales por distrito
> - [x] Panel integrado distrito-año
> - [x] EDA metodológico de variables predictoras
> - [x] Modelos multivariables MLP, LSTM y CNN
> - [x] Comparación O2 base vs. O3 extendido

**Ubicación:** `deforestation-forecast/data/interim/O3/`

---

### 📌 Objetivo 4 (O4): Evaluación de capacidad de generalización
**Estado:** ✅ Implementado

Evaluación del desempeño del modelo seleccionado en distritos no utilizados durante el entrenamiento, con análisis de error por distrito, departamento y factores territoriales.

**Código:** [`deforestation-forecast/src/O4/r12_r13_r14/`](deforestation-forecast/src/O4/r12_r13_r14/)

**Ejecución:**
```bash
cd deforestation-forecast/src
python -m O4.r12_r13_r14.main
```

> **📂 Entregables:**
> - [x] Dataset de generalización espacial
> - [x] Pronósticos para distritos fuera de muestra
> - [x] Informe de generalización
> - [x] Tablas y figuras finales de evaluación

**Ubicación:** `deforestation-forecast/data/interim/O4/`

---

## 🚀 Instalación y Uso

### Requisitos Previos
- Python 3.10+
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

### Dependencias Principales

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

**Fuente principal:** [MapBiomas Perú](https://peru.mapbiomas.org/)  
**Período:** 1985-2024  
**Resolución:** ~30m  
**Unidad de análisis:** distrito amazónico por año  
**CRS:** EPSG:4326

---

## 🗂️ Organización de Artefactos

El proyecto conserva una estructura cercana a la organización original por objetivos (`O1`, `O2`, `O3`, `O4`). Los primeros ajustes de ordenamiento consolidan archivos auxiliares sin cambiar los módulos principales:

- `data/raw/`: fuentes originales.
- `data/interim/`: productos intermedios y resultados por objetivo.
- `outputs/logs/`: logs consolidados de ejecución, conservados como evidencia de corridas.
- `outputs/figures/`: figuras auxiliares y gráficos generados.
- `outputs/debug/`: salidas temporales de depuración.
- `scripts/exploracion/`: scripts exploratorios que no forman parte del pipeline principal.

Los modelos, predicciones binarias y logs no se ignoran por defecto en Git en esta etapa. La decisión queda documentada en `.gitignore` como comentario para evaluarla más adelante según peso, trazabilidad y reproducibilidad.

---

## 👤 Autor

**Bryan Valdiviezo**  
Pontificia Universidad Católica del Perú | Tesis de pregrado en Ingeniería Informática

---

**Estado:** 🟡 En desarrollo  
**Última actualización:** Julio 2026
