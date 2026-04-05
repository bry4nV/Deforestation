# R3 - Detección de Cambios y Zonificación

## 📋 Cambio de Enfoque

### ✅ **Enfoque nuevo:**
- **Detección simple**: ¿El píxel cambió (bosque ↔ no bosque) en algún momento?
- Procesamiento por tiles para manejar memoria
- Generación de mapa binario de cambios

---

## 🎯 Objetivo

**Identificar qué píxeles han experimentado ALGÚN cambio entre bosque y no bosque durante la serie temporal 1985-2024.**

Este mapa de cambios será la base para:
1. Identificar zonas de cambio mediante clustering espacial
2. Analizar patrones espaciales de deforestación
3. Definir unidades de análisis para modelado predictivo

---

## 🏗️ Módulos Activos

```
r3/
├── deteccion_cambios.py     ✅ ACTIVO - Detección de píxeles con cambios
├── main.py            ✅ ACTIVO - Pipeline
```

---

## 🚀 Ejecución

### **Pipeline completo:**

```bash
cd d:\VisualCode\Tesis\deforestation-forecast
python -m O1.r3.main
```

Esto ejecutará:
1. Verificación de archivos de entrada
2. Detección de cambios por tiles
3. Generación de mapa de cambios
4. Exportación de estadísticas

---

## 📊 Lógica de Detección

### **¿Qué se considera "cambio"?**

Un píxel tiene cambio (=1) si en ALGÚN momento de la serie temporal hubo una transición entre estados:

- **0 → 1** (no bosque → bosque) - Recuperación/reforestación
- **1 → 0** (bosque → no bosque) - Deforestación

### **Ejemplos:**

```python
# Serie temporal de un píxel (40 años)
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# → Resultado: 0 (sin cambio, siempre bosque)

[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# → Resultado: 0 (sin cambio, siempre no-bosque)

[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# → Resultado: 1 (CAMBIÓ: transición en año 11)

[1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# → Resultado: 1 (CAMBIÓ: múltiples transiciones)
```

---

## 📂 Outputs Generados

```
data/interim/O1/mapas-cambios/
├── mapa_cambios_1985_2024.tif    # Mapa binario de cambios
│                                 # 1 = cambió, 0 = sin cambio, 255 = nodata
└── estadisticas_cambios.txt      # Estadísticas descriptivas
```

### **Estructura del mapa de cambios:**

| Valor | Significado |
|-------|-------------|
| **1** | Píxel que cambió (bosque ↔ no bosque) en algún momento |
| **0** | Píxel estable (sin cambios en toda la serie) |
| **255** | Nodata (píxel inválido o fuera de área de estudio) |

---

## 🔧 Procesamiento por Tiles

### **¿Por qué tiles?**

**Problema:** 40 años × 54,044 filas × 40,527 columnas = **81.6 GB** en memoria

**Solución:** Procesar en bloques espaciales de 5000×5000 píxeles:
- Cada tile requiere solo ~500 MB
- Se procesa secuencialmente
- Resultados se van escribiendo al mapa global

### **Configuración:**

```python
tile_size = 5000  # Default: balance memoria/velocidad

# Si tienes más RAM:
tile_size = 8000  # Más rápido, más memoria

# Si tienes menos RAM:
tile_size = 3000  # Más lento, menos memoria
```

---

## 📈 Ejemplo de Salida

```
======================================================================
DETECCIÓN DE CAMBIOS BOSQUE ↔ NO BOSQUE
======================================================================

[INFO] Dimensiones: 40527 x 54044 píxeles
[INFO] Serie temporal: 40 años
[INFO] Memoria requerida (completo): 81.60 GB
[INFO] Procesando en 9 x 11 = 99 tiles
[INFO] Tamaño de tile: 5000 x 5000 píxeles

  Tile 1/99: row=0, col=0, size=5000x5000
  Tile 2/99: row=0, col=5000, size=5000x5000
  ...
  Tile 99/99: row=50000, col=40000, size=527x4044

======================================================================
ESTADÍSTICAS DE CAMBIOS
======================================================================
  Píxeles válidos:      1,845,234,567
  Píxeles con cambio:   245,123,456 (13.28%)
  Píxeles sin cambio:   1,600,111,111 (86.72%)
======================================================================
```

---

## 🎨 Visualización

### **En QGIS:**

1. Abrir `mapa_cambios_1985_2024.tif`
2. Configurar simbología:
   - **0** (sin cambio): Transparente o gris claro
   - **1** (con cambio): Rojo intenso
   - **255** (nodata): Transparente

3. Superponer con límites administrativos o ríos

---

## 🔜 Próximos Pasos

### **Fase siguiente: Zonificación basada en cambios**

1. **Clustering espacial** de píxeles con cambio=1
   - Usar DBSCAN o componentes conexas
   - Identificar grupos contiguos de píxeles con cambio

2. **Definir zonas de análisis**
   - Cada cluster = una zona potencial
   - Filtrar por tamaño mínimo
   - Etiquetar por características

3. **Análisis temporal por zona**
   - Calcular serie temporal de pérdida para cada zona
   - Construir panel zona-año
   - Preparar para modelado

---

### **Optimizaciones aplicadas:**

- ✅ Procesamiento por tiles
- ✅ Lectura selectiva con ventanas de rasterio
- ✅ Liberación explícita de memoria (`del` después de cada tile)
- ✅ Compresión LZW en output

---

## 🐛 Troubleshooting

### **Error: "Archivos faltantes"**

```
[ERROR] Archivos faltantes:
  - d:\...\bosque_nobosque_amazonia_1985.tif
```

**Solución:** Ejecuta primero el pipeline R1/R2:
```bash
python -m O1.r1_r2.main
```

### **Error: "Unable to allocate X GB"**

**Solución:** Reduce `tile_size`:
```python
tile_size=3000  # En lugar de 5000
```

### **Proceso muy lento**

**Solución:** Aumenta `tile_size` (si tienes RAM):
```python
tile_size=8000  # En lugar de 5000
```

**Versión:** 2.0 (Simplificada)  
**Fecha:** Abril 2026  
**Estado:** ✅ Activo