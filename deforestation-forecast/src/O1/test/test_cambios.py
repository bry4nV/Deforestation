"""
Test rápido para verificar la detección de cambios en un área pequeña.
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from O1.config import ANIOS, MAPAS_RECLAS_DIR, O1_INTERIM_DIR
from O1.r3.deteccion_cambios import detectar_cambios_tile


def test_deteccion_cambios_pequeno():
    """
    Prueba la detección de cambios en una ventana pequeña (100x100 píxeles).
    """
    print("\n" + "="*70)
    print("TEST DE DETECCIÓN DE CAMBIOS (VENTANA PEQUEÑA)")
    print("="*70 + "\n")
    
    # Seleccionar primeros 5 años para prueba rápida
    anios_test = ANIOS[:5]
    
    print(f"[INFO] Años de prueba: {anios_test}")
    
    # Preparar rutas
    raster_paths = [
        os.path.join(MAPAS_RECLAS_DIR, f"bosque_nobosque_amazonia_{anio}.tif")
        for anio in anios_test
    ]
    
    # Verificar que existe al menos el primero
    if not os.path.exists(raster_paths[0]):
        print(f"[ERROR] No existe: {raster_paths[0]}")
        print("[INFO] Ejecuta primero el pipeline R1/R2")
        return
    
    # Leer una ventana pequeña (100x100) de todos los años
    window = Window(10000, 10000, 100, 100)
    
    print(f"\n[INFO] Leyendo ventana: col={window.col_off}, row={window.row_off}, size=100x100")
    
    tile_stack = np.zeros((len(anios_test), 100, 100), dtype=np.uint8)
    
    for t, path in enumerate(raster_paths):
        if os.path.exists(path):
            with rasterio.open(path) as src:
                tile_stack[t] = src.read(1, window=window)
                print(f"  Año {anios_test[t]}: cargado")
        else:
            print(f"  Año {anios_test[t]}: FALTA")
    
    # Detectar cambios en este tile pequeño
    print("\n[INFO] Detectando cambios...")
    cambios, stats = detectar_cambios_tile(tile_stack)
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("RESULTADOS DEL TEST")
    print("="*70)
    print(f"  Píxeles en ventana:   {100 * 100}")
    print(f"  Píxeles válidos:      {stats['pixeles_validos']}")
    print(f"  Píxeles con cambio:   {stats['pixeles_con_cambio']}")
    
    if stats['pixeles_validos'] > 0:
        pct = stats['pixeles_con_cambio'] / stats['pixeles_validos'] * 100
        print(f"  Porcentaje cambio:    {pct:.2f}%")
    
    print("\n  Distribución de valores en mapa de cambios:")
    valores, conteos = np.unique(cambios, return_counts=True)
    for val, count in zip(valores, conteos):
        if val == 0:
            label = "Sin cambio"
        elif val == 1:
            label = "Con cambio"
        elif val == 255:
            label = "Nodata"
        else:
            label = f"Valor {val}"
        print(f"    {label}: {count} píxeles")
    
    print("="*70 + "\n")
    
    # Guardar resultado de prueba
    CAMBIOS_DIR = os.path.join(O1_INTERIM_DIR, "mapas-cambios")
    os.makedirs(CAMBIOS_DIR, exist_ok=True)
    
    output_test = os.path.join(CAMBIOS_DIR, "test_cambios_100x100.npy")
    np.save(output_test, cambios)
    
    print(f"[OK] Resultado guardado: {output_test}")
    print("[INFO] Puedes inspeccionar con: np.load('{output_test}')")
    print()


if __name__ == "__main__":
    test_deteccion_cambios_pequeno()
