"""
Script de prueba rápida para zonificación.

Ejecuta el pipeline de zonificación sobre el mapa de cambios generado.
"""
import os
import sys

# Asegurar que podemos importar desde O1
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from O1.r3.zonificacion import pipeline_zonificacion
from O1.config import MAPAS_CAMBIOS_DIR, ZONAS_DIR


def main():
    """
    Ejecuta el pipeline de zonificación de forma rápida.
    """
    print("\n" + "="*70)
    print(" TEST RÁPIDO - ZONIFICACIÓN")
    print("="*70 + "\n")
    
    # Ruta al mapa de cambios
    mapa_cambios = os.path.join(MAPAS_CAMBIOS_DIR, "mapa_cambios_1985_2024.tif")
    
    # Verificar que existe
    if not os.path.exists(mapa_cambios):
        print(f"[ERROR] No existe el mapa de cambios: {mapa_cambios}")
        print("\n[INFO] Ejecuta primero:")
        print("  cd src")
        print("  python -m O1.r3.main_simple")
        return 1
    
    print(f"[OK] Mapa de cambios encontrado: {mapa_cambios}")
    
    # Ejecutar zonificación
    resumen = pipeline_zonificacion(
        mapa_cambios_path=mapa_cambios,
        output_dir=ZONAS_DIR,
        area_min_km2=50,   # Mínimo 50 km² por zona
        area_max_km2=2000  # Máximo 2000 km² (para advertir de zonas muy grandes)
    )
    
    print("\n" + "="*70)
    print(" RESUMEN FINAL")
    print("="*70)
    for key, value in resumen.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
