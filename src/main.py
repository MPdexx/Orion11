import sys
import os

# Añadir el directorio src al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.loader import OrionIIDataLoader
from visualization.showbasic import OrionIIVisualizer

def main():
    """Función principal para ejecutar la demo de OrionII"""
    print("🚀 INICIANDO DEMO DE ORIONII")
    print("="*50)
    
    # Reemplaza con tus rutas reales
    file_paths = [
        'data/raw/cumulative_2025.09.16_12.35.17.csv', 
        'data/raw/q1_q17_dr25_koi_2025.10.04_12.29.09.csv'
    ]
    
    try:
        # 1. Cargar datos
        print("📥 Cargando datos...")
        loader = OrionIIDataLoader()
        raw_data = loader.load_koi_data(file_paths)
        
        # 2. Limpiar datos
        print("🧹 Limpiando datos...")
        clean_data = loader.clean_koi_data()
        
        # 3. Visualizar y analizar
        print("📊 Analizando datos...")
        visualizer = OrionIIVisualizer(clean_data)
        visualizer.show_basic_stats()
        visualizer.create_visualizations()
        
        # 4. Mostrar resultados
        print("\n" + "="*50)
        print("📋 MUESTRA DE DATOS LIMPIOS")
        print("="*50)
        display_columns = [
            'kepoi_name', 'koi_disposition', 'koi_score',
            'koi_period', 'koi_prad', 'koi_teq', 'koi_steff'
        ]
        existing_display = [col for col in display_columns if col in clean_data.columns]
        print(clean_data[existing_display].head(10).to_string(index=False))
        
        # 5. Guardar datos
        output_file = 'data/processed/orion11_koi_data_clean.csv'
        clean_data.to_csv(output_file, index=False)
        print(f"\n💾 Datos limpios guardados en '{output_file}'")
        
        return clean_data
        
    except Exception as e:
        print(f"❌ Error en la demo: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    koi_data = main()