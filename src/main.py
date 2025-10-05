import sys
import os

# Configurar el path para importaciones
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_processing.loader import OrionIIDataLoader
from visualization.showbasic import OrionIIVisualizer
from models.trainer import OrionIITrainer
from models.detector import OrionIIDetector

def main():
    """Función principal para ejecutar la demo de OrionII"""
    print("🚀 ORION-11 - SISTEMA DE DETECCIÓN DE EXOPLANETAS")
    print("="*60)
    
    # Rutas de archivos que se van a procesar
    file_paths = ['data/raw/cumulative_2025.09.16_12.35.17.csv', 'data/raw/q1_q17_dr25_koi_2025.10.04_12.29.09.csv']
    
    try:
        # 1. Cargar, limpiar y procesar
        loader = OrionIIDataLoader()
        raw_data = loader.load_koi_data(file_paths)
        clean_data = loader.clean_koi_data()
        X_train, X_test, y_train, y_test = loader.prepare_ml_data()
        
        # 2. Entrenar modelo
        trainer = OrionIITrainer(model_type='random_forest')
        trainer.train(X_train, y_train)
        trainer.evaluate(X_test, y_test, loader.label_encoder)

        model_save_path = 'data/processed/orion11_trained_model.joblib'
        trainer.save_model(model_save_path)
        print(f"💾 Modelo guardado en: {model_save_path}")
        
        # 3. Usar detector para predecir candidatos
        print("\n" + "🔭 MÓDULO DE DETECCIÓN EN TIEMPO REAL")
        print("="*50)
        
        detector = OrionIIDetector('data/processed/orion11_best_model.joblib')
        
        # FASE 2: PREDECIR con tabla nueva
        print("\n🔭 FASE 2: PREDICIENDO NUEVOS CANDIDATOS")
        file_path_predict = 'data/raw/test.csv'
        
        loader_predict = OrionIIDataLoader()
        nuevos_datos = loader_predict.load_koi_data([file_path_predict])
        datos_limpios = loader_predict.clean_koi_data()

        # ⚠️ CARGAR con el MISMO nombre
        detector = OrionIIDetector(model_save_path)  # ← Usar misma ruta

        # Predecir todos los candidatos nuevos
        resultados = detector.predict_batch_planets(datos_limpios)

        # Guardar resultados
        resultados.to_csv('data/processed/predicciones_nuevos_candidatos.csv', index=False)
        print("💾 Predicciones guardadas en 'predicciones_nuevos_candidatos.csv'")
            
        # 4. Visualizaciones
        visualizer = OrionIIVisualizer(clean_data)
        visualizer.show_basic_stats()
        visualizer.create_visualizations()
        
        print("\n🎉 SISTEMA ORION-11 OPERATIVO")

        
        return {
            'loader': loader,
            'trainer': trainer, 
            'detector': detector,
            'data': clean_data,
            'results': resultados
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()