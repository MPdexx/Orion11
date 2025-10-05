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
    print("🚀 ORIONII - SISTEMA DE DETECCIÓN DE EXOPLANETAS")
    print("="*60)
    
    # REEMPLAZA CON TUS RUTAS REALES
    file_paths = ['data/raw/cumulative_2025.09.16_12.35.17.csv', 'data/raw/q1_q17_dr25_koi_2025.10.04_12.29.09.csv']
    
    try:
        # 1. Cargar, limpiar y entrenar
        loader = OrionIIDataLoader()
        raw_data = loader.load_koi_data(file_paths)
        clean_data = loader.clean_koi_data()
        X_train, X_test, y_train, y_test = loader.prepare_ml_data()
        
        # 2. Entrenar modelo
        trainer = OrionIITrainer(model_type='random_forest')
        trainer.train(X_train, y_train)
        trainer.evaluate(X_test, y_test, loader.label_encoder)
        trainer.save_model('data/processed/orionii_best_model.joblib')
        
        # 3. Usar detector para predecir candidatos
        print("\n" + "🔭 MÓDULO DE DETECCIÓN EN TIEMPO REAL")
        print("="*50)
        
        detector = OrionIIDetector('data/processed/orionii_best_model.joblib')
        
        # Ejemplo 1: Predecir un candidato específico
        print("\n🎯 EJEMPLO 1: DETECCIÓN INDIVIDUAL")
        candidate_1 = {
            'kepoi_name': 'K00001.01',
            'koi_period': 10.5,
            'koi_duration': 0.2,
            'koi_depth': 0.001,
            'koi_impact': 0.3,
            'koi_prad': 1.2,
            'koi_teq': 280,
            'koi_insol': 0.9,
            'koi_steff': 5800,
            'koi_srad': 1.0,
            'koi_score': 0.85,
            'koi_kepmag': 12.5,
            'koi_model_snr': 15.0
        }
        
        result_1 = detector.predict_single_planet(candidate_1)
        
        # Ejemplo 2: Predecir múltiples candidatos del dataset
        print("\n🎯 EJEMPLO 2: DETECCIÓN POR LOTES")
        # Tomar algunos candidatos no confirmados del dataset
        unconfirmed_candidates = clean_data[clean_data['koi_disposition'] == 'CANDIDATE'].head(5)
        if len(unconfirmed_candidates) > 0:
            batch_results = detector.predict_batch_planets(unconfirmed_candidates)
        
        # 4. Visualizaciones
        visualizer = OrionIIVisualizer(clean_data)
        visualizer.show_basic_stats()
        visualizer.create_visualizations()
        
        print("\n🎉 SISTEMA ORIONII OPERATIVO")
        print("💡 El modelo puede analizar nuevos candidatos en tiempo real")
        
        return {
            'loader': loader,
            'trainer': trainer, 
            'detector': detector,
            'data': clean_data
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()