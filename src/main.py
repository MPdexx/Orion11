import sys
import os

# Configurar el path para importaciones
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_processing.loader import OrionIIDataLoader
from visualization.showbasic import OrionIIVisualizer
from models.trainer import OrionIITrainer

def main():
    """Función principal para ejecutar la demo de OrionII"""
    print("🚀 INICIANDO DEMO DE ORIONII - DETECCIÓN DE EXOPLANETAS")
    print("="*60)
    
    # REEMPLAZA CON TUS RUTAS REALES
    file_paths = [
        'data/raw/cumulative_2025.09.16_12.35.17.csv',
        'data/raw/q1_q17_dr25_koi_2025.10.04_12.29.09.csv'
    ]
    
    try:
        # 1. Cargar y limpiar datos
        print("📥 Cargando y limpiando datos...")
        loader = OrionIIDataLoader()
        raw_data = loader.load_koi_data(file_paths)
        clean_data = loader.clean_koi_data()
        
        # 2. Preparar datos para ML
        print("🤖 Preparando datos para Machine Learning...")
        X_train, X_test, y_train, y_test = loader.prepare_ml_data()
        ml_data = loader.get_feature_importance_data()
        
        # 3. Entrenar modelo
        print("\n" + "🔮 ENTRENAMIENTO DE MODELO DE DETECCIÓN")
        print("="*50)
        
        # Probar ambos modelos
        models_to_train = ['random_forest', 'logistic_regression']
        best_model = None
        best_score = 0
        
        for model_type in models_to_train:
            print(f"\n🎯 Entrenando {model_type.replace('_', ' ').title()}...")
            
            trainer = OrionIITrainer(model_type=model_type)
            trainer.train(X_train, y_train)
            metrics = trainer.evaluate(X_test, y_test, ml_data['label_encoder'])
            
            # Guardar el mejor modelo
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_model = trainer
        
        # 4. Guardar el mejor modelo
        print(f"\n🏆 MEJOR MODELO: {best_model.model_type.replace('_', ' ').title()}")
        print(f"   Exactitud: {best_score:.4f}")
        
        best_model.save_model('models/orionii_best_model.joblib')
        
        # 5. Visualizar datos
        print("\n📊 VISUALIZACIÓN DE DATOS")
        print("="*50)
        visualizer = OrionIIVisualizer(clean_data)
        visualizer.show_basic_stats()
        visualizer.create_visualizations()
        
        # 6. Mostrar resultados finales
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETADA EXITOSAMENTE")
        print("="*60)
        print(f"📈 Resultados del modelo:")
        print(f"   🏆 Modelo: {best_model.model_type.replace('_', ' ').title()}")
        print(f"   🎯 Exactitud: {best_model.metrics['accuracy']:.4f}")
        print(f"   📊 Precisión: {best_model.metrics['precision']:.4f}")
        print(f"   🔍 Sensibilidad: {best_model.metrics['recall']:.4f}")
        print(f"   ⚖️  F1-Score: {best_model.metrics['f1_score']:.4f}")
        print(f"💾 Modelo guardado en: models/orionii_best_model.joblib")
        
        return {
            'clean_data': clean_data,
            'best_model': best_model,
            'ml_data': ml_data
        }
        
    except Exception as e:
        print(f"❌ Error en la demo: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()