import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

class OrionIITrainer:
    def __init__(self, model_type='random_forest'):
        """
        Inicializar el entrenador de modelos
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError("Model type debe ser 'random_forest' o 'logistic_regression'")
    
    def train(self, X_train, y_train):
        """
        Entrenar el modelo
        """
        print(f"🚀 ENTRENANDO MODELO {self.model_type.upper()}")
        print("="*50)
        
        print(f"📊 Dimensiones de entrenamiento:")
        print(f"   X_train: {X_train.shape}")
        print(f"   y_train: {y_train.shape}")
        
        # Entrenar modelo
        self.model.fit(X_train, y_train)
        
        print("✅ Modelo entrenado exitosamente")
        
        # Obtener importancia de características si es Random Forest
        if self.model_type == 'random_forest' and hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
    
    def evaluate(self, X_test, y_test, label_encoder=None):
        """
        Evaluar el modelo
        """
        print(f"\n📈 EVALUANDO MODELO {self.model_type.upper()}")
        print("="*50)
        
        # Predecir
        y_pred = self.model.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Guardar métricas
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Mostrar resultados
        print(f"🎯 Métricas de evaluación:")
        print(f"   Exactitud (Accuracy): {accuracy:.4f}")
        print(f"   Precisión (Precision): {precision:.4f}")
        print(f"   Sensibilidad (Recall): {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        return self.metrics
    
    def save_model(self, file_path='data/processed/orion11_exoplanet_model.joblib'):
        """
        Guardar modelo entrenado
        """
        if self.model is None:
            raise ValueError("No hay modelo entrenado para guardar")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        # Guardar modelo y metadata
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, file_path)
        print(f"💾 Modelo guardado en: {file_path}")