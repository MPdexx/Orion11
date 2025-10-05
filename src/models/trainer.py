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
        
        Args:
            model_type: 'random_forest' o 'logistic_regression'
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
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
        """
        print(f"🚀 ENTRENANDO MODELO {self.model_type.upper()}")
        print("="*50)
        
        print(f"📊 Dimensiones de entrenamiento:")
        print(f"   X_train: {X_train.shape}")
        print(f"   y_train: {y_train.shape}")
        print(f"   Clases únicas: {np.unique(y_train)}")
        
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
        
        Args:
            X_test: Características de prueba
            y_test: Etiquetas de prueba
            label_encoder: Codificador de etiquetas para nombres de clases
        """
        print(f"\n📈 EVALUANDO MODELO {self.model_type.upper()}")
        print("="*50)
        
        # Predecir
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
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
        
        # Reporte de clasificación detallado
        print(f"\n📋 Reporte de clasificación:")
        if label_encoder:
            class_names = label_encoder.classes_
            print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
        else:
            print(classification_report(y_test, y_pred, zero_division=0))
        
        # Matriz de confusión
        self._plot_confusion_matrix(y_test, y_pred, label_encoder)
        
        # Importancia de características
        if self.feature_importance is not None:
            self._plot_feature_importance()
        
        return self.metrics
    
    def _plot_confusion_matrix(self, y_test, y_pred, label_encoder):
        """Graficar matriz de confusión"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        
        if label_encoder:
            class_names = label_encoder.classes_
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title(f'Matriz de Confusión - {self.model_type.replace("_", " ").title()}')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.show()
    
    def _plot_feature_importance(self, top_n=10):
        """Graficar importancia de características (solo para Random Forest)"""
        if self.feature_importance is None:
            return
        
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Características Más Importantes - Random Forest')
        plt.xlabel('Importancia')
        plt.tight_layout()
        plt.show()
        
        print(f"\n🔍 Top {top_n} características más importantes:")
        for idx, row in top_features.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
    
    def save_model(self, file_path='orionii_exoplanet_model.joblib'):
        """
        Guardar modelo entrenado
        
        Args:
            file_path: Ruta donde guardar el modelo
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
    
    def load_model(self, file_path='orionii_exoplanet_model.joblib'):
        """
        Cargar modelo guardado
        
        Args:
            file_path: Ruta del modelo guardado
        """
        model_data = joblib.load(file_path)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data['feature_importance']
        self.metrics = model_data['metrics']
        
        print(f"📂 Modelo cargado desde: {file_path}")
        print(f"📊 Métricas del modelo cargado: {self.metrics}")
    
    def predict(self, X, label_encoder=None):
        """
        Hacer predicciones con el modelo entrenado
        
        Args:
            X: Datos para predecir
            label_encoder: Codificador para decodificar etiquetas
        
        Returns:
            Predicciones y probabilidades
        """
        if self.model is None:
            raise ValueError("El modelo no está entrenado o cargado")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        if label_encoder:
            predictions = label_encoder.inverse_transform(predictions)
        
        return predictions, probabilities