import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class OrionIIDetector:
    def __init__(self, model_path='models/orionii_best_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.model_type = None
        self.metrics = None
        self.feature_importance = None
        self.loaded_model = False
        self.expected_features = None  # ← NUEVO: Guardar features esperados
        
        self.load_model()
    
    def load_model(self):
        """Cargar modelo entrenado"""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.metrics = model_data.get('metrics', {})
            self.feature_importance = model_data.get('feature_importance', None)
            
            # Obtener las características esperadas del modelo
            if hasattr(self.model, 'feature_names_in_'):
                self.expected_features = list(self.model.feature_names_in_)
            else:
                # Si no están disponibles, usar lista por defecto
                self.expected_features = [
                    'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
                    'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff', 'koi_srad',
                    'koi_score', 'koi_kepmag', 'koi_model_snr',
                    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
                ]
            
            self.loaded_model = True
            print(f"✅ Modelo {self.model_type} cargado exitosamente")
            print(f"📊 Características esperadas: {len(self.expected_features)} features")
            print(f"   {self.expected_features}")
        except Exception as e:
            print(f"❌ Error cargando el modelo: {e}")
            self.loaded_model = False
    
    def predict_single_planet(self, planet_features):
        """
        Predecir si un único candidato es un exoplaneta real
        """
        if not self.loaded_model:
            raise ValueError("Modelo no cargado. No se pueden hacer predicciones.")
        
        # Obtener nombre del candidato si está disponible
        candidate_name = planet_features.get('kepoi_name', planet_features.get('kepler_name', 'Candidato Desconocido'))
        
        print(f"🔭 ANALIZANDO CANDIDATO: {candidate_name}")
        print("="*50)
        
        # Preparar características para predicción
        features_df = self._prepare_features(planet_features)
        
        # Hacer predicción
        try:
            prediction = self.model.predict(features_df)[0]
            probabilities = self.model.predict_proba(features_df)[0]
            
            # Interpretar resultados
            result = self._interpret_prediction(prediction, probabilities, planet_features, candidate_name)
            return result
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            # Fallback: predicción básica basada en koi_score
            return self._fallback_prediction(planet_features, candidate_name)
        
    def _prepare_features(self, planet_features):
        """
        Preparar características para que coincidan exactamente con el entrenamiento
        """
        features_dict = {}
        
        # Para cada característica esperada, usar el valor proporcionado o un valor por defecto
        for feature in self.expected_features:
            if feature in planet_features:
                features_dict[feature] = planet_features[feature]
            else:
                # Valor por defecto para características faltantes
                if feature.startswith('koi_fpflag'):
                    features_dict[feature] = 0  # Para flags, asumir 0 (no activado)
                elif feature in ['koi_period', 'koi_duration', 'koi_depth', 'koi_impact']:
                    features_dict[feature] = 1.0  # Valores típicos de tránsito
                elif feature in ['koi_prad', 'koi_teq', 'koi_insol']:
                    features_dict[feature] = 1.0  # Valores planetarios típicos
                elif feature in ['koi_steff', 'koi_srad']:
                    features_dict[feature] = 5000.0  # Valores estelares típicos
                elif feature == 'koi_score':
                    features_dict[feature] = 0.5  # Score medio
                elif feature == 'koi_kepmag':
                    features_dict[feature] = 15.0  # Magnitud típica
                elif feature == 'koi_model_snr':
                    features_dict[feature] = 10.0  # SNR medio
                else:
                    features_dict[feature] = 0.0  # Valor por defecto genérico
        
        return pd.DataFrame([features_dict])
    
    def _preprocess_features(self, features_df):
        """
        Preprocesar características para que coincidan con el entrenamiento
        """
        # Asegurar que tenemos todas las características en el orden correcto
        processed_df = features_df[self.expected_features].copy()
        
        # Llenar cualquier valor NaN restante
        processed_df = processed_df.fillna(processed_df.median())
        
        print(f"📋 Características procesadas: {list(processed_df.columns)}")
        print(f"📊 Valores: {processed_df.iloc[0].to_dict()}")
        
        return processed_df
    
    def _interpret_prediction(self, prediction, probabilities, original_features, candidate_name):
        """
        Interpretar y formatear los resultados de la predicción
        """
        # CLARIFICACIÓN DE CLASES:
        # 0 = CANDIDATE, 1 = CONFIRMED, 2 = FALSE POSITIVE
        
        if prediction == 1:
            prediction_label = "EXOPLANETA CONFIRMADO"
            # Probabilidad de ser CONFIRMED (clase 1)
            exoplanet_prob = probabilities[1]
        elif prediction == 2:
            prediction_label = "FALSO POSITIVO" 
            # Probabilidad de ser FALSE POSITIVE (clase 2)
            exoplanet_prob = probabilities[2]
        else:  # prediction == 0
            prediction_label = "CANDIDATO"
            # Probabilidad de ser CANDIDATE (clase 0)
            exoplanet_prob = probabilities[0]
        
        # 🎯 DEFINICIÓN CLARA: "Probabilidad de ser exoplaneta" = Probabilidad de clase CONFIRMED
        confirmed_probability = probabilities[1]
        
        result = {
            'candidato': candidate_name,
            'prediccion': prediction,
            'etiqueta': prediction_label,
            'probabilidad_exoplaneta': confirmed_probability,  # ← SIEMPRE probabilidad de CONFIRMED
            'probabilidad_clase_predicha': exoplanet_prob,    # ← Probabilidad de la clase que predijo
            'confianza': max(probabilities),
            'todas_probabilidades': {
                'CANDIDATO': probabilities[0],
                'CONFIRMED': probabilities[1], 
                'FALSE_POSITIVE': probabilities[2]
            },
            'caracteristicas_analizadas': original_features
        }
        
        print(f"🎯 RESULTADO PARA {candidate_name}:")
        print(f"   Predicción: {prediction_label}")
        print(f"   Probabilidad de ser exoplaneta CONFIRMADO: {confirmed_probability:.4f}")
        print(f"   Probabilidad de la clase predicha: {exoplanet_prob:.4f}")
        print(f"   Nivel de confianza: {'ALTO' if result['confianza'] >= 0.8 else 'MEDIO' if result['confianza'] >= 0.6 else 'BAJO'}")
        
        # Mostrar todas las probabilidades para debugging
        print(f"\n   📊 DISTRIBUCIÓN DE PROBABILIDADES:")
        print(f"   🟡 CANDIDATO: {probabilities[0]:.4f}")
        print(f"   🟢 CONFIRMED: {probabilities[1]:.4f}") 
        print(f"   🔴 FALSE POSITIVE: {probabilities[2]:.4f}")
        
        # Análisis de características clave
        self._analyze_characteristics(original_features, confirmed_probability, candidate_name)
        
        return result
    

    def _analyze_characteristics(self, features, exoplanet_prob, candidate_name):
        """
        Analizar características específicas del candidato
        """
        print(f"\n🔍 ANÁLISIS DETALLADO DE {candidate_name}:")
        
        if 'koi_score' in features:
            score = features['koi_score']
            print(f"   📊 KOI Score Original: {score:.3f} {'(ALTO)' if score > 0.7 else '(MEDIO)' if score > 0.3 else '(BAJO)'}")
        
        # Análisis de parámetros físicos
        if 'koi_period' in features:
            period = features['koi_period']
            period_analysis = " (cercano)" if period < 20 else " (lejano)" if period > 100 else " (medio)"
            print(f"   ⏱️  Período orbital: {period:.1f} días{period_analysis}")
        
        if 'koi_prad' in features:
            prad = features['koi_prad']
            size_analysis = " (Tierra)" if 0.8 <= prad <= 1.5 else " (mini-Neptuno)" if prad <= 4 else " (gigante)"
            print(f"   📏 Radio planetario: {prad:.1f} radios terrestres{size_analysis}")
        
        if 'koi_teq' in features:
            teq = features['koi_teq']
            temp_analysis = " (habitable 🌍)" if 200 <= teq <= 330 else " (frío)" if teq < 200 else " (caliente)"
            print(f"   🌡️  Temperatura: {teq:.0f} K{temp_analysis}")
        
        # Comparar KOI Score vs Nuestro Modelo
        if 'koi_score' in features:
            koi_score = features['koi_score']
            our_score = exoplanet_prob
            difference = our_score - koi_score
            
            print(f"\n   📊 COMPARACIÓN DE CONFIANZA:")
            print(f"   🤖 OrionII: {our_score:.3f}")
            print(f"   🛰️  KOI: {koi_score:.3f}")
            
            if difference > 0.2:
                print(f"   ✅ Nuestro modelo es MÁS optimista que KOI")
            elif difference < -0.2:
                print(f"   ⚠️  Nuestro modelo es MÁS conservador que KOI")
            else:
                print(f"   📍 Ambos modelos tienen confianza similar")
        
        # Verificar flags de falsos positivos
        fp_flags = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
        active_flags = []
        for flag in fp_flags:
            if flag in features and features[flag] == 1:
                active_flags.append(flag)
        
        if active_flags:
            print(f"   ⚠️  Flags de falso positivo activos: {active_flags}")
        else:
            print(f"   ✅ Ningún flag de falso positivo activo")
        
        # Recomendación basada en probabilidad
        print(f"\n   💡 RECOMENDACIÓN PARA {candidate_name}:")
        if exoplanet_prob >= 0.8:
            print(f"   🎯 ALTA PRIORIDAD: Fuertes indicios de exoplaneta real")
        elif exoplanet_prob >= 0.6:
            print(f"   🔍 MEDIA PRIORIDAD: Posible exoplaneta, requiere más observación")
        else:
            print(f"   📉 BAJA PRIORIDAD: Probable falso positivo")
        # ... (el resto de los métodos permanece igual)

    def predict_batch_planets(self, planets_dataframe):
        """
        Predecir múltiples candidatos a exoplanetas
        """
        if not self.loaded_model:
            raise ValueError("Modelo no cargado. No se pueden hacer predicciones.")
        
        print(f"🔭 ANALIZANDO {len(planets_dataframe)} CANDIDATOS A EXOPLANETAS")
        print("="*50)
        
        try:
            # Preparar características para todos los candidatos
            processed_features_list = []
            
            for idx, row in planets_dataframe.iterrows():
                # Preparar características para cada fila
                features_dict = {}
                for feature in self.expected_features:
                    if feature in row:
                        features_dict[feature] = row[feature]
                    else:
                        features_dict[feature] = self._get_default_value(feature)
                
                features_df = pd.DataFrame([features_dict])[self.expected_features]
                processed_features_list.append(features_df)
            
            # Combinar todas las características
            all_features = pd.concat(processed_features_list, ignore_index=True)
            
            # Llenar valores NaN de manera segura
            all_features = all_features.apply(pd.to_numeric, errors='coerce')
            all_features = all_features.fillna(all_features.median())
            
            print(f"📋 Procesados {len(all_features)} candidatos con {len(all_features.columns)} características")
            
            # Hacer predicciones
            predictions = self.model.predict(all_features)
            probabilities = self.model.predict_proba(all_features)
            
            # Añadir predicciones al DataFrame original
            results_df = planets_dataframe.copy()
            results_df['prediccion'] = predictions
            results_df['probabilidad_exoplaneta'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
            results_df['confianza'] = np.max(probabilities, axis=1)
            
            # Mapear predicciones a etiquetas
            results_df['etiqueta_prediccion'] = results_df['prediccion'].map(
                {1: 'EXOPLANETA CONFIRMADO', 0: 'FALSO POSITIVO', 2: 'CANDIDATO'}
            )
            
            # Clasificar confianza
            results_df['nivel_confianza'] = results_df['confianza'].apply(
                lambda x: 'ALTA' if x >= 0.8 else 'MEDIA' if x >= 0.6 else 'BAJA'
            )
            
            # Mostrar resumen
            self._show_batch_summary(results_df)
            
            return results_df
            
        except Exception as e:
            print(f"❌ Error en predicción por lotes: {e}")
            import traceback
            traceback.print_exc()
            return None
        

    def _show_batch_summary(self, results_df):
        """
        Mostrar resumen de predicciones por lote - MÉTODO PRIVADO
        """
        print(f"\n📈 RESUMEN DE DETECCIÓN POR LOTES")
        print("="*50)
        
        total = len(results_df)
        
        # Contar por tipo de predicción (basado en tu encoding: 0=CANDIDATE, 1=CONFIRMED, 2=FALSE POSITIVE)
        confirmed = len(results_df[results_df['prediccion'] == 1])
        false_positives = len(results_df[results_df['prediccion'] == 2])
        candidates = len(results_df[results_df['prediccion'] == 0])
        
        print(f"   Total de candidatos analizados: {total}")
        print(f"   🪐 Exoplanetas confirmados: {confirmed} ({confirmed/total*100:.1f}%)")
        print(f"   ⚠️  Candidatos pendientes: {candidates} ({candidates/total*100:.1f}%)")
        print(f"   ❌ Falsos positivos: {false_positives} ({false_positives/total*100:.1f}%)")
        
        # Distribución de confianza
        if 'nivel_confianza' in results_df.columns:
            high_conf = len(results_df[results_df['nivel_confianza'] == 'ALTA'])
            medium_conf = len(results_df[results_df['nivel_confianza'] == 'MEDIA'])
            low_conf = len(results_df[results_df['nivel_confianza'] == 'BAJA'])
            
            print(f"\n   🎯 Niveles de confianza:")
            print(f"      ALTA: {high_conf} candidatos")
            print(f"      MEDIA: {medium_conf} candidatos") 
            print(f"      BAJA: {low_conf} candidatos")
        
        # Mostrar resultados individuales
        print(f"\n   📋 Resultados individuales:")
        for idx, row in results_df.iterrows():
            kepoi_name = row.get('kepoi_name', f'Candidato_{idx}')
            # Mapear numérico a texto
            pred_num = row['prediccion']
            if pred_num == 1:
                prediccion = "CONFIRMADO"
            elif pred_num == 2:
                prediccion = "FALSO POSITIVO"
            else:
                prediccion = "CANDIDATO"
                
            probabilidad = row.get('probabilidad_exoplaneta', 0)
            confianza = row.get('nivel_confianza', 'N/A')
            
            print(f"      {kepoi_name}: {prediccion} (prob: {probabilidad:.3f}) [{confianza}]")

    