import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class OrionIIDataLoader:
    def __init__(self):
        self.raw_data = None
        self.clean_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_koi_data(self, file_paths):
        """Cargar múltiples archivos CSV de KOI"""
        dataframes = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path, comment='#')
                print(f"✅ Archivo {file_path} cargado: {len(df)} filas")
                dataframes.append(df)
            except Exception as e:
                print(f"❌ Error cargando {file_path}: {e}")
        
        if not dataframes:
            raise ValueError("No se pudieron cargar ningún archivo")
        
        # Combinar todos los dataframes
        self.raw_data = pd.concat(dataframes, ignore_index=True)
        print(f"📊 Dataset combinado: {len(self.raw_data)} filas totales")
        return self.raw_data

    def clean_koi_data(self, df=None):
        """Limpiar y preprocesar los datos de KOI"""
        if df is None:
            df = self.raw_data
        
        # Columnas seleccionadas basadas en tus parámetros
        selected_columns = [
            'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition', 
            'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
            'koi_period', 'koi_period_err1', 'koi_period_err2', 'koi_time0bk', 
            'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_impact', 'koi_impact_err1', 
            'koi_impact_err2', 'koi_duration', 'koi_duration_err1', 'koi_duration_err2', 
            'koi_depth', 'koi_depth_err1', 'koi_depth_err2', 'koi_prad', 'koi_prad_err1', 
            'koi_prad_err2', 'koi_teq', 'koi_teq_err1', 'koi_teq_err2', 'koi_insol', 
            'koi_insol_err1', 'koi_insol_err2', 'koi_model_snr', 'koi_tce_plnt_num', 
            'koi_tce_delivname', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2', 
            'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2', 'koi_srad', 'koi_srad_err1', 
            'koi_srad_err2', 'ra', 'dec', 'koi_kepmag'
        ]
        
        # Seleccionar solo las columnas que existen en el dataframe
        available_columns = [col for col in selected_columns if col in df.columns]
        df_clean = df[available_columns].copy()
        
        # Eliminar filas con valores nulos en columnas críticas
        critical_columns = [
            'kepoi_name', 'koi_disposition', 'koi_period', 'koi_prad', 
            'koi_teq', 'koi_steff', 'koi_srad', 'koi_score'
        ]
        
        # Solo considerar las columnas críticas que existen
        existing_critical = [col for col in critical_columns if col in df_clean.columns]
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=existing_critical)
        final_count = len(df_clean)
        print(f"🗑️  Eliminadas {initial_count - final_count} filas con valores nulos en columnas críticas")
        
        # Manejar valores faltantes en otras columnas numéricas
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in df_clean.columns and col not in ['koi_score']:  # No tocar koi_score
                null_count = df_clean[col].isnull().sum()
                if null_count > 0:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    print(f"🔧 Rellenados {null_count} valores nulos en {col} con la mediana")
        
        # Normalizar columnas numéricas clave (excluyendo koi_score porque ya está en [0,1])
        columns_to_normalize = [
            'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
            'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff', 'koi_srad'
        ]
        
        # Solo normalizar columnas que existen
        existing_to_normalize = [col for col in columns_to_normalize if col in df_clean.columns]
        
        for col in existing_to_normalize:
            normalized_col = f'{col}_normalized'
            df_clean[normalized_col] = self.scaler.fit_transform(df_clean[[col]])
            print(f"📐 Normalizada columna: {col}")
        
        self.clean_data = df_clean
        print(f"✅ Datos limpios: {len(self.clean_data)} filas, {len(self.clean_data.columns)} columnas")
        return self.clean_data

    def prepare_ml_data(self, target_column='koi_disposition', test_size=0.2, random_state=42):
        """
        Preparar datos para Machine Learning
        
        Args:
            target_column: Columna objetivo para clasificación
            test_size: Proporción para test split
            random_state: Semilla para reproducibilidad
        """
        if self.clean_data is None:
            raise ValueError("Primero debe limpiar los datos con clean_koi_data()")
        
        print(f"\n🤖 PREPARANDO DATOS PARA MACHINE LEARNING")
        print("="*50)
        
        # 1. Seleccionar características para ML (excluir columnas de identificación y metadatos)
        exclude_columns = [
            'kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition',
            'koi_tce_delivname', 'koi_score_category'
        ]
        
        # Columnas numéricas para características
        feature_columns = [
            'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
            'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff', 'koi_srad',
            'koi_score', 'koi_kepmag', 'koi_model_snr',
            'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
        ]
        
        # Solo usar columnas que existen en los datos
        ml_features = [col for col in feature_columns if col in self.clean_data.columns]
        ml_features = [col for col in ml_features if col not in exclude_columns]
        
        print(f"📊 Características seleccionadas: {ml_features}")
        
        # 2. Separar características (X) y etiquetas (y)
        X = self.clean_data[ml_features].copy()
        y = self.clean_data[target_column].copy()
        
        print(f"🎯 Variable objetivo: {target_column}")
        print(f"📈 Distribución de clases:")
        print(y.value_counts())
        
        # 3. Codificar etiquetas si son categóricas
        if y.dtype == 'object':
            y_encoded = self.label_encoder.fit_transform(y)
            print(f"🔤 Etiquetas codificadas: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        else:
            y_encoded = y
        
        # 4. Manejar valores NaN restantes en características
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        
        # 5. Estandarizar características
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # 6. Dividir en train y test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        self.feature_names = ml_features
        
        print(f"✅ Datos preparados para ML:")
        print(f"   📐 Características: {X.shape[1]}")
        print(f"   🏋️  Training set: {self.X_train.shape[0]} muestras")
        print(f"   🧪 Test set: {self.X_test.shape[0]} muestras")
        print(f"   🎯 Clases: {len(np.unique(y_encoded))}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_feature_importance_data(self):
        """Obtener datos para análisis de importancia de características"""
        if self.X_train is None:
            raise ValueError("Primero debe preparar los datos con prepare_ml_data()")
        
        return {
            'feature_names': self.feature_names,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'label_encoder': self.label_encoder
        }