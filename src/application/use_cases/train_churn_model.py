"""
Caso de uso para entrenar el modelo de predicción de churn.
Orquesta el proceso de entrenamiento del modelo ML.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

from ...domain.entities.customer import Customer
from ...domain.services.churn_analysis_service import ChurnAnalysisService


logger = logging.getLogger(__name__)


class TrainChurnModelUseCase:
    """
    Caso de uso para entrenar el modelo de predicción de churn.
    
    Este caso de uso orquesta:
    1. Carga de datos de entrenamiento
    2. Preprocesamiento de datos
    3. Entrenamiento del modelo
    4. Evaluación del modelo
    5. Guardado del modelo y artefactos
    """
    
    def __init__(self, 
                 data_path: str,
                 model_output_path: str,
                 preprocessor_output_path: str,
                 metrics_output_path: Optional[str] = None):
        """
        Inicializa el caso de uso con las rutas necesarias.
        
        Args:
            data_path (str): Ruta al archivo de datos de entrenamiento
            model_output_path (str): Ruta para guardar el modelo entrenado
            preprocessor_output_path (str): Ruta para guardar el preprocesador
            metrics_output_path (Optional[str]): Ruta opcional para guardar métricas
        """
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.preprocessor_output_path = preprocessor_output_path
        self.metrics_output_path = metrics_output_path
        
        # Configuración del modelo
        self.test_size = 0.2
        self.random_state = 42
        self.n_estimators = 100
        self.max_depth = 10
    
    def execute(self) -> Dict[str, Any]:
        """
        Ejecuta el proceso completo de entrenamiento del modelo.
        
        Returns:
            Dict[str, Any]: Resultados del entrenamiento incluyendo métricas
        """
        logger.info("Iniciando entrenamiento del modelo de churn")
        
        try:
            # 1. Cargar datos
            logger.info("Cargando datos de entrenamiento")
            data = self._load_data()
            
            # 2. Preparar datos
            logger.info("Preparando datos para entrenamiento")
            X, y, preprocessor = self._prepare_data(data)
            
            # 3. Dividir datos
            logger.info("Dividiendo datos en entrenamiento y prueba")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            # 4. Entrenar modelo
            logger.info("Entrenando modelo")
            model = self._train_model(X_train, y_train)
            
            # 5. Evaluar modelo
            logger.info("Evaluando modelo")
            metrics = self._evaluate_model(model, X_test, y_test)
            
            # 6. Guardar modelo y artefactos
            logger.info("Guardando modelo y artefactos")
            self._save_model_and_artifacts(model, preprocessor, metrics)
            
            # 7. Preparar resultados
            results = {
                'status': 'success',
                'training_date': datetime.now().isoformat(),
                'model_type': 'RandomForestClassifier',
                'parameters': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'random_state': self.random_state,
                    'test_size': self.test_size
                },
                'data_info': {
                    'total_samples': len(data),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features': len(X.columns.tolist()),
                    'feature_names': X.columns.tolist()
                },
                'metrics': metrics,
                'model_path': self.model_output_path,
                'preprocessor_path': self.preprocessor_output_path
            }
            
            logger.info(f"Entrenamiento completado exitosamente. Accuracy: {metrics['accuracy']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {str(e)}")
            raise TrainingError(f"Error en entrenamiento del modelo: {str(e)}") from e
    
    def _load_data(self) -> pd.DataFrame:
        """
        Carga los datos de entrenamiento desde el archivo.
        
        Returns:
            pd.DataFrame: Datos cargados
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"No se encontró el archivo de datos: {self.data_path}")
        
        data = pd.read_csv(self.data_path)
        
        # Validar columnas requeridas
        required_columns = [
            'gender', 'senior_citizen', 'partner', 'dependents',
            'phone_service', 'multiple_lines', 'internet_service', 'online_security',
            'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
            'streaming_movies', 'contract_type', 'paperless_billing', 'payment_method',
            'monthly_charges', 'total_charges', 'tenure_months', 'churn'
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Columnas faltantes en los datos: {missing_columns}")
        
        logger.info(f"Datos cargados: {len(data)} filas, {len(data.columns)} columnas")
        return data
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
        """
        Prepara los datos para el entrenamiento.
        
        Args:
            data (pd.DataFrame): Datos crudos
            
        Returns:
            Tuple[pd.DataFrame, pd.Series, ColumnTransformer]: X, y y preprocesador
        """
        # Separar características y target
        X = data.drop('churn', axis=1)
        y = data['churn'].map({'Yes': True, 'No': False})  # Convertir a booleano
        
        # Identificar tipos de columnas
        categorical_features = [
            'gender', 'multiple_lines', 'internet_service', 'online_security',
            'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
            'streaming_movies', 'contract_type', 'payment_method'
        ]
        
        numerical_features = [
            'senior_citizen', 'partner', 'dependents', 'phone_service',
            'paperless_billing', 'monthly_charges', 'total_charges', 'tenure_months'
        ]
        
        # Crear preprocesador
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
        numerical_transformer = StandardScaler()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Aplicar preprocesamiento
        X_processed = preprocessor.fit_transform(X)
        
        # Convertir de vuelta a DataFrame para mantener nombres de columnas
        feature_names = (numerical_features + 
                        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
        X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        
        logger.info(f"Datos preparados: {X_processed.shape[0]} muestras, {X_processed.shape[1]} características")
        return X_processed, y, preprocessor
    
    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        Entrena el modelo Random Forest.
        
        Args:
            X_train (pd.DataFrame): Datos de entrenamiento
            y_train (pd.Series): Etiquetas de entrenamiento
            
        Returns:
            RandomForestClassifier: Modelo entrenado
        """
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight='balanced'  # Manejar desbalance de clases
        )
        
        model.fit(X_train, y_train)
        logger.info("Modelo entrenado exitosamente")
        return model
    
    def _evaluate_model(self, model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evalúa el modelo y calcula métricas.
        
        Args:
            model (RandomForestClassifier): Modelo entrenado
            X_test (pd.DataFrame): Datos de prueba
            y_test (pd.Series): Etiquetas de prueba
            
        Returns:
            Dict[str, float]: Métricas del modelo
        """
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        logger.info(f"Métricas del modelo - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        
        return metrics
    
    def _save_model_and_artifacts(self, model: RandomForestClassifier, preprocessor: ColumnTransformer, metrics: Dict[str, float]):
        """
        Guarda el modelo y los artefactos relacionados.
        
        Args:
            model (RandomForestClassifier): Modelo entrenado
            preprocessor (ColumnTransformer): Preprocesador
            metrics (Dict[str, float]): Métricas del modelo
        """
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.preprocessor_output_path), exist_ok=True)
        
        # Guardar modelo
        joblib.dump(model, self.model_output_path)
        logger.info(f"Modelo guardado en: {self.model_output_path}")
        
        # Guardar preprocesador
        joblib.dump(preprocessor, self.preprocessor_output_path)
        logger.info(f"Preprocesador guardado en: {self.preprocessor_output_path}")
        
        # Guardar métricas si se especificó ruta
        if self.metrics_output_path:
            os.makedirs(os.path.dirname(self.metrics_output_path), exist_ok=True)
            
            metrics_data = {
                'training_date': datetime.now().isoformat(),
                'metrics': metrics,
                'model_parameters': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'random_state': self.random_state
                }
            }
            
            with open(self.metrics_output_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"Métricas guardadas en: {self.metrics_output_path}")


class TrainingError(Exception):
    """
    Excepción personalizada para errores durante el entrenamiento.
    """
    pass