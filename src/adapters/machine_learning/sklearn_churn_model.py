"""
Adaptador de modelo de machine learning para predicción de churn.
Implementa la interfaz ChurnPredictionModel usando scikit-learn.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

from ...domain.entities.customer import Customer
from ...domain.services.churn_prediction_service import ChurnPredictionModel, PredictionResult


logger = logging.getLogger(__name__)


class SklearnChurnPredictionModel(ChurnPredictionModel):
    """
    Adaptador de modelo de predicción usando scikit-learn.
    
    Este adaptador permite:
    - Cargar modelos entrenados de scikit-learn
    - Realizar predicciones de churn
    - Calcular probabilidades y confianza
    - Gestionar versiones del modelo
    """
    
    def __init__(self, model_path: str, preprocessor_path: Optional[str] = None, model_version: str = "1.0.0"):
        """
        Inicializa el modelo con la ruta al archivo del modelo.
        
        Args:
            model_path (str): Ruta al archivo del modelo entrenado (.joblib)
            preprocessor_path (Optional[str]): Ruta al preprocesador (opcional)
            model_version (str): Versión del modelo para tracking
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model_version = model_version
        self.model = None
        self.preprocessor = None
        self._load_model()
    
    def _load_model(self):
        """
        Carga el modelo y preprocesador desde archivos.
        """
        try:
            # Cargar modelo
            self.model = joblib.load(self.model_path)
            logger.info(f"Modelo cargado exitosamente: {self.model_path}")
            
            # Cargar preprocesador si existe
            if self.preprocessor_path and self.preprocessor_path.exists():
                self.preprocessor = joblib.load(self.preprocessor_path)
                logger.info(f"Preprocesador cargado: {self.preprocessor_path}")
            
        except FileNotFoundError:
            logger.error(f"Archivo de modelo no encontrado: {self.model_path}")
            raise ModelNotFoundError(f"No se encontró el modelo en: {self.model_path}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            raise ModelLoadingError(f"Error al cargar el modelo: {str(e)}") from e
    
    def predict(self, customer: Customer) -> PredictionResult:
        """
        Realiza la predicción de churn para un cliente.
        
        Args:
            customer (Customer): Cliente para predecir
            
        Returns:
            PredictionResult: Resultado de la predicción
        """
        if self.model is None:
            raise ModelNotLoadedError("El modelo no está cargado")
        
        try:
            # Convertir cliente a formato de entrada del modelo
            input_data = self._customer_to_model_input(customer)
            
            # Realizar predicción
            prediction = self.model.predict(input_data)[0]
            probability = self.model.predict_proba(input_data)[0]
            
            # Calcular métricas adicionales
            churn_probability = probability[1] if len(probability) > 1 else probability[0]
            confidence = self._calculate_confidence(probability)
            risk_level = self._determine_risk_level(churn_probability, confidence)
            
            # Crear resultado
            result = PredictionResult(
                customer_id=customer.customer_id,
                predicted_class=bool(prediction),
                probability=float(churn_probability),
                confidence=float(confidence),
                risk_level=risk_level,
                timestamp=datetime.now(),
                model_version=self.model_version,
                additional_info={
                    'probability_stay': float(1.0 - churn_probability),
                    'model_type': type(self.model).__name__,
                    'features_used': self._get_feature_names()
                }
            )
            
            logger.info(f"Predicción realizada para cliente {customer.customer_id}: "
                       f"{'Churn' if prediction else 'No Churn'} "
                       f"(prob: {churn_probability:.3f}, conf: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error realizando predicción: {str(e)}")
            raise PredictionError(f"Error en predicción: {str(e)}") from e
    
    def predict_batch(self, customers: list[Customer]) -> list[PredictionResult]:
        """
        Realiza predicciones para múltiples clientes.
        
        Args:
            customers (list[Customer]): Lista de clientes para predecir
            
        Returns:
            list[PredictionResult]: Lista de resultados de predicción
        """
        if self.model is None:
            raise ModelNotLoadedError("El modelo no está cargado")
        
        try:
            # Convertir clientes a formato de entrada del modelo
            input_data = self._customers_to_model_input(customers)
            
            # Realizar predicciones
            predictions = self.model.predict(input_data)
            probabilities = self.model.predict_proba(input_data)
            
            # Crear resultados
            results = []
            for i, customer in enumerate(customers):
                prediction = predictions[i]
                probability = probabilities[i]
                
                churn_probability = probability[1] if len(probability) > 1 else probability[0]
                confidence = self._calculate_confidence(probability)
                risk_level = self._determine_risk_level(churn_probability, confidence)
                
                result = PredictionResult(
                    customer_id=customer.customer_id,
                    predicted_class=bool(prediction),
                    probability=float(churn_probability),
                    confidence=float(confidence),
                    risk_level=risk_level,
                    timestamp=datetime.now(),
                    model_version=self.model_version,
                    additional_info={
                        'probability_stay': float(1.0 - churn_probability),
                        'model_type': type(self.model).__name__,
                        'batch_index': i
                    }
                )
                
                results.append(result)
            
            logger.info(f"Predicciones en lote completadas: {len(results)} clientes")
            return results
            
        except Exception as e:
            logger.error(f"Error realizando predicciones en lote: {str(e)}")
            raise PredictionError(f"Error en predicciones en lote: {str(e)}") from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el modelo cargado.
        
        Returns:
            Dict[str, Any]: Información del modelo
        """
        if self.model is None:
            return {'status': 'not_loaded'}
        
        info = {
            'status': 'loaded',
            'model_version': self.model_version,
            'model_type': type(self.model).__name__,
            'model_path': str(self.model_path),
            'preprocessor_path': str(self.preprocessor_path) if self.preprocessor_path else None,
            'features': self._get_feature_names(),
            'loaded_at': datetime.now().isoformat()
        }
        
        # Agregar parámetros del modelo si están disponibles
        if hasattr(self.model, 'get_params'):
            info['model_parameters'] = self.model.get_params()
        
        return info
    
    def get_model_version(self) -> str:
        """
        Obtiene la versión del modelo.
        
        Returns:
            str: Versión del modelo
        """
        return self.model_version
    
    def _customer_to_model_input(self, customer: Customer) -> pd.DataFrame:
        """
        Convierte un objeto Customer al formato de entrada del modelo.
        
        Args:
            customer (Customer): Cliente a convertir
            
        Returns:
            pd.DataFrame: Datos en formato de entrada del modelo
        """
        # Extraer características del cliente
        features = {
            'senior_citizen': int(customer.senior_citizen),
            'partner': int(customer.partner),
            'dependents': int(customer.dependents),
            'phone_service': int(customer.phone_service),
            'paperless_billing': int(customer.paperless_billing),
            'monthly_charges': customer.monthly_charges,
            'total_charges': customer.total_charges,
            'tenure_months': customer.tenure_months,
            'gender_male': int(customer.gender.lower() == 'male'),
            'multiple_lines_no_phone_service': int(customer.multiple_lines == 'No phone service'),
            'multiple_lines_yes': int(customer.multiple_lines == 'Yes'),
            'internet_service_fiber_optic': int(customer.internet_service == 'Fiber optic'),
            'internet_service_yes': int(customer.internet_service == 'Yes'),
            'online_security_no_internet_service': int(customer.online_security == 'No internet service'),
            'online_security_yes': int(customer.online_security == 'Yes'),
            'online_backup_no_internet_service': int(customer.online_backup == 'No internet service'),
            'online_backup_yes': int(customer.online_backup == 'Yes'),
            'device_protection_no_internet_service': int(customer.device_protection == 'No internet service'),
            'device_protection_yes': int(customer.device_protection == 'Yes'),
            'tech_support_no_internet_service': int(customer.tech_support == 'No internet service'),
            'tech_support_yes': int(customer.tech_support == 'Yes'),
            'streaming_tv_no_internet_service': int(customer.streaming_tv == 'No internet service'),
            'streaming_tv_yes': int(customer.streaming_tv == 'Yes'),
            'streaming_movies_no_internet_service': int(customer.streaming_movies == 'No internet service'),
            'streaming_movies_yes': int(customer.streaming_movies == 'Yes'),
            'contract_type_one_year': int(customer.contract_type == 'One year'),
            'contract_type_two_year': int(customer.contract_type == 'Two year'),
            'payment_method_bank_transfer_automatic': int(customer.payment_method == 'Bank transfer (automatic)'),
            'payment_method_credit_card_automatic': int(customer.payment_method == 'Credit card (automatic)'),
            'payment_method_electronic_check': int(customer.payment_method == 'Electronic check'),
            'payment_method_mailed_check': int(customer.payment_method == 'Mailed check')
        }
        
        # Crear DataFrame con una sola fila
        return pd.DataFrame([features])
    
    def _customers_to_model_input(self, customers: list[Customer]) -> pd.DataFrame:
        """
        Convierte múltiples objetos Customer al formato de entrada del modelo.
        
        Args:
            customers (list[Customer]): Lista de clientes a convertir
            
        Returns:
            pd.DataFrame: Datos en formato de entrada del modelo
        """
        features_list = []
        
        for customer in customers:
            features = self._customer_to_model_input(customer).iloc[0].to_dict()
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _calculate_confidence(self, probabilities: np.ndarray) -> float:
        """
        Calcula el nivel de confianza basado en las probabilidades.
        
        Args:
            probabilities (np.ndarray): Array de probabilidades
            
        Returns:
            float: Nivel de confianza (0-1)
        """
        if len(probabilities) == 1:
            return 1.0
        
        # Confianza basada en la diferencia entre las probabilidades
        sorted_probs = np.sort(probabilities)[::-1]
        confidence = sorted_probs[0] - sorted_probs[1]
        return float(confidence)
    
    def _determine_risk_level(self, churn_probability: float, confidence: float) -> str:
        """
        Determina el nivel de riesgo basado en la probabilidad y confianza.
        
        Args:
            churn_probability (float): Probabilidad de churn
            confidence (float): Nivel de confianza
            
        Returns:
            str: Nivel de riesgo (Low, Medium, High, Critical)
        """
        if churn_probability > 0.8 and confidence > 0.7:
            return "Critical"
        elif churn_probability > 0.6 and confidence > 0.5:
            return "High"
        elif churn_probability > 0.4 and confidence > 0.3:
            return "Medium"
        else:
            return "Low"
    
    def _get_feature_names(self) -> list[str]:
        """
        Obtiene los nombres de las características que espera el modelo.
        
        Returns:
            list[str]: Lista de nombres de características
        """
        # Basado en el formato de entrada del modelo
        return [
            'senior_citizen', 'partner', 'dependents', 'phone_service', 'paperless_billing',
            'monthly_charges', 'total_charges', 'tenure_months', 'gender_male',
            'multiple_lines_no_phone_service', 'multiple_lines_yes',
            'internet_service_fiber_optic', 'internet_service_yes',
            'online_security_no_internet_service', 'online_security_yes',
            'online_backup_no_internet_service', 'online_backup_yes',
            'device_protection_no_internet_service', 'device_protection_yes',
            'tech_support_no_internet_service', 'tech_support_yes',
            'streaming_tv_no_internet_service', 'streaming_tv_yes',
            'streaming_movies_no_internet_service', 'streaming_movies_yes',
            'contract_type_one_year', 'contract_type_two_year',
            'payment_method_bank_transfer_automatic', 'payment_method_credit_card_automatic',
            'payment_method_electronic_check', 'payment_method_mailed_check'
        ]


class ModelNotFoundError(Exception):
    """Excepción cuando no se encuentra el archivo del modelo."""
    pass


class ModelLoadingError(Exception):
    """Excepción cuando hay error al cargar el modelo."""
    pass


class ModelNotLoadedError(Exception):
    """Excepción cuando el modelo no está cargado."""
    pass


class PredictionError(Exception):
    """Excepción durante la predicción."""
    pass