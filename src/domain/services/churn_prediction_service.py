"""
Servicio de dominio para la predicción de churn.
Define la interfaz para los modelos de predicción de churn.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List
import numpy as np


class ChurnPredictionModel(Protocol):
    """
    Protocolo que define el contrato para modelos de predicción de churn.
    
    Este protocolo permite que diferentes implementaciones de modelos
    sean intercambiables sin afectar la lógica de negocio.
    """
    
    @abstractmethod
    def predict(self, customer_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza una predicción de churn para un cliente.
        
        Args:
            customer_features (Dict[str, Any]): Características del cliente
            
        Returns:
            Dict[str, Any]: Predicción con probabilidad y clase predicha
        """
        pass
    
    @abstractmethod
    def predict_batch(self, customers_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Realiza predicciones de churn para múltiples clientes.
        
        Args:
            customers_features (List[Dict[str, Any]]): Lista de características de clientes
            
        Returns:
            List[Dict[str, Any]]: Lista de predicciones
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Obtiene la importancia de las características del modelo.
        
        Returns:
            Dict[str, float]: Importancia de cada característica
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el modelo.
        
        Returns:
            Dict[str, Any]: Información del modelo (tipo, versión, métricas, etc.)
        """
        pass


class AbstractChurnPredictionModel(ABC):
    """
    Clase abstracta alternativa al protocolo para implementaciones que prefieran herencia.
    """
    
    @abstractmethod
    def predict(self, customer_features: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def predict_batch(self, customers_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass


class PredictionResult:
    """
    Objeto de valor que encapsula el resultado de una predicción de churn.
    """
    
    def __init__(self, customer_id: str, churn_probability: float, predicted_class: bool, 
                 confidence: float, model_version: str, features_used: List[str]):
        """
        Inicializa el resultado de predicción.
        
        Args:
            customer_id (str): ID del cliente
            churn_probability (float): Probabilidad de churn (0-1)
            predicted_class (bool): Clase predicha (True=churn, False=no churn)
            confidence (float): Confianza en la predicción (0-1)
            model_version (str): Versión del modelo usado
            features_used (List[str]): Características utilizadas en la predicción
        """
        self.customer_id = customer_id
        self.churn_probability = churn_probability
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.model_version = model_version
        self.features_used = features_used
        self.predicted_at = np.datetime64('now')
    
    @property
    def risk_level(self) -> str:
        """
        Determina el nivel de riesgo basado en la probabilidad de churn.
        
        Returns:
            str: Nivel de riesgo ('low', 'medium', 'high', 'critical')
        """
        if self.churn_probability < 0.3:
            return 'low'
        elif self.churn_probability < 0.5:
            return 'medium'
        elif self.churn_probability < 0.7:
            return 'high'
        else:
            return 'critical'
    
    @property
    def is_high_risk(self) -> bool:
        """
        Determina si el cliente es de alto riesgo.
        
        Returns:
            bool: True si es alto riesgo, False en caso contrario
        """
        return self.churn_probability >= 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el resultado a diccionario para facilitar la serialización.
        
        Returns:
            Dict[str, Any]: Representación en diccionario
        """
        return {
            'customer_id': self.customer_id,
            'churn_probability': self.churn_probability,
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'risk_level': self.risk_level,
            'is_high_risk': self.is_high_risk,
            'model_version': self.model_version,
            'features_used': self.features_used,
            'predicted_at': str(self.predicted_at)
        }