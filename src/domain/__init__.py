"""
Archivo de inicialización para el paquete de dominio.
Exporta las clases principales del dominio para facilitar su importación.
"""

from .entities.customer import Customer
from .repositories.customer_repository import CustomerRepository, AbstractCustomerRepository
from .services.churn_analysis_service import ChurnAnalysisService
from .services.churn_prediction_service import (
    ChurnPredictionModel, 
    AbstractChurnPredictionModel, 
    PredictionResult
)

__all__ = [
    'Customer',
    'CustomerRepository',
    'AbstractCustomerRepository',
    'ChurnAnalysisService',
    'ChurnPredictionModel',
    'AbstractChurnPredictionModel',
    'PredictionResult'
]