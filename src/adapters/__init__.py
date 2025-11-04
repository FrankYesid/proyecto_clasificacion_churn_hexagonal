"""
Módulo de inicialización para el paquete de adaptadores.
Exporta los adaptadores principales disponibles.
"""

from .repositories.csv_customer_repository import CsvCustomerRepository
from .machine_learning.sklearn_churn_model import (
    SklearnChurnPredictionModel,
    ModelNotFoundError,
    ModelLoadingError,
    ModelNotLoadedError,
    PredictionError
)

__all__ = [
    # Repositorios
    'CsvCustomerRepository',
    
    # Modelos ML
    'SklearnChurnPredictionModel',
    
    # Excepciones
    'ModelNotFoundError',
    'ModelLoadingError',
    'ModelNotLoadedError',
    'PredictionError'
]