"""
Módulo de inicialización para el paquete de aplicación.
Exporta los casos de uso y DTOs principales.
"""

from .dto.customer_dto import (
    CustomerInputDTO,
    CustomerOutputDTO,
    BatchPredictionInputDTO,
    BatchPredictionOutputDTO
)

from .use_cases.predict_customer_churn import PredictCustomerChurnUseCase
from .use_cases.train_churn_model import TrainChurnModelUseCase, TrainingError
from .use_cases.process_customer_batch import ProcessCustomerBatchUseCase, ValidationError, ProcessingError

__all__ = [
    # DTOs
    'CustomerInputDTO',
    'CustomerOutputDTO',
    'BatchPredictionInputDTO',
    'BatchPredictionOutputDTO',
    
    # Casos de uso
    'PredictCustomerChurnUseCase',
    'TrainChurnModelUseCase',
    'ProcessCustomerBatchUseCase',
    
    # Excepciones
    'TrainingError',
    'ValidationError',
    'ProcessingError'
]