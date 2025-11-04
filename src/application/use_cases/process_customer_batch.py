"""
Caso de uso para procesar datos de clientes en lote.
Procesa múltiples clientes y devuelve predicciones.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from ...domain.entities.customer import Customer
from ...domain.services.churn_analysis_service import ChurnAnalysisService
from ...domain.services.churn_prediction_service import ChurnPredictionModel, PredictionResult
from ..dto.customer_dto import CustomerInputDTO, CustomerOutputDTO, BatchPredictionInputDTO, BatchPredictionOutputDTO


logger = logging.getLogger(__name__)


class ProcessCustomerBatchUseCase:
    """
    Caso de uso para procesar múltiples clientes en lote.
    
    Este caso de uso orquesta:
    1. Validación de datos de entrada
    2. Conversión de DTOs a entidades de dominio
    3. Predicción de churn para cada cliente
    4. Análisis adicional (segmentación, riesgo)
    5. Conversión de resultados a DTOs de salida
    """
    
    def __init__(self, 
                 prediction_model: ChurnPredictionModel,
                 analysis_service: ChurnAnalysisService):
        """
        Inicializa el caso de uso con las dependencias necesarias.
        
        Args:
            prediction_model (ChurnPredictionModel): Modelo de predicción de churn
            analysis_service (ChurnAnalysisService): Servicio de análisis de churn
        """
        self.prediction_model = prediction_model
        self.analysis_service = analysis_service
    
    def execute(self, batch_input: BatchPredictionInputDTO) -> BatchPredictionOutputDTO:
        """
        Procesa un lote de clientes y devuelve las predicciones.
        
        Args:
            batch_input (BatchPredictionInputDTO): Datos de entrada con lista de clientes
            
        Returns:
            BatchPredictionOutputDTO: Resultados del procesamiento en lote
        """
        logger.info(f"Procesando lote de {len(batch_input.customers)} clientes")
        
        try:
            # Validar datos de entrada
            self._validate_batch_input(batch_input)
            
            # Procesar cada cliente
            results = []
            errors = []
            
            for i, customer_input in enumerate(batch_input.customers):
                try:
                    # Procesar cliente individual
                    result = self._process_single_customer(customer_input)
                    results.append(result)
                    logger.debug(f"Cliente {i+1} procesado exitosamente")
                    
                except Exception as e:
                    # Manejar errores individuales sin detener el procesamiento
                    error_msg = f"Error procesando cliente {i+1}: {str(e)}"
                    logger.error(error_msg)
                    errors.append({
                        'customer_index': i,
                        'customer_id': getattr(customer_input, 'customer_id', 'unknown'),
                        'error': str(e)
                    })
            
            # Preparar resultados del lote
            batch_output = BatchPredictionOutputDTO(
                batch_id=batch_input.batch_id,
                processed_count=len(results),
                error_count=len(errors),
                results=results,
                errors=errors if errors else None
            )
            
            logger.info(f"Lote procesado: {len(results)} exitosos, {len(errors)} errores")
            return batch_output
            
        except ValidationError as e:
            logger.error(f"Error de validación en lote: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado procesando lote: {str(e)}")
            raise ProcessingError(f"Error procesando lote: {str(e)}") from e
    
    def _validate_batch_input(self, batch_input: BatchPredictionInputDTO):
        """
        Valida los datos de entrada del lote.
        
        Args:
            batch_input (BatchPredictionInputDTO): Datos a validar
            
        Raises:
            ValidationError: Si los datos no son válidos
        """
        if not batch_input.customers:
            raise ValidationError("El lote no contiene clientes")
        
        if len(batch_input.customers) > 1000:  # Límite arbitrario para protección
            raise ValidationError("El lote excede el tamaño máximo permitido (1000 clientes)")
        
        # Validar cada cliente individualmente
        for i, customer_input in enumerate(batch_input.customers):
            try:
                customer_input.validate()
            except ValueError as e:
                raise ValidationError(f"Error en cliente {i+1}: {str(e)}")
    
    def _process_single_customer(self, customer_input: CustomerInputDTO) -> CustomerOutputDTO:
        """
        Procesa un cliente individual y devuelve la predicción.
        
        Args:
            customer_input (CustomerInputDTO): Datos del cliente
            
        Returns:
            CustomerOutputDTO: Resultado del procesamiento
        """
        # Convertir DTO a entidad de dominio
        customer = customer_input.to_entity()
        
        # Realizar predicción
        prediction_result = self.prediction_model.predict(customer)
        
        # Calcular análisis adicional
        risk_score = self.analysis_service.calculate_risk_score(customer)
        customer_segment = self.analysis_service.segment_customer(customer, prediction_result)
        lifetime_value = self.analysis_service.calculate_customer_lifetime_value(customer)
        
        # Convertir resultado a DTO de salida
        customer_output = CustomerOutputDTO(
            customer_id=customer.customer_id,
            prediction=prediction_result.predicted_class,
            probability_churn=prediction_result.probability,
            probability_stay=1.0 - prediction_result.probability,
            confidence=prediction_result.confidence,
            risk_level=prediction_result.risk_level,
            risk_score=risk_score,
            customer_segment=customer_segment,
            lifetime_value=lifetime_value,
            model_version=self.prediction_model.get_model_version(),
            prediction_timestamp=prediction_result.timestamp
        )
        
        return customer_output


class ValidationError(Exception):
    """Excepción para errores de validación."""
    pass


class ProcessingError(Exception):
    """Excepción para errores durante el procesamiento."""
    pass