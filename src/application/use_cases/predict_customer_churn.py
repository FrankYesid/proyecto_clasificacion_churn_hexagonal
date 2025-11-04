"""
Caso de uso para predecir el churn de un cliente individual.
Implementa la lógica de aplicación para la predicción individual.
"""

from typing import Dict, Any, Optional
import time
from ...domain.entities.customer import Customer
from ...domain.services.churn_analysis_service import ChurnAnalysisService
from ...domain.services.churn_prediction_service import ChurnPredictionModel, PredictionResult
from ..dto.customer_dto import CustomerInputDTO, CustomerOutputDTO


class PredictCustomerChurnUseCase:
    """
    Caso de uso para predecir el churn de un cliente individual.
    
    Este caso de uso orquesta la lógica de dominio para:
    1. Convertir DTO a entidad de dominio
    2. Realizar predicción usando el modelo ML
    3. Calcular análisis adicional (riesgo, segmentación)
    4. Retornar resultado estructurado
    """
    
    def __init__(self, prediction_model: ChurnPredictionModel):
        """
        Inicializa el caso de uso con el modelo de predicción requerido.
        
        Args:
            prediction_model (ChurnPredictionModel): Modelo de predicción de churn
        """
        self.prediction_model = prediction_model
    
    def execute(self, customer_dto: CustomerInputDTO, include_analysis: bool = True) -> CustomerOutputDTO:
        """
        Ejecuta la predicción de churn para un cliente.
        
        Args:
            customer_dto (CustomerInputDTO): Datos del cliente
            include_analysis (bool): ¿Incluir análisis adicional?
            
        Returns:
            CustomerOutputDTO: Resultado de la predicción con análisis opcional
        """
        start_time = time.time()
        
        try:
            # Convertir DTO a entidad de dominio
            customer = self._dto_to_entity(customer_dto)
            
            # Realizar predicción
            prediction_result = self._predict_churn(customer)
            
            # Calcular análisis adicional si se solicita
            analysis_data = {}
            if include_analysis:
                analysis_data = self._calculate_additional_analysis(customer, prediction_result)
            
            # Construir respuesta
            output_dto = CustomerOutputDTO(
                customer_data=customer_dto,
                prediction=prediction_result.to_dict() if prediction_result else None,
                risk_score=analysis_data.get('risk_score'),
                segment=analysis_data.get('segment')
            )
            
            return output_dto
            
        except Exception as e:
            # Manejo de errores - podría ser más específico
            raise PredictionError(f"Error al predecir churn: {str(e)}") from e
    
    def _dto_to_entity(self, customer_dto: CustomerInputDTO) -> Customer:
        """
        Convierte el DTO de entrada a una entidad de dominio.
        
        Args:
            customer_dto (CustomerInputDTO): DTO con datos del cliente
            
        Returns:
            Customer: Entidad de dominio
        """
        return Customer(
            customer_id=customer_dto.customer_id or "unknown",
            gender=customer_dto.gender,
            senior_citizen=customer_dto.senior_citizen,
            partner=customer_dto.partner,
            dependents=customer_dto.dependents,
            phone_service=customer_dto.phone_service,
            multiple_lines=customer_dto.multiple_lines,
            internet_service=customer_dto.internet_service,
            online_security=customer_dto.online_security,
            online_backup=customer_dto.online_backup,
            device_protection=customer_dto.device_protection,
            tech_support=customer_dto.tech_support,
            streaming_tv=customer_dto.streaming_tv,
            streaming_movies=customer_dto.streaming_movies,
            contract_type=customer_dto.contract_type,
            paperless_billing=customer_dto.paperless_billing,
            payment_method=customer_dto.payment_method,
            monthly_charges=customer_dto.monthly_charges,
            total_charges=customer_dto.total_charges,
            tenure_months=customer_dto.tenure_months,
            churn=None  # No se conoce el churn real para predicción
        )
    
    def _predict_churn(self, customer: Customer) -> PredictionResult:
        """
        Realiza la predicción de churn usando el modelo ML.
        
        Args:
            customer (Customer): Cliente para predecir
            
        Returns:
            PredictionResult: Resultado de la predicción
        """
        # Convertir entidad a formato que espera el modelo
        customer_features = customer.to_dict()
        
        # Realizar predicción
        prediction_dict = self.prediction_model.predict(customer_features)
        
        # Crear objeto de resultado
        return PredictionResult(
            customer_id=customer.customer_id,
            churn_probability=prediction_dict['churn_probability'],
            predicted_class=prediction_dict['predicted_class'],
            confidence=prediction_dict.get('confidence', 0.8),  # Valor por defecto si no se proporciona
            model_version=prediction_dict.get('model_version', 'unknown'),
            features_used=prediction_dict.get('features_used', list(customer_features.keys()))
        )
    
    def _calculate_additional_analysis(self, customer: Customer, prediction_result: PredictionResult) -> Dict[str, Any]:
        """
        Calcula análisis adicional como riesgo y segmentación.
        
        Args:
            customer (Customer): Cliente analizado
            prediction_result (PredictionResult): Resultado de predicción
            
        Returns:
            Dict[str, Any]: Datos de análisis adicional
        """
        # Calcular puntaje de riesgo basado en lógica de negocio
        risk_score = ChurnAnalysisService.calculate_churn_risk_score(customer)
        
        # Determinar segmento (simplificado)
        segment = self._determine_customer_segment(customer, prediction_result)
        
        return {
            'risk_score': risk_score,
            'segment': segment
        }
    
    def _determine_customer_segment(self, customer: Customer, prediction_result: PredictionResult) -> str:
        """
        Determina el segmento del cliente basándose en valor y riesgo.
        
        Args:
            customer (Customer): Cliente a segmentar
            prediction_result (PredictionResult): Resultado de predicción
            
        Returns:
            str: Segmento del cliente
        """
        is_high_value = customer.is_high_value_customer
        is_high_risk = prediction_result.is_high_risk
        
        if is_high_value and not is_high_risk:
            return "high_value_low_risk"
        elif is_high_value and is_high_risk:
            return "high_value_high_risk"
        elif not is_high_value and not is_high_risk:
            return "low_value_low_risk"
        else:
            return "low_value_high_risk"


class PredictionError(Exception):
    """
    Excepción personalizada para errores en la predicción.
    """
    pass