"""
DTO (Data Transfer Object) para la entrada de datos de clientes.
Utilizado para transferir datos entre capas sin exponer la entidad de dominio.
"""

from typing import Optional
from pydantic import BaseModel, Field, validator


class CustomerInputDTO(BaseModel):
    """
    DTO para la entrada de datos de un cliente.
    
    Este DTO valida y estructura los datos de entrada antes de procesarlos
    en el dominio.
    """
    
    # Información básica del cliente
    customer_id: Optional[str] = Field(None, description="ID único del cliente")
    gender: str = Field(..., description="Género del cliente", regex="^(Male|Female)$")
    senior_citizen: bool = Field(..., description="¿Es ciudadano senior?")
    partner: bool = Field(..., description="¿Tiene pareja?")
    dependents: bool = Field(..., description="¿Tiene dependientes?")
    
    # Información de servicios
    phone_service: bool = Field(..., description="¿Tiene servicio telefónico?")
    multiple_lines: str = Field(..., description="Líneas múltiples", regex="^(Yes|No|No phone service)$")
    internet_service: str = Field(..., description="Servicio de internet", regex="^(DSL|Fiber optic|No)$")
    online_security: str = Field(..., description="Seguridad en línea", regex="^(Yes|No|No internet service)$")
    online_backup: str = Field(..., description="Respaldo en línea", regex="^(Yes|No|No internet service)$")
    device_protection: str = Field(..., description="Protección de dispositivos", regex="^(Yes|No|No internet service)$")
    tech_support: str = Field(..., description="Soporte técnico", regex="^(Yes|No|No internet service)$")
    streaming_tv: str = Field(..., description="Streaming TV", regex="^(Yes|No|No internet service)$")
    streaming_movies: str = Field(..., description="Películas en streaming", regex="^(Yes|No|No internet service)$")
    
    # Información de cuenta
    contract_type: str = Field(..., description="Tipo de contrato", regex="^(Month-to-month|One year|Two year)$")
    paperless_billing: bool = Field(..., description="¿Facturación sin papel?")
    payment_method: str = Field(..., description="Método de pago", 
                               regex="^(Electronic check|Mailed check|Bank transfer \(automatic\)|Credit card \(automatic\))$")
    
    # Información financiera
    monthly_charges: float = Field(..., gt=0, description="Cargos mensuales del cliente")
    total_charges: float = Field(..., ge=0, description="Cargos totales del cliente")
    tenure_months: int = Field(..., ge=0, description="Antigüedad en meses")
    
    @validator('total_charges')
    def validate_total_charges(cls, v, values):
        """Valida que los cargos totales sean coherentes con los cargos mensuales."""
        if 'monthly_charges' in values and 'tenure_months' in values:
            monthly_charges = values['monthly_charges']
            tenure_months = values['tenure_months']
            
            # Los cargos totales no deberían ser significativamente menores que mensual * antigüedad
            expected_min_total = monthly_charges * tenure_months * 0.8  # 20% de tolerancia
            
            if v < expected_min_total and tenure_months > 0:
                raise ValueError(f'Los cargos totales ({v}) parecen inconsistentes con los cargos mensuales ({monthly_charges}) y antigüedad ({tenure_months})')
        
        return v
    
    @validator('multiple_lines')
    def validate_multiple_lines(cls, v, values):
        """Valida que las líneas múltiples sean coherentes con el servicio telefónico."""
        if 'phone_service' in values:
            phone_service = values['phone_service']
            
            if not phone_service and v != 'No phone service':
                raise ValueError('No puede tener líneas múltiples sin servicio telefónico')
            
            if phone_service and v == 'No phone service':
                raise ValueError('No puede tener líneas múltiples marcadas como "No phone service" si tiene servicio telefónico')
        
        return v
    
    @validator('internet_service')
    def validate_internet_service_consistency(cls, v, values):
        """Valida la coherencia de los servicios de internet."""
        internet_fields = [
            'online_security', 'online_backup', 'device_protection',
            'tech_support', 'streaming_tv', 'streaming_movies'
        ]
        
        if v == 'No':
            # Si no hay servicio de internet, todos los servicios relacionados deben ser 'No internet service'
            for field in internet_fields:
                if field in values and values[field] not in ['No internet service', 'No']:
                    raise ValueError(f'Si no hay servicio de internet, {field} debe ser "No internet service" o "No"')
        
        return v
    
    class Config:
        """Configuración del modelo Pydantic."""
        schema_extra = {
            "example": {
                "customer_id": "12345",
                "gender": "Male",
                "senior_citizen": False,
                "partner": True,
                "dependents": False,
                "phone_service": True,
                "multiple_lines": "No",
                "internet_service": "DSL",
                "online_security": "Yes",
                "online_backup": "No",
                "device_protection": "Yes",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "No",
                "contract_type": "Month-to-month",
                "paperless_billing": True,
                "payment_method": "Electronic check",
                "monthly_charges": 65.5,
                "total_charges": 650.0,
                "tenure_months": 10
            }
        }


class CustomerOutputDTO(BaseModel):
    """
    DTO para la salida de datos de un cliente con predicción.
    """
    
    customer_data: CustomerInputDTO
    prediction: Optional[Dict[str, any]] = None
    risk_score: Optional[float] = None
    segment: Optional[str] = None
    
    class Config:
        """Configuración del modelo Pydantic."""
        schema_extra = {
            "example": {
                "customer_data": CustomerInputDTO.Config.schema_extra["example"],
                "prediction": {
                    "churn_probability": 0.75,
                    "predicted_class": True,
                    "confidence": 0.85,
                    "risk_level": "high",
                    "model_version": "v1.0"
                },
                "risk_score": 75.0,
                "segment": "high_value_high_risk"
            }
        }


class BatchPredictionInputDTO(BaseModel):
    """
    DTO para la entrada de predicciones por lote.
    """
    
    customers: List[CustomerInputDTO]
    include_analysis: bool = Field(True, description="¿Incluir análisis adicional?")
    
    @validator('customers')
    def validate_customers_not_empty(cls, v):
        """Valida que la lista de clientes no esté vacía."""
        if not v:
            raise ValueError('La lista de clientes no puede estar vacía')
        return v


class BatchPredictionOutputDTO(BaseModel):
    """
    DTO para la salida de predicciones por lote.
    """
    
    predictions: List[CustomerOutputDTO]
    summary: Dict[str, Any]
    processing_time_seconds: float
    total_customers: int
    high_risk_count: int
    low_risk_count: int