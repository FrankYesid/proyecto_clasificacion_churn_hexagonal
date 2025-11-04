"""
Entidad de dominio que representa a un cliente en el sistema de churn.
Esta entidad contiene todos los atributos relevantes para el análisis de churn.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class Customer:
    """
    Entidad Customer que representa la información de un cliente.
    
    Esta entidad es agnóstica a la infraestructura y contiene solo
    la lógica de negocio relevante para el dominio de churn.
    """
    
    # Información básica del cliente
    customer_id: str
    gender: str
    senior_citizen: bool
    partner: bool
    dependents: bool
    
    # Información de servicios
    phone_service: bool
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    
    # Información de cuenta
    contract_type: str
    paperless_billing: bool
    payment_method: str
    
    # Información financiera
    monthly_charges: float
    total_charges: float
    tenure_months: int
    
    # Información de churn
    churn: Optional[bool] = None
    
    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validación y transformación de datos después de la inicialización."""
        if self.created_at is None:
            self.created_at = datetime.now()
        
        # Validar valores críticos
        if self.monthly_charges < 0:
            raise ValueError("Los cargos mensuales no pueden ser negativos")
        
        if self.total_charges < 0:
            raise ValueError("Los cargos totales no pueden ser negativos")
        
        if self.tenure_months < 0:
            raise ValueError("La antigüedad no puede ser negativa")
    
    @property
    def is_high_value_customer(self) -> bool:
        """
        Determina si el cliente es de alto valor basado en cargos mensuales.
        
        Returns:
            bool: True si el cliente es de alto valor, False en caso contrario
        """
        return self.monthly_charges > 70.0
    
    @property
    def has_premium_services(self) -> bool:
        """
        Determina si el cliente tiene servicios premium.
        
        Returns:
            bool: True si tiene servicios premium, False en caso contrario
        """
        premium_services = [
            self.online_security == 'Yes',
            self.online_backup == 'Yes',
            self.device_protection == 'Yes',
            self.tech_support == 'Yes',
            self.streaming_tv == 'Yes',
            self.streaming_movies == 'Yes'
        ]
        return any(premium_services)
    
    @property
    def service_count(self) -> int:
        """
        Cuenta el número de servicios adicionales que tiene el cliente.
        
        Returns:
            int: Número de servicios adicionales
        """
        services = [
            self.phone_service,
            self.multiple_lines == 'Yes',
            self.internet_service != 'No',
            self.online_security == 'Yes',
            self.online_backup == 'Yes',
            self.device_protection == 'Yes',
            self.tech_support == 'Yes',
            self.streaming_tv == 'Yes',
            self.streaming_movies == 'Yes'
        ]
        return sum(services)
    
    def to_dict(self) -> dict:
        """
        Convierte la entidad a un diccionario para facilitar la serialización.
        
        Returns:
            dict: Representación en diccionario del cliente
        """
        return {
            'customer_id': self.customer_id,
            'gender': self.gender,
            'senior_citizen': self.senior_citizen,
            'partner': self.partner,
            'dependents': self.dependents,
            'phone_service': self.phone_service,
            'multiple_lines': self.multiple_lines,
            'internet_service': self.internet_service,
            'online_security': self.online_security,
            'online_backup': self.online_backup,
            'device_protection': self.device_protection,
            'tech_support': self.tech_support,
            'streaming_tv': self.streaming_tv,
            'streaming_movies': self.streaming_movies,
            'contract_type': self.contract_type,
            'paperless_billing': self.paperless_billing,
            'payment_method': self.payment_method,
            'monthly_charges': self.monthly_charges,
            'total_charges': self.total_charges,
            'tenure_months': self.tenure_months,
            'churn': self.churn,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Customer':
        """
        Crea una instancia de Customer desde un diccionario.
        
        Args:
            data (dict): Diccionario con los datos del cliente
            
        Returns:
            Customer: Nueva instancia de Customer
        """
        # Convertir strings de fecha a objetos datetime si es necesario
        created_at = data.get('created_at')
        updated_at = data.get('updated_at')
        
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return cls(
            customer_id=data['customer_id'],
            gender=data['gender'],
            senior_citizen=data['senior_citizen'],
            partner=data['partner'],
            dependents=data['dependents'],
            phone_service=data['phone_service'],
            multiple_lines=data['multiple_lines'],
            internet_service=data['internet_service'],
            online_security=data['online_security'],
            online_backup=data['online_backup'],
            device_protection=data['device_protection'],
            tech_support=data['tech_support'],
            streaming_tv=data['streaming_tv'],
            streaming_movies=data['streaming_movies'],
            contract_type=data['contract_type'],
            paperless_billing=data['paperless_billing'],
            payment_method=data['payment_method'],
            monthly_charges=data['monthly_charges'],
            total_charges=data['total_charges'],
            tenure_months=data['tenure_months'],
            churn=data.get('churn'),
            created_at=created_at,
            updated_at=updated_at
        )