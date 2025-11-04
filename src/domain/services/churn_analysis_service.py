"""
Servicio de dominio que contiene la lógica de negocio para el análisis de churn.
Estos servicios son independientes de la infraestructura y contienen solo lógica de negocio.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import numpy as np
from ..entities.customer import Customer


class ChurnAnalysisService:
    """
    Servicio que contiene la lógica de negocio para el análisis de churn.
    
    Este servicio es independiente de la infraestructura y se enfoca
    en implementar las reglas de negocio relacionadas con el churn.
    """
    
    @staticmethod
    def calculate_customer_lifetime_value(customer: Customer) -> float:
        """
        Calcula el valor de vida del cliente (CLV) basado en cargos mensuales y antigüedad.
        
        Args:
            customer (Customer): Cliente para calcular CLV
            
        Returns:
            float: Valor estimado de vida del cliente
        """
        # Estimación simple de CLV basada en cargos mensuales y antigüedad
        avg_monthly_charge = customer.monthly_charges
        
        # Si el cliente ya tiene churn, usar la antigüedad real
        if customer.churn:
            return avg_monthly_charge * customer.tenure_months
        
        # Si no tiene churn, estimar vida útil restante basada en antigüedad
        # Clientes con más antigüedad tienden a permanecer más tiempo
        if customer.tenure_months < 12:
            estimated_remaining_months = 24  # Estimación conservadora para nuevos clientes
        elif customer.tenure_months < 24:
            estimated_remaining_months = 36
        else:
            estimated_remaining_months = 48  # Clientes leales
        
        return avg_monthly_charge * (customer.tenure_months + estimated_remaining_months)
    
    @staticmethod
    def calculate_churn_risk_score(customer: Customer) -> float:
        """
        Calcula un puntaje de riesgo de churn basado en características del cliente.
        
        Este es un cálculo simple basado en heurísticas de negocio.
        
        Args:
            customer (Customer): Cliente para evaluar
            
        Returns:
            float: Puntaje de riesgo entre 0 y 100
        """
        risk_score = 0.0
        
        # Factores que aumentan el riesgo de churn
        
        # 1. Tipo de contrato (mensual = mayor riesgo)
        if customer.contract_type == 'Month-to-month':
            risk_score += 30.0
        elif customer.contract_type == 'One year':
            risk_score += 15.0
        # Two year contract no añade riesgo
        
        # 2. Método de pago (electrónico = menor riesgo)
        if customer.payment_method in ['Mailed check', 'Bank transfer (automatic)']:
            risk_score += 20.0
        elif customer.payment_method == 'Credit card (automatic)':
            risk_score += 10.0
        # Electronic check no añade riesgo adicional
        
        # 3. Servicios adicionales (más servicios = menor riesgo)
        service_count = customer.service_count
        if service_count <= 3:
            risk_score += 25.0
        elif service_count <= 6:
            risk_score += 15.0
        elif service_count <= 9:
            risk_score += 5.0
        # Más de 9 servicios no añade riesgo
        
        # 4. Antigüedad (menor antigüedad = mayor riesgo)
        if customer.tenure_months < 12:
            risk_score += 20.0
        elif customer.tenure_months < 24:
            risk_score += 10.0
        elif customer.tenure_months < 36:
            risk_score += 5.0
        # Más de 36 meses no añade riesgo
        
        # 5. Cargos mensuales (cargos muy altos = mayor riesgo)
        if customer.monthly_charges > 100:
            risk_score += 15.0
        elif customer.monthly_charges > 80:
            risk_score += 10.0
        elif customer.monthly_charges > 60:
            risk_score += 5.0
        # Menos de 60 no añade riesgo
        
        # 6. Factores demográficos
        if customer.senior_citizen:
            risk_score += 5.0
        
        if not customer.partner and not customer.dependents:
            risk_score += 10.0  # Clientes solteros sin dependientes
        
        # Normalizar el puntaje a rango 0-100
        return min(risk_score, 100.0)
    
    @staticmethod
    def identify_at_risk_customers(customers: List[Customer], risk_threshold: float = 70.0) -> List[Customer]:
        """
        Identifica clientes en riesgo basándose en el puntaje de riesgo.
        
        Args:
            customers (List[Customer]): Lista de clientes a evaluar
            risk_threshold (float): Umbral de riesgo (default: 70.0)
            
        Returns:
            List[Customer]: Clientes que superan el umbral de riesgo
        """
        at_risk = []
        
        for customer in customers:
            # Solo evaluar clientes activos
            if customer.churn is False or customer.churn is None:
                risk_score = ChurnAnalysisService.calculate_churn_risk_score(customer)
                if risk_score >= risk_threshold:
                    at_risk.append(customer)
        
        return at_risk
    
    @staticmethod
    def segment_customers(customers: List[Customer]) -> Dict[str, List[Customer]]:
        """
        Segmenta clientes en grupos basándose en características de negocio.
        
        Args:
            customers (List[Customer]): Lista de clientes a segmentar
            
        Returns:
            Dict[str, List[Customer]]: Diccionario con segmentos de clientes
        """
        segments = {
            'high_value_low_risk': [],
            'high_value_high_risk': [],
            'low_value_low_risk': [],
            'low_value_high_risk': [],
            'new_customers': [],
            'churned': []
        }
        
        for customer in customers:
            # Clientes con churn
            if customer.churn:
                segments['churned'].append(customer)
                continue
            
            # Clientes nuevos (< 6 meses)
            if customer.tenure_months < 6:
                segments['new_customers'].append(customer)
                continue
            
            # Calcular valor y riesgo
            is_high_value = customer.is_high_value_customer
            risk_score = ChurnAnalysisService.calculate_churn_risk_score(customer)
            is_high_risk = risk_score >= 70.0
            
            # Segmentar basándose en valor y riesgo
            if is_high_value and not is_high_risk:
                segments['high_value_low_risk'].append(customer)
            elif is_high_value and is_high_risk:
                segments['high_value_high_risk'].append(customer)
            elif not is_high_value and not is_high_risk:
                segments['low_value_low_risk'].append(customer)
            else:  # not is_high_value and is_high_risk
                segments['low_value_high_risk'].append(customer)
        
        return segments
    
    @staticmethod
    def calculate_segment_metrics(customers: List[Customer]) -> Dict[str, Any]:
        """
        Calcula métricas clave por segmento de clientes.
        
        Args:
            customers (List[Customer]): Lista de clientes
            
        Returns:
            Dict[str, Any]: Métricas por segmento
        """
        segments = ChurnAnalysisService.segment_customers(customers)
        
        metrics = {}
        
        for segment_name, segment_customers in segments.items():
            if not segment_customers:
                metrics[segment_name] = {
                    'count': 0,
                    'avg_monthly_charges': 0,
                    'avg_tenure_months': 0,
                    'total_revenue': 0
                }
                continue
            
            monthly_charges = [c.monthly_charges for c in segment_customers]
            tenures = [c.tenure_months for c in segment_customers]
            
            metrics[segment_name] = {
                'count': len(segment_customers),
                'avg_monthly_charges': np.mean(monthly_charges),
                'avg_tenure_months': np.mean(tenures),
                'total_revenue': sum(c.monthly_charges * c.tenure_months for c in segment_customers)
            }
        
        return metrics