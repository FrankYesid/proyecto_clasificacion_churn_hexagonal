"""
Contrato (interfaz) para el repositorio de clientes.
Define las operaciones que cualquier implementación de repositorio debe proporcionar.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Protocol
from .customer import Customer


class CustomerRepository(Protocol):
    """
    Protocolo que define el contrato para el repositorio de clientes.
    
    Esta interfaz sigue el principio de inversión de dependencias,
    donde las capas de alto nivel no dependen de implementaciones concretas.
    """
    
    @abstractmethod
    def save(self, customer: Customer) -> Customer:
        """
        Guarda un cliente en el repositorio.
        
        Args:
            customer (Customer): Cliente a guardar
            
        Returns:
            Customer: Cliente guardado con posibles actualizaciones (ej: ID generado)
        """
        pass
    
    @abstractmethod
    def find_by_id(self, customer_id: str) -> Optional[Customer]:
        """
        Busca un cliente por su ID.
        
        Args:
            customer_id (str): ID del cliente
            
        Returns:
            Optional[Customer]: Cliente encontrado o None si no existe
        """
        pass
    
    @abstractmethod
    def find_all(self, limit: Optional[int] = None) -> List[Customer]:
        """
        Obtiene todos los clientes del repositorio.
        
        Args:
            limit (Optional[int]): Límite opcional de resultados
            
        Returns:
            List[Customer]: Lista de todos los clientes
        """
        pass
    
    @abstractmethod
    def find_churned_customers(self) -> List[Customer]:
        """
        Obtiene todos los clientes que han abandonado el servicio.
        
        Returns:
            List[Customer]: Lista de clientes con churn
        """
        pass
    
    @abstractmethod
    def find_active_customers(self) -> List[Customer]:
        """
        Obtiene todos los clientes activos (sin churn).
        
        Returns:
            List[Customer]: Lista de clientes activos
        """
        pass
    
    @abstractmethod
    def update(self, customer: Customer) -> Customer:
        """
        Actualiza un cliente existente.
        
        Args:
            customer (Customer): Cliente con datos actualizados
            
        Returns:
            Customer: Cliente actualizado
        """
        pass
    
    @abstractmethod
    def delete(self, customer_id: str) -> bool:
        """
        Elimina un cliente del repositorio.
        
        Args:
            customer_id (str): ID del cliente a eliminar
            
        Returns:
            bool: True si se eliminó exitosamente, False en caso contrario
        """
        pass
    
    @abstractmethod
    def count_total_customers(self) -> int:
        """
        Cuenta el total de clientes en el repositorio.
        
        Returns:
            int: Número total de clientes
        """
        pass
    
    @abstractmethod
    def count_churned_customers(self) -> int:
        """
        Cuenta el total de clientes que han abandonado el servicio.
        
        Returns:
            int: Número de clientes con churn
        """
        pass


class AbstractCustomerRepository(ABC):
    """
    Clase abstracta alternativa al protocolo para implementaciones que prefieran herencia.
    """
    
    @abstractmethod
    def save(self, customer: Customer) -> Customer:
        pass
    
    @abstractmethod
    def find_by_id(self, customer_id: str) -> Optional[Customer]:
        pass
    
    @abstractmethod
    def find_all(self, limit: Optional[int] = None) -> List[Customer]:
        pass
    
    @abstractmethod
    def find_churned_customers(self) -> List[Customer]:
        pass
    
    @abstractmethod
    def find_active_customers(self) -> List[Customer]:
        pass
    
    @abstractmethod
    def update(self, customer: Customer) -> Customer:
        pass
    
    @abstractmethod
    def delete(self, customer_id: str) -> bool:
        pass
    
    @abstractmethod
    def count_total_customers(self) -> int:
        pass
    
    @abstractmethod
    def count_churned_customers(self) -> int:
        pass