"""
Adaptador de repositorio para datos de clientes.
Implementa la interfaz CustomerRepository usando CSV como fuente de datos.
"""

import os
import logging
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime

from ...domain.entities.customer import Customer
from ...domain.repositories.customer_repository import CustomerRepository


logger = logging.getLogger(__name__)


class CsvCustomerRepository(CustomerRepository):
    """
    Repositorio de clientes que utiliza archivos CSV como fuente de datos.
    
    Esta implementación permite:
    - Cargar datos desde archivos CSV
    - Guardar datos en formato CSV
    - Realizar consultas básicas sobre clientes
    - Filtrar por estado de churn
    """
    
    def __init__(self, csv_file_path: str):
        """
        Inicializa el repositorio con la ruta al archivo CSV.
        
        Args:
            csv_file_path (str): Ruta al archivo CSV con datos de clientes
        """
        self.csv_file_path = csv_file_path
        self._data: Optional[pd.DataFrame] = None
        self._load_data()
    
    def _load_data(self):
        """
        Carga los datos desde el archivo CSV.
        """
        if not os.path.exists(self.csv_file_path):
            logger.warning(f"Archivo CSV no encontrado: {self.csv_file_path}")
            self._data = pd.DataFrame()
            return
        
        try:
            self._data = pd.read_csv(self.csv_file_path)
            logger.info(f"Datos cargados exitosamente: {len(self._data)} registros")
            
            # Limpiar nombres de columnas
            self._data.columns = self._data.columns.str.lower().str.replace(' ', '_')
            
            # Asegurar tipos de datos correctos
            self._ensure_data_types()
            
        except Exception as e:
            logger.error(f"Error cargando datos CSV: {str(e)}")
            self._data = pd.DataFrame()
            raise
    
    def _ensure_data_types(self):
        """
        Asegura que los tipos de datos sean correctos.
        """
        if self._data.empty:
            return
        
        # Convertir columnas booleanas
        bool_columns = ['senior_citizen', 'partner', 'dependents', 'phone_service', 'paperless_billing']
        for col in bool_columns:
            if col in self._data.columns:
                self._data[col] = self._data[col].astype(bool)
        
        # Convertir columnas numéricas
        numeric_columns = ['tenure_months', 'monthly_charges', 'total_charges']
        for col in numeric_columns:
            if col in self._data.columns:
                self._data[col] = pd.to_numeric(self._data[col], errors='coerce')
        
        # Convertir churn a booleano si existe
        if 'churn' in self._data.columns:
            self._data['churn'] = self._data['churn'].map({'Yes': True, 'No': False})
    
    def save(self, customer: Customer) -> Customer:
        """
        Guarda un cliente en el repositorio.
        
        Args:
            customer (Customer): Cliente a guardar
            
        Returns:
            Customer: Cliente guardado con ID asignado
        """
        # Convertir cliente a diccionario
        customer_data = customer.to_dict()
        
        # Si es un cliente nuevo, agregar al final
        if customer.customer_id is None:
            customer_data['customer_id'] = self._generate_new_id()
            customer = Customer.from_dict(customer_data)
        
        # Agregar o actualizar en el DataFrame
        customer_series = pd.Series(customer_data)
        
        if customer.customer_id in self._data['customer_id'].values:
            # Actualizar cliente existente
            mask = self._data['customer_id'] == customer.customer_id
            for column, value in customer_series.items():
                if column in self._data.columns:
                    self._data.loc[mask, column] = value
            logger.info(f"Cliente actualizado: {customer.customer_id}")
        else:
            # Agregar nuevo cliente
            new_row = pd.DataFrame([customer_series])
            self._data = pd.concat([self._data, new_row], ignore_index=True)
            logger.info(f"Nuevo cliente agregado: {customer.customer_id}")
        
        # Guardar cambios en archivo
        self._save_to_file()
        
        return customer
    
    def find_by_id(self, customer_id: str) -> Optional[Customer]:
        """
        Busca un cliente por su ID.
        
        Args:
            customer_id (str): ID del cliente
            
        Returns:
            Optional[Customer]: Cliente encontrado o None
        """
        if self._data.empty or 'customer_id' not in self._data.columns:
            return None
        
        customer_data = self._data[self._data['customer_id'] == customer_id]
        
        if customer_data.empty:
            return None
        
        # Convertir la fila a diccionario y luego a Customer
        customer_dict = customer_data.iloc[0].to_dict()
        return Customer.from_dict(customer_dict)
    
    def find_all(self) -> List[Customer]:
        """
        Obtiene todos los clientes del repositorio.
        
        Returns:
            List[Customer]: Lista de todos los clientes
        """
        if self._data.empty:
            return []
        
        customers = []
        for _, row in self._data.iterrows():
            try:
                customer = Customer.from_dict(row.to_dict())
                customers.append(customer)
            except Exception as e:
                logger.warning(f"Error convirtiendo fila a Customer: {str(e)}")
                continue
        
        return customers
    
    def find_churned_customers(self) -> List[Customer]:
        """
        Obtiene todos los clientes que han cancelado el servicio.
        
        Returns:
            List[Customer]: Lista de clientes con churn
        """
        if self._data.empty or 'churn' not in self._data.columns:
            return []
        
        churned_data = self._data[self._data['churn'] == True]
        
        customers = []
        for _, row in churned_data.iterrows():
            try:
                customer = Customer.from_dict(row.to_dict())
                customers.append(customer)
            except Exception as e:
                logger.warning(f"Error convirtiendo cliente churned a Customer: {str(e)}")
                continue
        
        return customers
    
    def find_active_customers(self) -> List[Customer]:
        """
        Obtiene todos los clientes activos (sin churn).
        
        Returns:
            List[Customer]: Lista de clientes activos
        """
        if self._data.empty or 'churn' not in self._data.columns:
            return []
        
        active_data = self._data[self._data['churn'] == False]
        
        customers = []
        for _, row in active_data.iterrows():
            try:
                customer = Customer.from_dict(row.to_dict())
                customers.append(customer)
            except Exception as e:
                logger.warning(f"Error convirtiendo cliente activo a Customer: {str(e)}")
                continue
        
        return customers
    
    def update(self, customer: Customer) -> Optional[Customer]:
        """
        Actualiza un cliente existente.
        
        Args:
            customer (Customer): Cliente con datos actualizados
            
        Returns:
            Optional[Customer]: Cliente actualizado o None si no existe
        """
        if customer.customer_id is None:
            return None
        
        existing = self.find_by_id(customer.customer_id)
        if existing is None:
            return None
        
        return self.save(customer)
    
    def delete(self, customer_id: str) -> bool:
        """
        Elimina un cliente del repositorio.
        
        Args:
            customer_id (str): ID del cliente a eliminar
            
        Returns:
            bool: True si se eliminó exitosamente, False si no existe
        """
        if self._data.empty or 'customer_id' not in self._data.columns:
            return False
        
        if customer_id not in self._data['customer_id'].values:
            return False
        
        # Eliminar el cliente
        self._data = self._data[self._data['customer_id'] != customer_id]
        
        # Guardar cambios
        self._save_to_file()
        
        logger.info(f"Cliente eliminado: {customer_id}")
        return True
    
    def count_total_customers(self) -> int:
        """
        Cuenta el total de clientes en el repositorio.
        
        Returns:
            int: Número total de clientes
        """
        return len(self._data) if not self._data.empty else 0
    
    def count_churned_customers(self) -> int:
        """
        Cuenta el número de clientes que han cancelado el servicio.
        
        Returns:
            int: Número de clientes con churn
        """
        if self._data.empty or 'churn' not in self._data.columns:
            return 0
        
        return len(self._data[self._data['churn'] == True])
    
    def _generate_new_id(self) -> str:
        """
        Genera un nuevo ID único para un cliente.
        
        Returns:
            str: Nuevo ID generado
        """
        if self._data.empty or 'customer_id' not in self._data.columns:
            return "CUST-0001"
        
        # Encontrar el ID numérico más alto
        existing_ids = self._data['customer_id'].tolist()
        max_num = 0
        
        for customer_id in existing_ids:
            try:
                # Extraer número del ID (formato: CUST-XXXX)
                if customer_id.startswith('CUST-'):
                    num = int(customer_id.split('-')[1])
                    max_num = max(max_num, num)
            except (ValueError, IndexError):
                continue
        
        new_num = max_num + 1
        return f"CUST-{new_num:04d}"
    
    def _save_to_file(self):
        """
        Guarda los datos actuales en el archivo CSV.
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.csv_file_path), exist_ok=True)
            
            # Guardar datos
            self._data.to_csv(self.csv_file_path, index=False)
            logger.info(f"Datos guardados en archivo: {self.csv_file_path}")
            
        except Exception as e:
            logger.error(f"Error guardando datos en archivo: {str(e)}")
            raise
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas resumidas de los datos.
        
        Returns:
            Dict[str, Any]: Estadísticas del repositorio
        """
        if self._data.empty:
            return {
                'total_customers': 0,
                'churned_customers': 0,
                'churn_rate': 0.0,
                'average_tenure': 0.0,
                'average_monthly_charges': 0.0
            }
        
        total_customers = self.count_total_customers()
        churned_customers = self.count_churned_customers()
        churn_rate = churned_customers / total_customers if total_customers > 0 else 0.0
        
        stats = {
            'total_customers': total_customers,
            'churned_customers': churned_customers,
            'churn_rate': churn_rate,
            'average_tenure': self._data['tenure_months'].mean() if 'tenure_months' in self._data.columns else 0.0,
            'average_monthly_charges': self._data['monthly_charges'].mean() if 'monthly_charges' in self._data.columns else 0.0
        }
        
        return stats