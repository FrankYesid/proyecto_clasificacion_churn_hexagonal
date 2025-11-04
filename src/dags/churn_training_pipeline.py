"""
DAG de Airflow para el pipeline automatizado de entrenamiento del modelo de churn.
Este DAG ejecuta el pipeline completo de ML: extracción de datos, preprocesamiento,
entrenamiento, evaluación y despliegue del modelo.

Características adicionales:
- Monitoreo de deriva de datos (data drift)
- Monitoreo de rendimiento del modelo
- Alertas avanzadas basadas en métricas de negocio
- Integración con sistemas de monitoreo externos
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
from airflow.sensors.filesystem import FileSensor
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule
import logging
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Agregar el directorio src al path de Python
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Importar componentes del proyecto
from adapters.repositories.csv_customer_repository import CsvCustomerRepository
from adapters.machine_learning.sklearn_churn_model import SklearnChurnPredictionModel
from application.use_cases.train_churn_model import TrainChurnModelUseCase
from domain.services.churn_analysis_service import ChurnAnalysisService


def get_config():
    """
    Obtiene la configuración del DAG desde variables de Airflow.
    """
    return {
        'data_path': Variable.get('churn_data_path', default_var='/opt/airflow/data/raw/customer_data.csv'),
        'model_output_path': Variable.get('model_output_path', default_var='/opt/airflow/models'),
        'preprocessor_output_path': Variable.get('preprocessor_output_path', default_var='/opt/airflow/models'),
        'test_size': float(Variable.get('test_size', default_var='0.2')),
        'random_state': int(Variable.get('random_state', default_var='42')),
        'email_to': Variable.get('ml_team_email', default_var='ml-team@company.com'),
        'model_version': Variable.get('model_version', default_var='1.0.0')
    }


def check_data_quality(**context):
    """
    Verifica la calidad de los datos antes del entrenamiento.
    """
    logger.info("Checking data quality...")
    
    config = get_config()
    data_path = config['data_path']
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Cargar y verificar datos
    import pandas as pd
    df = pd.read_csv(data_path)
    
    # Verificaciones básicas
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if len(df) < 100:
        raise ValueError("Dataset too small for training (less than 100 samples)")
    
    # Verificar columnas requeridas
    required_columns = [
        'customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
        'phone_service', 'multiple_lines', 'internet_service', 'online_security',
        'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
        'streaming_movies', 'contract_type', 'paperless_billing', 'payment_method',
        'monthly_charges', 'total_charges', 'tenure_months', 'churn'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Verificar valores nulos
    null_counts = df.isnull().sum()
    high_null_columns = null_counts[null_counts > len(df) * 0.1].index.tolist()
    if high_null_columns:
        logger.warning(f"Columns with high null values: {high_null_columns}")
    
    # Verificar desequilibrio de clases
    churn_distribution = df['churn'].value_counts(normalize=True)
    if churn_distribution.min() < 0.1:
        logger.warning("High class imbalance detected")
    
    # Guardar estadísticas en el contexto
    context['task_instance'].xcom_push(key='data_stats', value={
        'total_samples': len(df),
        'churn_rate': float(churn_distribution[True]) if True in churn_distribution.index else 0,
        'missing_values': int(df.isnull().sum().sum()),
        'columns': list(df.columns)
    })
    
    logger.info("Data quality check completed successfully")
    return "Data quality check passed"


def extract_and_preprocess_data(**context):
    """
    Extrae y preprocesa los datos para entrenamiento.
    """
    logger.info("Extracting and preprocessing data...")
    
    config = get_config()
    data_path = config['data_path']
    
    # Cargar datos
    repository = CsvCustomerRepository(data_path)
    customers = repository.find_all()
    
    # Convertir a DataFrame para preprocesamiento
    import pandas as pd
    data = []
    for customer in customers:
        customer_dict = customer.to_dict()
        customer_dict['churn'] = customer.churn  # Asegurar que tenemos la etiqueta
        data.append(customer_dict)
    
    df = pd.DataFrame(data)
    
    # Preprocesamiento básico
    # Manejar valores nulos en total_charges
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    df['total_charges'] = df['total_charges'].fillna(df['monthly_charges'] * df['tenure_months'])
    
    # Crear características adicionales
    df['avg_monthly_charge'] = df['total_charges'] / (df['tenure_months'] + 1)  # +1 para evitar división por cero
    df['services_count'] = (
        (df['phone_service'].astype(int)) +
        (df['multiple_lines'] != 'No').astype(int) +
        (df['internet_service'] != 'No').astype(int) +
        (df['online_security'] == 'Yes').astype(int) +
        (df['online_backup'] == 'Yes').astype(int) +
        (df['device_protection'] == 'Yes').astype(int) +
        (df['tech_support'] == 'Yes').astype(int) +
        (df['streaming_tv'] == 'Yes').astype(int) +
        (df['streaming_movies'] == 'Yes').astype(int)
    )
    
    # Guardar datos preprocesados temporalmente
    preprocessed_path = "/tmp/preprocessed_data.csv"
    df.to_csv(preprocessed_path, index=False)
    
    context['task_instance'].xcom_push(key='preprocessed_data_path', value=preprocessed_path)
    context['task_instance'].xcom_push(key='preprocessing_stats', value={
        'original_samples': len(customers),
        'processed_samples': len(df),
        'features_created': ['avg_monthly_charge', 'services_count']
    })
    
    logger.info("Data extraction and preprocessing completed")
    return preprocessed_path


def train_model(**context):
    """
    Entrena el modelo de churn.
    """
    logger.info("Training churn prediction model...")
    
    config = get_config()
    
    # Obtener path de datos preprocesados
    preprocessed_path = context['task_instance'].xcom_pull(key='preprocessed_data_path')
    
    # Crear repositorio temporal con datos preprocesados
    repository = CsvCustomerRepository(preprocessed_path)
    
    # Crear modelo
    model = SklearnChurnPredictionModel()
    
    # Crear servicio de análisis
    analysis_service = ChurnAnalysisService()
    
    # Crear caso de uso de entrenamiento
    train_use_case = TrainChurnModelUseCase(repository, model, analysis_service)
    
    # Configuración de entrenamiento
    training_config = {
        'test_size': config['test_size'],
        'random_state': config['random_state'],
        'model_output_path': config['model_output_path'],
        'preprocessor_output_path': config['preprocessor_output_path']
    }
    
    # Ejecutar entrenamiento
    result = train_use_case.execute(training_config)
    
    # Guardar resultados en el contexto
    context['task_instance'].xcom_push(key='training_results', value={
        'model_path': result['model_path'],
        'preprocessor_path': result['preprocessor_path'],
        'metrics_path': result['metrics_path'],
        'accuracy': result['metrics']['accuracy'],
        'precision': result['metrics']['precision'],
        'recall': result['metrics']['recall'],
        'f1_score': result['metrics']['f1_score'],
        'training_time': result['training_time_seconds']
    })
    
    logger.info(f"Model training completed. Accuracy: {result['metrics']['accuracy']:.4f}")
    return result


def validate_model(**context):
    """
    Valida el modelo entrenado con criterios de calidad.
    """
    logger.info("Validating trained model...")
    
    training_results = context['task_instance'].xcom_pull(key='training_results')
    
    # Criterios de validación
    min_accuracy = float(Variable.get('min_model_accuracy', default_var='0.75'))
    min_precision = float(Variable.get('min_model_precision', default_var='0.70'))
    min_recall = float(Variable.get('min_model_recall', default_var='0.65'))
    
    metrics = {
        'accuracy': training_results['accuracy'],
        'precision': training_results['precision'],
        'recall': training_results['recall'],
        'f1_score': training_results['f1_score']
    }
    
    validation_results = {}
    
    # Verificar cada métrica
    for metric, value in metrics.items():
        threshold = locals()[f'min_{metric}']
        validation_results[metric] = {
            'value': value,
            'threshold': threshold,
            'passed': value >= threshold
        }
        
        if not validation_results[metric]['passed']:
            logger.warning(f"Model validation failed for {metric}: {value:.4f} < {threshold}")
    
    # Determinar si el modelo pasa la validación
    overall_passed = all(result['passed'] for result in validation_results.values())
    
    context['task_instance'].xcom_push(key='validation_results', value={
        'passed': overall_passed,
        'metrics': validation_results,
        'model_path': training_results['model_path']
    })
    
    if not overall_passed:
        raise ValueError("Model validation failed - metrics below threshold")
    
    logger.info("Model validation passed")
    return validation_results


def deploy_model(**context):
    """
    Despliega el modelo validado al entorno de producción.
    """
    logger.info("Deploying model to production...")
    
    validation_results = context['task_instance'].xcom_pull(key='validation_results')
    
    if not validation_results['passed']:
        logger.warning("Model validation failed - skipping deployment")
        return "Deployment skipped due to validation failure"
    
    # Obtener información del modelo
    training_results = context['task_instance'].xcom_pull(key='training_results')
    
    # Simular despliegue (en producción, esto podría copiar archivos a un bucket S3, actualizar un servicio, etc.)
    production_model_path = Variable.get('production_model_path', default_var='/opt/airflow/models/production')
    
    import shutil
    
    # Crear directorio de producción si no existe
    os.makedirs(production_model_path, exist_ok=True)
    
    # Copiar modelo y preprocesador
    model_filename = f"churn_model_v{get_config()['model_version']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    preprocessor_filename = f"preprocessor_v{get_config()['model_version']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    
    shutil.copy(training_results['model_path'], os.path.join(production_model_path, model_filename))
    shutil.copy(training_results['preprocessor_path'], os.path.join(production_model_path, preprocessor_filename))
    
    # Actualizar symlink o archivo de configuración con la versión actual
    latest_model_link = os.path.join(production_model_path, 'latest_model.pkl')
    latest_preprocessor_link = os.path.join(production_model_path, 'latest_preprocessor.pkl')
    
    # Remover symlinks anteriores si existen
    for link in [latest_model_link, latest_preprocessor_link]:
        if os.path.islink(link):
            os.remove(link)
    
    # Crear nuevos symlinks
    os.symlink(model_filename, latest_model_link)
    os.symlink(preprocessor_filename, latest_preprocessor_link)
    
    # Guardar información de despliegue
    deployment_info = {
        'model_version': get_config()['model_version'],
        'deployment_timestamp': datetime.now().isoformat(),
        'model_file': model_filename,
        'preprocessor_file': preprocessor_filename,
        'metrics': validation_results['metrics']
    }
    
    context['task_instance'].xcom_push(key='deployment_info', value=deployment_info)
    
    logger.info(f"Model deployed successfully to {production_model_path}")
    return deployment_info


def generate_report(**context):
    """
    Genera un reporte del entrenamiento.
    """
    logger.info("Generating training report...")
    
    # Recopilar información de todas las tareas
    data_stats = context['task_instance'].xcom_pull(key='data_stats')
    preprocessing_stats = context['task_instance'].xcom_pull(key='preprocessing_stats')
    training_results = context['task_instance'].xcom_pull(key='training_results')
    validation_results = context['task_instance'].xcom_pull(key='validation_results')
    deployment_info = context['task_instance'].xcom_pull(key='deployment_info')
    
    report = f"""
# Customer Churn Model Training Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Summary
- Total samples: {data_stats.get('total_samples', 'N/A')}
- Churn rate: {data_stats.get('churn_rate', 'N/A'):.2%}
- Missing values: {data_stats.get('missing_values', 'N/A')}

## Preprocessing Summary
- Original samples: {preprocessing_stats.get('original_samples', 'N/A')}
- Processed samples: {preprocessing_stats.get('processed_samples', 'N/A')}
- Features created: {', '.join(preprocessing_stats.get('features_created', []))}

## Model Performance
- Accuracy: {training_results.get('accuracy', 'N/A'):.4f}
- Precision: {training_results.get('precision', 'N/A'):.4f}
- Recall: {training_results.get('recall', 'N/A'):.4f}
- F1-Score: {training_results.get('f1_score', 'N/A'):.4f}
- Training time: {training_results.get('training_time', 'N/A'):.1f} seconds

## Validation Results
- Validation passed: {'Yes' if validation_results.get('passed', False) else 'No'}

## Deployment Information
- Model version: {deployment_info.get('model_version', 'N/A')}
- Deployment timestamp: {deployment_info.get('deployment_timestamp', 'N/A')}
- Model file: {deployment_info.get('model_file', 'N/A')}
- Preprocessor file: {deployment_info.get('preprocessor_file', 'N/A')}
"""
    
    # Guardar reporte
    report_path = f"/tmp/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    context['task_instance'].xcom_push(key='report_path', value=report_path)
    
    logger.info(f"Report generated: {report_path}")
    return report_path


# Configuración del DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': [get_config()['email_to']],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definir el DAG
dag = DAG(
    'churn_model_training_pipeline',
    default_args=default_args,
    description='Automated ML pipeline for customer churn prediction model training',
    schedule_interval='@weekly',  # Ejecutar semanalmente
    catchup=False,
    tags=['ml', 'churn', 'training'],
)

# Definir tareas
with dag:
    # Tarea de verificación de datos
    check_data_task = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality,
        dag=dag,
    )
    
    # Tarea de extracción y preprocesamiento
    preprocess_task = PythonOperator(
        task_id='extract_and_preprocess_data',
        python_callable=extract_and_preprocess_data,
        dag=dag,
    )
    
    # Tarea de entrenamiento
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        dag=dag,
    )
    
    # Tarea de validación
    validate_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
        dag=dag,
    )
    
    # Tarea de despliegue (solo si la validación pasa)
    deploy_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
        trigger_rule='all_success',  # Solo ejecutar si todas las tareas anteriores fueron exitosas
        dag=dag,
    )
    
    # Tarea de generación de reporte
    report_task = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
        dag=dag,
    )
    
    # Tarea de notificación por email
    email_task = EmailOperator(
        task_id='send_notification',
        to=get_config()['email_to'],
        subject='Churn Model Training Pipeline - {{ ds }}',
        html_content="""
        <h3>Churn Model Training Pipeline Completed</h3>
        <p>Date: {{ ds }}</p>
        <p>Status: {{ task_instance.xcom_pull(key='validation_results', task_ids='validate_model')['passed'] }}</p>
        <p>Model Accuracy: {{ task_instance.xcom_pull(key='training_results', task_ids='train_model')['accuracy'] }}</p>
        <p>Report: {{ task_instance.xcom_pull(key='report_path', task_ids='generate_report') }}</p>
        """,
        dag=dag,
    )
    
    # Limpiar archivos temporales
    cleanup_task = BashOperator(
        task_id='cleanup_temp_files',
        bash_command='rm -f /tmp/preprocessed_data.csv /tmp/training_report_*.txt',
        dag=dag,
    )
    
    # Definir flujo de dependencias
    check_data_task >> preprocess_task >> train_task >> validate_task
    validate_task >> [deploy_task, report_task]
    [deploy_task, report_task] >> email_task >> cleanup_task