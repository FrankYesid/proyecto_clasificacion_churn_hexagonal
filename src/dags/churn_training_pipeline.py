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
from scipy import stats

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


def detect_data_drift(**context):
    """
    Detecta deriva de datos comparando distribuciones actuales con referencia histórica.
    """
    logger.info("Detecting data drift...")
    
    config = get_config()
    data_path = config['data_path']
    
    # Cargar datos actuales
    df_current = pd.read_csv(data_path)
    
    # Intentar cargar datos de referencia histórica
    reference_path = "/opt/airflow/data/reference/reference_data.csv"
    drift_detected = False
    drift_metrics = {}
    drift_severity = 'NONE'
    
    if os.path.exists(reference_path):
        df_reference = pd.read_csv(reference_path)
        
        # Columnas numéricas para análisis
        numeric_columns = ['monthly_charges', 'total_charges', 'tenure_months']
        
        # Umbrales configurables para drift
        drift_threshold_medium = float(Variable.get('drift_threshold_medium', default_var='2.0'))
        drift_threshold_high = float(Variable.get('drift_threshold_high', default_var='3.0'))
        drift_threshold_critical = float(Variable.get('drift_threshold_critical', default_var='5.0'))
        
        for col in numeric_columns:
            if col in df_current.columns and col in df_reference.columns:
                # Calcular estadísticas
                current_mean = df_current[col].mean()
                reference_mean = df_reference[col].mean()
                current_std = df_current[col].std()
                reference_std = df_reference[col].std()
                
                # Detectar drift usando cambio significativo en media (desviaciones estándar)
                std_diff = abs(current_mean - reference_mean) / reference_std if reference_std > 0 else 0
                
                # Calcular Kolmogorov-Smirnov test para distribución completa
                try:
                    ks_statistic, ks_pvalue = stats.ks_2samp(df_current[col].dropna(), df_reference[col].dropna())
                except:
                    ks_statistic, ks_pvalue = 0, 1
                
                # Determinar severidad
                col_severity = 'NONE'
                if std_diff >= drift_threshold_critical:
                    col_severity = 'CRITICAL'
                    drift_detected = True
                elif std_diff >= drift_threshold_high:
                    col_severity = 'HIGH'
                    drift_detected = True
                elif std_diff >= drift_threshold_medium:
                    col_severity = 'MEDIUM'
                    drift_detected = True
                elif ks_pvalue < 0.01:  # Distribución significativamente diferente
                    col_severity = 'MEDIUM'
                    drift_detected = True
                
                drift_metrics[col] = {
                    'current_mean': float(current_mean),
                    'reference_mean': float(reference_mean),
                    'current_std': float(current_std),
                    'reference_std': float(reference_std),
                    'mean_difference': float(abs(current_mean - reference_mean)),
                    'std_differences': float(std_diff),
                    'ks_statistic': float(ks_statistic),
                    'ks_pvalue': float(ks_pvalue),
                    'drift_detected': std_diff >= drift_threshold_medium or ks_pvalue < 0.01,
                    'severity': col_severity
                }
                
                # Actualizar severidad global
                if col_severity == 'CRITICAL':
                    drift_severity = 'CRITICAL'
                elif col_severity == 'HIGH' and drift_severity not in ['CRITICAL']:
                    drift_severity = 'HIGH'
                elif col_severity == 'MEDIUM' and drift_severity not in ['CRITICAL', 'HIGH']:
                    drift_severity = 'MEDIUM'
                
                if drift_metrics[col]['drift_detected']:
                    logger.warning(f"Data drift detected in {col}: {std_diff:.2f} std deviations, KS p-value: {ks_pvalue:.4f}, Severity: {col_severity}")
        
        # Análisis de distribución de clases
        current_churn_dist = df_current['churn'].value_counts(normalize=True)
        reference_churn_dist = df_reference['churn'].value_counts(normalize=True)
        
        churn_dist_threshold = float(Variable.get('churn_dist_drift_threshold', default_var='0.05'))
        
        if True in current_churn_dist.index and True in reference_churn_dist.index:
            churn_diff = abs(current_churn_dist[True] - reference_churn_dist[True])
            churn_drift_detected = churn_diff > churn_dist_threshold
            
            drift_metrics['churn_distribution'] = {
                'current_churn_rate': float(current_churn_dist[True]),
                'reference_churn_rate': float(reference_churn_dist[True]),
                'difference': float(churn_diff),
                'drift_detected': churn_drift_detected,
                'severity': 'HIGH' if churn_diff > 0.10 else 'MEDIUM' if churn_drift_detected else 'NONE'
            }
            
            if churn_drift_detected:
                drift_detected = True
                if drift_metrics['churn_distribution']['severity'] == 'HIGH':
                    drift_severity = 'HIGH' if drift_severity not in ['CRITICAL'] else drift_severity
                else:
                    drift_severity = 'MEDIUM' if drift_severity not in ['CRITICAL', 'HIGH'] else drift_severity
                logger.warning(f"Churn distribution drift detected: {churn_diff:.2%}, Severity: {drift_metrics['churn_distribution']['severity']}")
        
        # Análisis de características categóricas
        categorical_columns = ['gender', 'partner', 'dependents', 'contract_type', 'payment_method']
        for col in categorical_columns:
            if col in df_current.columns and col in df_reference.columns:
                current_dist = df_current[col].value_counts(normalize=True)
                reference_dist = df_reference[col].value_counts(normalize=True)
                
                # Calcular diferencia total en distribución
                all_categories = set(current_dist.index) | set(reference_dist.index)
                total_diff = sum(abs(current_dist.get(cat, 0) - reference_dist.get(cat, 0)) for cat in all_categories) / 2
                
                if total_diff > 0.10:  # 10% diferencia total
                    drift_metrics[f'{col}_distribution'] = {
                        'total_variation_distance': float(total_diff),
                        'drift_detected': True,
                        'severity': 'MEDIUM' if total_diff > 0.20 else 'LOW'
                    }
                    drift_detected = True
                    logger.warning(f"Categorical drift detected in {col}: {total_diff:.2%}")
    
    else:
        # Si no hay referencia, usar primera ejecución como referencia
        logger.info("No reference data found. Creating reference from current data...")
        os.makedirs(os.path.dirname(reference_path), exist_ok=True)
        df_current.to_csv(reference_path, index=False)
        logger.info(f"Reference data saved to {reference_path}")
    
    # Guardar resultados
    context['task_instance'].xcom_push(key='drift_detection', value={
        'drift_detected': drift_detected,
        'severity': drift_severity,
        'metrics': drift_metrics,
        'timestamp': datetime.now().isoformat()
    })
    
    # Si se detecta drift significativo, podríamos querer alertar o incluso detener el pipeline
    drift_action = Variable.get('drift_action_on_detection', default_var='warn')  # warn, stop, ignore
    
    if drift_detected:
        if drift_action == 'stop' and drift_severity in ['HIGH', 'CRITICAL']:
            raise ValueError(f"Significant data drift detected (Severity: {drift_severity}) - stopping pipeline execution")
        elif drift_action == 'warn':
            logger.warning(f"Data drift detected (Severity: {drift_severity}) - continuing with caution")
        elif drift_action == 'stop':
            logger.warning(f"Data drift detected but action is set to 'stop' only for HIGH/CRITICAL - continuing")
    
    logger.info(f"Data drift detection completed. Drift detected: {drift_detected}, Severity: {drift_severity}")
    return f"Drift detection completed - Severity: {drift_severity}"


def monitor_model_performance(**context):
    """
    Monitorea el rendimiento del modelo en producción y detecta degradación.
    """
    logger.info("Monitoring model performance...")
    
    # Intentar cargar métricas de rendimiento recientes
    performance_log_path = "/opt/airflow/models/performance_log.json"
    performance_degradation = False
    performance_metrics = {}
    degradation_severity = 'NONE'
    
    # Umbrales configurables
    degradation_threshold_low = float(Variable.get('performance_degradation_threshold_low', default_var='0.05'))
    degradation_threshold_medium = float(Variable.get('performance_degradation_threshold_medium', default_var='0.10'))
    degradation_threshold_high = float(Variable.get('performance_degradation_threshold_high', default_var='0.20'))
    min_data_points = int(Variable.get('performance_min_data_points', default_var='5'))
    
    if os.path.exists(performance_log_path):
        try:
            with open(performance_log_path, 'r') as f:
                performance_log = json.load(f)
            
            # Analizar tendencias de métricas clave
            if len(performance_log) >= min_data_points:
                window_size = min(5, len(performance_log) // 2)
                recent_metrics = performance_log[-window_size:]  # Ventana reciente
                
                # Calcular promedios móviles y tendencias
                recent_accuracy = np.mean([m.get('accuracy', 0) for m in recent_metrics if m.get('accuracy')])
                baseline_size = min(len(performance_log) - window_size, window_size * 2)
                baseline_metrics = performance_log[-baseline_size-window_size:-window_size] if baseline_size > 0 else performance_log[:-window_size]
                baseline_accuracy = np.mean([m.get('accuracy', 0) for m in baseline_metrics if m.get('accuracy')]) if baseline_metrics else recent_accuracy
                
                recent_precision = np.mean([m.get('precision', 0) for m in recent_metrics if m.get('precision')])
                baseline_precision = np.mean([m.get('precision', 0) for m in baseline_metrics if m.get('precision')]) if baseline_metrics else recent_precision
                
                recent_recall = np.mean([m.get('recall', 0) for m in recent_metrics if m.get('recall')])
                baseline_recall = np.mean([m.get('recall', 0) for m in baseline_metrics if m.get('recall')]) if baseline_metrics else recent_recall
                
                recent_f1 = np.mean([m.get('f1_score', 0) for m in recent_metrics if m.get('f1_score')])
                baseline_f1 = np.mean([m.get('f1_score', 0) for m in baseline_metrics if m.get('f1_score')]) if baseline_metrics else recent_f1
                
                # Detectar degradación (caída porcentual)
                accuracy_degradation = (baseline_accuracy - recent_accuracy) / baseline_accuracy if baseline_accuracy > 0 else 0
                precision_degradation = (baseline_precision - recent_precision) / baseline_precision if baseline_precision > 0 else 0
                recall_degradation = (baseline_recall - recent_recall) / baseline_recall if baseline_recall > 0 else 0
                f1_degradation = (baseline_f1 - recent_f1) / baseline_f1 if baseline_f1 > 0 else 0
                
                # Determinar severidad basada en la peor degradación
                max_degradation = max(accuracy_degradation, precision_degradation, recall_degradation, f1_degradation)
                
                if max_degradation >= degradation_threshold_high:
                    degradation_severity = 'CRITICAL'
                    performance_degradation = True
                elif max_degradation >= degradation_threshold_medium:
                    degradation_severity = 'HIGH'
                    performance_degradation = True
                elif max_degradation >= degradation_threshold_low:
                    degradation_severity = 'MEDIUM'
                    performance_degradation = True
                
                # Análisis de tendencia (regresión lineal)
                if len(recent_metrics) >= 3:
                    accuracy_values = [m.get('accuracy', 0) for m in recent_metrics if m.get('accuracy')]
                    if len(accuracy_values) >= 3:
                        x = np.arange(len(accuracy_values))
                        slope = np.polyfit(x, accuracy_values, 1)[0] if len(accuracy_values) > 1 else 0
                        trend = 'decreasing' if slope < -0.001 else 'increasing' if slope > 0.001 else 'stable'
                    else:
                        trend = 'unknown'
                else:
                    trend = 'insufficient_data'
                
                performance_metrics = {
                    'baseline_accuracy': float(baseline_accuracy),
                    'recent_accuracy': float(recent_accuracy),
                    'accuracy_degradation': float(accuracy_degradation),
                    'baseline_precision': float(baseline_precision),
                    'recent_precision': float(recent_precision),
                    'precision_degradation': float(precision_degradation),
                    'baseline_recall': float(baseline_recall),
                    'recent_recall': float(recent_recall),
                    'recall_degradation': float(recall_degradation),
                    'baseline_f1': float(baseline_f1),
                    'recent_f1': float(recent_f1),
                    'f1_degradation': float(f1_degradation),
                    'max_degradation': float(max_degradation),
                    'trend': trend,
                    'data_points': len(performance_log),
                    'window_size': window_size
                }
                
                if performance_degradation:
                    logger.warning(f"Model performance degradation detected: Max degradation {max_degradation:.1%}, Severity: {degradation_severity}, Trend: {trend}")
                
        except Exception as e:
            logger.error(f"Error monitoring model performance: {str(e)}")
            # No fallar el pipeline por problemas de monitoreo
    else:
        logger.info("No performance log found. This may be the first run or performance logging is not enabled.")
    
    # Guardar métricas actuales del entrenamiento si están disponibles
    training_results = context['task_instance'].xcom_pull(key='training_results', task_ids='train_model')
    if training_results and os.path.exists(performance_log_path):
        try:
            with open(performance_log_path, 'r') as f:
                performance_log = json.load(f)
            
            # Agregar métricas actuales
            new_entry = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': training_results.get('accuracy'),
                'precision': training_results.get('precision'),
                'recall': training_results.get('recall'),
                'f1_score': training_results.get('f1_score'),
                'model_version': get_config().get('model_version', '1.0.0'),
                'dag_run_id': context['run_id']
            }
            performance_log.append(new_entry)
            
            # Mantener solo últimos 50 registros
            performance_log = performance_log[-50:]
            
            with open(performance_log_path, 'w') as f:
                json.dump(performance_log, f, indent=2)
            
            logger.info("Performance metrics logged successfully")
        except Exception as e:
            logger.error(f"Error logging performance metrics: {str(e)}")
    elif training_results:
        # Crear nuevo archivo de log
        try:
            os.makedirs(os.path.dirname(performance_log_path), exist_ok=True)
            new_entry = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': training_results.get('accuracy'),
                'precision': training_results.get('precision'),
                'recall': training_results.get('recall'),
                'f1_score': training_results.get('f1_score'),
                'model_version': get_config().get('model_version', '1.0.0'),
                'dag_run_id': context['run_id']
            }
            with open(performance_log_path, 'w') as f:
                json.dump([new_entry], f, indent=2)
            logger.info("Created new performance log file")
        except Exception as e:
            logger.error(f"Error creating performance log: {str(e)}")
    
    # Guardar resultados
    context['task_instance'].xcom_push(key='performance_monitoring', value={
        'degradation_detected': performance_degradation,
        'severity': degradation_severity,
        'metrics': performance_metrics,
        'timestamp': datetime.now().isoformat()
    })
    
    logger.info(f"Model performance monitoring completed. Degradation: {performance_degradation}, Severity: {degradation_severity}")
    return f"Performance monitoring completed - Severity: {degradation_severity}"


def send_advanced_alerts(**context):
    """
    Envía alertas avanzadas basadas en métricas de negocio y condiciones especiales.
    """
    logger.info("Sending advanced alerts...")
    
    # Recopilar información de todas las tareas
    drift_detection = context['task_instance'].xcom_pull(key='drift_detection')
    performance_monitoring = context['task_instance'].xcom_pull(key='performance_monitoring')
    training_results = context['task_instance'].xcom_pull(key='training_results')
    validation_results = context['task_instance'].xcom_pull(key='validation_results')
    
    alerts = []
    
    # Alertas de deriva de datos
    if drift_detection and drift_detection.get('drift_detected', False):
        drift_severity = drift_detection.get('severity', 'MEDIUM')
        alert_severity = 'CRITICAL' if drift_severity == 'CRITICAL' else 'HIGH' if drift_severity == 'HIGH' else 'WARNING'
        alerts.append({
            'type': 'DATA_DRIFT',
            'severity': alert_severity,
            'message': f'Data drift detected in training data (Severity: {drift_severity})',
            'details': {
                'drift_severity': drift_severity,
                'metrics': drift_detection.get('metrics', {}),
                'timestamp': drift_detection.get('timestamp')
            }
        })
    
    # Alertas de degradación de modelo
    if performance_monitoring and performance_monitoring.get('degradation_detected', False):
        perf_severity = performance_monitoring.get('severity', 'MEDIUM')
        alert_severity = 'CRITICAL' if perf_severity == 'CRITICAL' else 'HIGH' if perf_severity == 'HIGH' else 'WARNING'
        metrics = performance_monitoring.get('metrics', {})
        alerts.append({
            'type': 'MODEL_DEGRADATION',
            'severity': alert_severity,
            'message': f'Model performance degradation detected (Severity: {perf_severity}, Max degradation: {metrics.get("max_degradation", 0):.1%})',
            'details': {
                'degradation_severity': perf_severity,
                'metrics': metrics,
                'timestamp': performance_monitoring.get('timestamp')
            }
        })
    
    # Alertas de bajo rendimiento de entrenamiento
    if training_results and validation_results:
        accuracy = training_results.get('accuracy', 0)
        precision = training_results.get('precision', 0)
        recall = training_results.get('recall', 0)
        f1_score = training_results.get('f1_score', 0)
        
        # Umbrales configurables
        alert_accuracy_threshold = float(Variable.get('alert_accuracy_threshold', default_var='0.80'))
        alert_precision_threshold = float(Variable.get('alert_precision_threshold', default_var='0.75'))
        alert_recall_threshold = float(Variable.get('alert_recall_threshold', default_var='0.70'))
        alert_f1_threshold = float(Variable.get('alert_f1_threshold', default_var='0.75'))
        
        if accuracy < alert_accuracy_threshold:
            alerts.append({
                'type': 'LOW_ACCURACY',
                'severity': 'WARNING',
                'message': f'Model accuracy ({accuracy:.3f}) below alert threshold ({alert_accuracy_threshold})',
                'details': {'accuracy': accuracy, 'threshold': alert_accuracy_threshold}
            })
        
        if precision < alert_precision_threshold:
            alerts.append({
                'type': 'LOW_PRECISION',
                'severity': 'WARNING',
                'message': f'Model precision ({precision:.3f}) below alert threshold ({alert_precision_threshold})',
                'details': {'precision': precision, 'threshold': alert_precision_threshold}
            })
        
        if recall < alert_recall_threshold:
            alerts.append({
                'type': 'LOW_RECALL',
                'severity': 'WARNING',
                'message': f'Model recall ({recall:.3f}) below alert threshold ({alert_recall_threshold})',
                'details': {'recall': recall, 'threshold': alert_recall_threshold}
            })
        
        if f1_score < alert_f1_threshold:
            alerts.append({
                'type': 'LOW_F1_SCORE',
                'severity': 'WARNING',
                'message': f'Model F1-score ({f1_score:.3f}) below alert threshold ({alert_f1_threshold})',
                'details': {'f1_score': f1_score, 'threshold': alert_f1_threshold}
            })
    
    # Alertas de validación fallida
    if validation_results and not validation_results.get('passed', False):
        alerts.append({
            'type': 'VALIDATION_FAILED',
            'severity': 'CRITICAL',
            'message': 'Model validation failed - metrics below minimum thresholds',
            'details': validation_results.get('metrics', {})
        })
    
    # Enviar alertas si hay alguna
    if alerts:
        alert_message = "\n\n".join([
            f"ALERT: {alert['type']} - {alert['severity']}\n{alert['message']}\nDetails: {json.dumps(alert['details'], indent=2)}"
            for alert in alerts
        ])
        
        # Enviar email de alerta adicional
        try:
            from airflow.operators.email import send_email
            
            subject = f"🚨 Churn Model Pipeline Alerts - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            body = f"""
The following alerts were generated during the churn model training pipeline:

{alert_message}

Please review the pipeline execution and take appropriate action.

Pipeline Execution Date: {context['ds']}
DAG Run ID: {context['run_id']}
            """
            
            send_email(
                to=get_config()['email_to'],
                subject=subject,
                html_content=body.replace('\n', '<br>')
            )
            
            logger.info(f"Sent {len(alerts)} advanced alerts")
            
        except Exception as e:
            logger.error(f"Error sending advanced alerts: {str(e)}")
    
    # Guardar historial de alertas
    alerts_history_path = f"/tmp/alerts_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(alerts_history_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'alerts': alerts,
            'dag_run_id': context['run_id']
        }, f, indent=2)
    
    context['task_instance'].xcom_push(key='alerts_sent', value=len(alerts))
    
    logger.info("Advanced alerts processing completed")
    return f"Sent {len(alerts)} alerts"


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

# Definir el DAG con configuración mejorada
dag = DAG(
    'churn_model_training_pipeline',
    default_args=default_args,
    description='Automated ML pipeline for customer churn prediction model training with advanced monitoring',
    schedule_interval='@weekly',  # Ejecutar semanalmente
    catchup=False,
    tags=['ml', 'churn', 'training', 'monitoring', 'production'],
    max_active_runs=1,  # Solo una ejecución a la vez
    concurrency=2,  # Máximo 2 tareas en paralelo
)

# Definir tareas
with dag:
    # Sensor para esperar disponibilidad de datos
    wait_for_data = FileSensor(
        task_id='wait_for_data',
        filepath=get_config()['data_path'],
        fs_conn_id='fs_default',
        poke_interval=60,  # Revisar cada minuto
        timeout=60*60*24,  # Timeout después de 24 horas
        mode='poke',
    )
    
    # Tarea de inicio
    start_pipeline = DummyOperator(
        task_id='start_pipeline',
        dag=dag,
    )
    
    # Grupo de tareas de monitoreo
    with TaskGroup("monitoring_and_preparation", dag=dag) as monitoring_group:
        # Detección de deriva de datos
        detect_drift_task = PythonOperator(
            task_id='detect_data_drift',
            python_callable=detect_data_drift,
            dag=dag,
        )
        
        # Monitoreo de rendimiento del modelo
        monitor_performance_task = PythonOperator(
            task_id='monitor_model_performance',
            python_callable=monitor_model_performance,
            dag=dag,
        )
        
        # Verificación de calidad de datos (original)
        check_data_task = PythonOperator(
            task_id='check_data_quality',
            python_callable=check_data_quality,
            dag=dag,
        )
        
        # Dependencias dentro del grupo
        [detect_drift_task, monitor_performance_task] >> check_data_task
    
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
        trigger_rule=TriggerRule.ALL_SUCCESS,  # Solo ejecutar si todas las tareas anteriores fueron exitosas
        dag=dag,
    )
    
    # Tarea de generación de reporte
    report_task = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
        dag=dag,
    )
    
    # Tarea de alertas avanzadas
    advanced_alerts_task = PythonOperator(
        task_id='send_advanced_alerts',
        python_callable=send_advanced_alerts,
        trigger_rule=TriggerRule.ALL_DONE,  # Ejecutar siempre, independientemente del resultado
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
        <p>Status: {{ 'SUCCESS' if task_instance.xcom_pull(key='validation_results', task_ids='validate_model')['passed'] else 'FAILED' }}</p>
        <p>Model Accuracy: {{ task_instance.xcom_pull(key='training_results', task_ids='train_model')['accuracy'] }}</p>
        <p>Data Drift Detected: {{ 'YES' if task_instance.xcom_pull(key='drift_detection', task_ids='detect_data_drift')['drift_detected'] else 'NO' }}</p>
        <p>Performance Degradation: {{ 'YES' if task_instance.xcom_pull(key='performance_monitoring', task_ids='monitor_model_performance')['degradation_detected'] else 'NO' }}</p>
        <p>Report: {{ task_instance.xcom_pull(key='report_path', task_ids='generate_report') }}</p>
        <p>Alerts Sent: {{ task_instance.xcom_pull(key='alerts_sent', task_ids='send_advanced_alerts') or 0 }}</p>
        """,
        dag=dag,
    )
    
    # Tarea de limpieza
    cleanup_task = BashOperator(
        task_id='cleanup_temp_files',
        bash_command='rm -f /tmp/preprocessed_data.csv /tmp/training_report_*.txt /tmp/alerts_history_*.json',
        dag=dag,
    )
    
    # Tarea final
    end_pipeline = DummyOperator(
        task_id='end_pipeline',
        trigger_rule=TriggerRule.ALL_DONE,  # Ejecutar siempre
        dag=dag,
    )
    
    # Definir flujo de dependencias mejorado
    wait_for_data >> start_pipeline >> monitoring_group >> preprocess_task >> train_task >> validate_task
    validate_task >> [deploy_task, report_task] >> advanced_alerts_task >> email_task >> cleanup_task >> end_pipeline