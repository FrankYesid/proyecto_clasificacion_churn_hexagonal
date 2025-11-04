"""
Aplicación FastAPI para el servicio de predicción de churn.
Proporciona endpoints REST para predicciones y monitoreo.
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importar componentes del proyecto
from ...application.dto.customer_dto import CustomerInputDTO, BatchPredictionInputDTO
from ...application.use_cases.predict_customer_churn import PredictCustomerChurnUseCase
from ...application.use_cases.process_customer_batch import ProcessCustomerBatchUseCase
from ...adapters.machine_learning.sklearn_churn_model import SklearnChurnPredictionModel
from ...domain.services.churn_analysis_service import ChurnAnalysisService


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API para predicción de churn de clientes usando machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modelos de respuesta
class HealthResponse(BaseModel):
    """Modelo de respuesta para el endpoint de salud."""
    status: str = Field(..., description="Estado del servicio")
    timestamp: str = Field(..., description="Timestamp de la respuesta")
    model_status: str = Field(..., description="Estado del modelo ML")
    model_version: Optional[str] = Field(None, description="Versión del modelo")
    uptime_seconds: float = Field(..., description="Tiempo de actividad en segundos")


class PredictionResponse(BaseModel):
    """Modelo de respuesta para predicciones individuales."""
    customer_id: str = Field(..., description="ID del cliente")
    prediction: bool = Field(..., description="Predicción de churn (True=Churn, False=No Churn)")
    probability_churn: float = Field(..., description="Probabilidad de churn (0-1)")
    probability_stay: float = Field(..., description="Probabilidad de quedarse (0-1)")
    confidence: float = Field(..., description="Nivel de confianza de la predicción (0-1)")
    risk_level: str = Field(..., description="Nivel de riesgo (Low, Medium, High, Critical)")
    risk_score: float = Field(..., description="Puntuación de riesgo calculada")
    customer_segment: str = Field(..., description="Segmento del cliente")
    lifetime_value: float = Field(..., description="Valor de vida del cliente")
    model_version: str = Field(..., description="Versión del modelo usado")
    prediction_timestamp: str = Field(..., description="Timestamp de la predicción")


class BatchPredictionResponse(BaseModel):
    """Modelo de respuesta para predicciones en lote."""
    batch_id: str = Field(..., description="ID del lote procesado")
    processed_count: int = Field(..., description="Número de clientes procesados")
    error_count: int = Field(..., description="Número de errores")
    results: List[PredictionResponse] = Field(..., description="Lista de predicciones")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Lista de errores si los hay")
    processing_time_seconds: float = Field(..., description="Tiempo de procesamiento en segundos")


class ModelInfoResponse(BaseModel):
    """Modelo de respuesta para información del modelo."""
    model_version: str = Field(..., description="Versión del modelo")
    model_type: str = Field(..., description="Tipo de modelo")
    features: List[str] = Field(..., description="Características usadas por el modelo")
    loaded_at: str = Field(..., description="Timestamp de carga del modelo")
    status: str = Field(..., description="Estado del modelo")


# Variables globales para el estado de la aplicación
app_start_time = datetime.now()
prediction_model = None
churn_analysis_service = None
predict_use_case = None
batch_process_use_case = None


def initialize_services():
    """
    Inicializa los servicios y modelos necesarios.
    """
    global prediction_model, churn_analysis_service, predict_use_case, batch_process_use_case
    
    try:
        # Rutas de archivos (usando variables de entorno con valores por defecto)
        model_path = os.getenv("MODEL_PATH", "models/churn_model.joblib")
        preprocessor_path = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.joblib")
        
        # Verificar que existan los archivos
        if not os.path.exists(model_path):
            logger.warning(f"Modelo no encontrado en: {model_path}")
            logger.warning("El servicio funcionará en modo demo sin predicciones reales")
            return False
        
        # Inicializar modelo de predicción
        prediction_model = SklearnChurnPredictionModel(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            model_version=os.getenv("MODEL_VERSION", "1.0.0")
        )
        
        # Inicializar servicio de análisis
        churn_analysis_service = ChurnAnalysisService()
        
        # Inicializar casos de uso
        predict_use_case = PredictCustomerChurnUseCase(
            prediction_model=prediction_model,
            analysis_service=churn_analysis_service
        )
        
        batch_process_use_case = ProcessCustomerBatchUseCase(
            prediction_model=prediction_model,
            analysis_service=churn_analysis_service
        )
        
        logger.info("Servicios inicializados exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error inicializando servicios: {str(e)}")
        return False


# Inicializar servicios al arrancar la aplicación
@app.on_event("startup")
async def startup_event():
    """
    Evento de inicio de la aplicación.
    """
    logger.info("Iniciando Customer Churn Prediction API")
    success = initialize_services()
    
    if success:
        logger.info("API iniciada exitosamente")
    else:
        logger.warning("API iniciada en modo demo (sin modelo cargado)")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Endpoint de salud para monitoreo.
    
    Returns:
        HealthResponse: Estado del servicio
    """
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    # Determinar estado del modelo
    if prediction_model is None:
        model_status = "not_loaded"
        model_version = None
    else:
        model_status = "loaded"
        model_version = prediction_model.get_model_version()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_status=model_status,
        model_version=model_version,
        uptime_seconds=uptime
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Obtiene información sobre el modelo cargado.
    
    Returns:
        ModelInfoResponse: Información del modelo
    """
    if prediction_model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. El servicio está en modo demo."
        )
    
    model_info = prediction_model.get_model_info()
    
    return ModelInfoResponse(
        model_version=model_info['model_version'],
        model_type=model_info['model_type'],
        features=model_info['features'],
        loaded_at=model_info['loaded_at'],
        status=model_info['status']
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single_customer(customer_data: CustomerInputDTO):
    """
    Realiza predicción de churn para un cliente individual.
    
    Args:
        customer_data (CustomerInputDTO): Datos del cliente
        
    Returns:
        PredictionResponse: Predicción y análisis del cliente
    """
    if predict_use_case is None:
        raise HTTPException(
            status_code=503,
            detail="Servicio de predicción no disponible. El modelo no está cargado."
        )
    
    try:
        # Realizar predicción
        result = predict_use_case.execute(customer_data)
        
        # Convertir a respuesta de API
        return PredictionResponse(
            customer_id=result.customer_id,
            prediction=result.prediction,
            probability_churn=result.probability_churn,
            probability_stay=result.probability_stay,
            confidence=result.confidence,
            risk_level=result.risk_level,
            risk_score=result.risk_score,
            customer_segment=result.customer_segment,
            lifetime_value=result.lifetime_value,
            model_version=result.model_version,
            prediction_timestamp=result.prediction_timestamp.isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Datos inválidos: {str(e)}")
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_customers(batch_data: BatchPredictionInputDTO, background_tasks: BackgroundTasks):
    """
    Realiza predicciones de churn para múltiples clientes.
    
    Args:
        batch_data (BatchPredictionInputDTO): Datos del lote de clientes
        background_tasks (BackgroundTasks): Tareas en segundo plano
        
    Returns:
        BatchPredictionResponse: Predicciones y análisis del lote
    """
    if batch_process_use_case is None:
        raise HTTPException(
            status_code=503,
            detail="Servicio de predicción no disponible. El modelo no está cargado."
        )
    
    start_time = datetime.now()
    
    try:
        # Realizar predicción en lote
        batch_result = batch_process_use_case.execute(batch_data)
        
        # Convertir resultados individuales
        api_results = []
        for result in batch_result.results:
            api_result = PredictionResponse(
                customer_id=result.customer_id,
                prediction=result.prediction,
                probability_churn=result.probability_churn,
                probability_stay=result.probability_stay,
                confidence=result.confidence,
                risk_level=result.risk_level,
                risk_score=result.risk_score,
                customer_segment=result.customer_segment,
                lifetime_value=result.lifetime_value,
                model_version=result.model_version,
                prediction_timestamp=result.prediction_timestamp.isoformat()
            )
            api_results.append(api_result)
        
        # Calcular tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchPredictionResponse(
            batch_id=batch_result.batch_id,
            processed_count=batch_result.processed_count,
            error_count=batch_result.error_count,
            results=api_results,
            errors=batch_result.errors,
            processing_time_seconds=processing_time
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Datos inválidos: {str(e)}")
    except Exception as e:
        logger.error(f"Error en predicción en lote: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post("/predict/upload")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Realiza predicciones desde un archivo CSV subido.
    
    Args:
        file (UploadFile): Archivo CSV con datos de clientes
        
    Returns:
        BatchPredictionResponse: Predicciones para los clientes del archivo
    """
    if batch_process_use_case is None:
        raise HTTPException(
            status_code=503,
            detail="Servicio de predicción no disponible. El modelo no está cargado."
        )
    
    try:
        # Validar tipo de archivo
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="El archivo debe ser CSV")
        
        # Leer archivo CSV
        import pandas as pd
        from io import StringIO
        
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Convertir DataFrame a lista de CustomerInputDTO
        customers = []
        for _, row in df.iterrows():
            customer_data = CustomerInputDTO(
                customer_id=row.get('customer_id', f"AUTO_{_}"),
                gender=row.get('gender', 'Female'),
                senior_citizen=bool(row.get('senior_citizen', False)),
                partner=bool(row.get('partner', False)),
                dependents=bool(row.get('dependents', False)),
                phone_service=bool(row.get('phone_service', False)),
                multiple_lines=row.get('multiple_lines', 'No'),
                internet_service=row.get('internet_service', 'No'),
                online_security=row.get('online_security', 'No'),
                online_backup=row.get('online_backup', 'No'),
                device_protection=row.get('device_protection', 'No'),
                tech_support=row.get('tech_support', 'No'),
                streaming_tv=row.get('streaming_tv', 'No'),
                streaming_movies=row.get('streaming_movies', 'No'),
                contract_type=row.get('contract_type', 'Month-to-month'),
                paperless_billing=bool(row.get('paperless_billing', False)),
                payment_method=row.get('payment_method', 'Electronic check'),
                monthly_charges=float(row.get('monthly_charges', 0.0)),
                total_charges=float(row.get('total_charges', 0.0)),
                tenure_months=int(row.get('tenure_months', 0))
            )
            customers.append(customer_data)
        
        # Crear lote de predicción
        batch_input = BatchPredictionInputDTO(
            batch_id=f"UPLOAD_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            customers=customers
        )
        
        # Realizar predicción en lote
        return await predict_batch_customers(batch_input, BackgroundTasks())
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")
    except Exception as e:
        logger.error(f"Error procesando archivo CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")


@app.get("/")
async def root():
    """
    Endpoint raíz con información básica de la API.
    
    Returns:
        Dict: Información de la API
    """
    return {
        "name": "Customer Churn Prediction API",
        "version": "1.0.0",
        "description": "API para predicción de churn de clientes usando machine learning",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "predict_upload": "/predict/upload",
            "model_info": "/model/info"
        },
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


# Manejo de errores
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """
    Manejador de errores 404.
    """
    return {"error": "Endpoint no encontrado", "status": 404}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """
    Manejador de errores 500.
    """
    logger.error(f"Error interno: {str(exc)}")
    return {"error": "Error interno del servidor", "status": 500}


if __name__ == "__main__":
    import uvicorn
    
    # Configuración para desarrollo
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )