# Guía de Configuración de Variables de Airflow para Pipeline de Entrenamiento

## 📋 Introducción

Este documento describe todas las variables de configuración disponibles para el DAG de Airflow `churn_model_training_pipeline`. Estas variables permiten personalizar el comportamiento del pipeline sin modificar el código.

## 🔧 Variables de Configuración

### Variables Requeridas

#### 1. Ruta de Datos
```bash
# Ruta al archivo de datos de entrenamiento
airflow variables set churn_data_path "/opt/airflow/data/raw/customer_data.csv"
```

#### 2. Rutas de Salida
```bash
# Directorio para modelos entrenados
airflow variables set model_output_path "/opt/airflow/models"

# Directorio para preprocesadores
airflow variables set preprocessor_output_path "/opt/airflow/models"
```

#### 3. Configuración de Email
```bash
# Email del equipo ML para notificaciones
airflow variables set ml_team_email "ml-team@company.com"
```

### Variables Opcionales

#### 4. Configuración de Entrenamiento
```bash
# Proporción de datos para prueba (default: 0.2)
airflow variables set test_size "0.2"

# Semilla aleatoria para reproducibilidad (default: 42)
airflow variables set random_state "42"

# Versión del modelo (default: 1.0.0)
airflow variables set model_version "1.0.0"
```

#### 5. Umbrales de Validación
```bash
# Precisión mínima requerida (default: 0.75)
airflow variables set min_model_accuracy "0.75"

# Precisión mínima requerida (default: 0.70)
airflow variables set min_model_precision "0.70"

# Recall mínimo requerido (default: 0.65)
airflow variables set min_model_recall "0.65"
```

#### 6. Configuración de Despliegue
```bash
# Directorio de modelos en producción
airflow variables set production_model_path "/opt/airflow/models/production"
```

#### 7. Configuración de Monitoreo Avanzado
```bash
# Acción ante detección de deriva de datos: warn, stop, ignore (default: warn)
airflow variables set drift_action_on_detection "warn"

# Umbral de alerta para precisión (default: 0.80)
airflow variables set alert_accuracy_threshold "0.80"
```

## 🚀 Comandos de Configuración

### Configuración Completa Inicial

```bash
# Variables requeridas
airflow variables set churn_data_path "/opt/airflow/data/raw/customer_data.csv"
airflow variables set model_output_path "/opt/airflow/models"
airflow variables set preprocessor_output_path "/opt/airflow/models"
airflow variables set ml_team_email "ml-team@company.com"

# Variables opcionales con valores recomendados
airflow variables set test_size "0.2"
airflow variables set random_state "42"
airflow variables set model_version "1.0.0"
airflow variables set min_model_accuracy "0.75"
airflow variables set min_model_precision "0.70"
airflow variables set min_model_recall "0.65"
airflow variables set production_model_path "/opt/airflow/models/production"
airflow variables set drift_action_on_detection "warn"
airflow variables set alert_accuracy_threshold "0.80"
```

### Ver Variables Actuales

```bash
# Listar todas las variables
airflow variables list

# Obtener valor de variable específica
airflow variables get churn_data_path

# Exportar todas las variables
airflow variables export /tmp/airflow_variables.json
```

### Actualizar Variables

```bash
# Actualizar variable individual
airflow variables set test_size "0.25"

# Importar variables desde archivo
airflow variables import /tmp/airflow_variables.json
```

### Eliminar Variables

```bash
# Eliminar variable individual
airflow variables delete test_size

# Eliminar todas las variables (¡CUIDADO!)
airflow variables delete --all
```

## 📊 Configuraciones por Ambiente

### Desarrollo
```bash
airflow variables set churn_data_path "/opt/airflow/data/raw/customer_data_dev.csv"
airflow variables set model_version "1.0.0-dev"
airflow variables set min_model_accuracy "0.70"  # Umbrales más bajos
airflow variables set drift_action_on_detection "warn"
```

### Staging
```bash
airflow variables set churn_data_path "/opt/airflow/data/raw/customer_data_staging.csv"
airflow variables set model_version "1.0.0-rc"
airflow variables set min_model_accuracy "0.75"
airflow variables set drift_action_on_detection "warn"
```

### Producción
```bash
airflow variables set churn_data_path "/opt/airflow/data/raw/customer_data.csv"
airflow variables set model_version "1.0.0"
airflow variables set min_model_accuracy "0.80"  # Umbrales más altos
airflow variables set drift_action_on_detection "stop"  # Detener si hay deriva
airflow variables set alert_accuracy_threshold "0.85"
```

## 🔍 Monitoreo y Troubleshooting

### Verificar Configuración
```bash
# Verificar que todas las variables requeridas estén configuradas
airflow variables get churn_data_path || echo "❌ churn_data_path no configurado"
airflow variables get model_output_path || echo "❌ model_output_path no configurado"
airflow variables get ml_team_email || echo "❌ ml_team_email no configurado"
```

### Logs de Configuración
El DAG registra automáticamente la configuración utilizada en cada ejecución. Revisar los logs de la tarea `start_pipeline` para ver la configuración completa.

### Validación de Variables
El DAG incluye validación automática de:
- Existencia de rutas de archivos
- Rangos válidos para parámetros numéricos
- Formato correcto de emails
- Valores de enumeración (como `drift_action_on_detection`)

## ⚠️ Mejores Prácticas

### 1. Seguridad
- No almacenar contraseñas o información sensible en variables
- Usar conexiones de Airflow para credenciales
- Validar rutas de archivos antes de configurar

### 2. Versionado
- Usar versiones semánticas para `model_version`
- Documentar cambios en variables críticas
- Mantener respaldo de configuraciones

### 3. Testing
- Probar cambios en ambiente de desarrollo primero
- Validar que los nuevos umbrales sean alcanzables
- Monitorear impacto de cambios en métricas

### 4. Documentación
- Documentar razones de cambios en variables
- Mantener historial de valores anteriores
- Comunicar cambios al equipo

## 🔄 Actualización de Variables en Caliente

### Actualización sin Reiniciar
Las variables pueden actualizarse sin reiniciar Airflow:
```bash
airflow variables set min_model_accuracy "0.85"
```
Los cambios se aplicarán en la próxima ejecución del DAG.

### Actualización Masiva
Para actualizar múltiples variables:
```bash
# Crear archivo de configuración
cat > /tmp/new_config.json << EOF
{
  "min_model_accuracy": "0.85",
  "min_model_precision": "0.80",
  "drift_action_on_detection": "stop"
}
EOF

# Importar configuración
for key in $(jq -r 'keys[]' /tmp/new_config.json); do
  value=$(jq -r ".[\"$key\"]" /tmp/new_config.json)
  airflow variables set "$key" "$value"
done
```

## 📈 Métricas de Configuración

### Umbrales Recomendados por Tipo de Modelo

| Tipo de Modelo | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Alta Precisión | 0.85+ | 0.80+ | 0.70+ | 0.75+ |
| Balanceado | 0.80+ | 0.75+ | 0.75+ | 0.75+ |
| Alto Recall | 0.80+ | 0.70+ | 0.80+ | 0.75+ |
| Experimental | 0.70+ | 0.65+ | 0.65+ | 0.65+ |

### Configuración de Deriva de Datos

| Nivel de Alerta | Umbral de Deriva | Acción Recomendada |
|-----------------|------------------|---------------------|
| Bajo | 1-2 σ | Monitorear continuo |
| Medio | 2-3 σ | Alerta y revisión |
| Alto | 3+ σ | Detener pipeline |
| Crítico | 5+ σ | Investigación inmediata |

## 🎯 Conclusión

La configuración adecuada de variables de Airflow es crucial para el funcionamiento óptimo del pipeline de entrenamiento. Esta guía proporciona:

- ✅ Variables requeridas y opcionales
- ✅ Comandos de configuración prácticos
- ✅ Configuraciones por ambiente
- ✅ Mejores prácticas de seguridad
- ✅ Guías de troubleshooting

Para soporte adicional, revisar los logs del DAG o contactar al equipo de ML.