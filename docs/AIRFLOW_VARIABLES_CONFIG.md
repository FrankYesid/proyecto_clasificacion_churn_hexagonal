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

#### 7. Configuración de Monitoreo Avanzado - Data Drift
```bash
# Acción ante detección de deriva de datos: warn, stop, ignore (default: warn)
airflow variables set drift_action_on_detection "warn"

# Umbrales de detección de drift (en desviaciones estándar)
airflow variables set drift_threshold_medium "2.0"    # Detección media
airflow variables set drift_threshold_high "3.0"      # Detección alta
airflow variables set drift_threshold_critical "5.0"  # Detección crítica

# Umbral para drift en distribución de churn (default: 0.05 = 5%)
airflow variables set churn_dist_drift_threshold "0.05"
```

#### 8. Configuración de Monitoreo de Rendimiento
```bash
# Umbrales de degradación de rendimiento (proporción decimal)
airflow variables set performance_degradation_threshold_low "0.05"     # 5% degradación
airflow variables set performance_degradation_threshold_medium "0.10"  # 10% degradación
airflow variables set performance_degradation_threshold_high "0.20"     # 20% degradación

# Puntos mínimos de datos para análisis de rendimiento (default: 5)
airflow variables set performance_min_data_points "5"
```

#### 9. Configuración de Alertas
```bash
# Umbrales de alerta para métricas del modelo
airflow variables set alert_accuracy_threshold "0.80"   # Umbral de alerta para accuracy
airflow variables set alert_precision_threshold "0.75"   # Umbral de alerta para precision
airflow variables set alert_recall_threshold "0.70"      # Umbral de alerta para recall
airflow variables set alert_f1_threshold "0.75"          # Umbral de alerta para F1-score
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

# Configuración de Data Drift
airflow variables set drift_action_on_detection "warn"
airflow variables set drift_threshold_medium "2.0"
airflow variables set drift_threshold_high "3.0"
airflow variables set drift_threshold_critical "5.0"
airflow variables set churn_dist_drift_threshold "0.05"

# Configuración de Monitoreo de Rendimiento
airflow variables set performance_degradation_threshold_low "0.05"
airflow variables set performance_degradation_threshold_medium "0.10"
airflow variables set performance_degradation_threshold_high "0.20"
airflow variables set performance_min_data_points "5"

# Configuración de Alertas
airflow variables set alert_accuracy_threshold "0.80"
airflow variables set alert_precision_threshold "0.75"
airflow variables set alert_recall_threshold "0.70"
airflow variables set alert_f1_threshold "0.75"

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
airflow variables set drift_threshold_medium "2.0"
airflow variables set drift_threshold_high "3.0"
airflow variables set drift_threshold_critical "5.0"
airflow variables set performance_degradation_threshold_low "0.05"
airflow variables set performance_degradation_threshold_medium "0.10"
airflow variables set performance_degradation_threshold_high "0.20"
airflow variables set alert_accuracy_threshold "0.85"
airflow variables set alert_precision_threshold "0.80"
airflow variables set alert_recall_threshold "0.75"
airflow variables set alert_f1_threshold "0.80"
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
- Umbrales de drift y degradación dentro de rangos razonables

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

### Configuración de Deriva de Datos (Data Drift)

| Variable | Valor Default | Descripción | Rango Recomendado |
|----------|---------------|-------------|-------------------|
| `drift_threshold_medium` | 2.0 | Umbral medio (desviaciones estándar) | 1.5 - 2.5 |
| `drift_threshold_high` | 3.0 | Umbral alto (desviaciones estándar) | 2.5 - 4.0 |
| `drift_threshold_critical` | 5.0 | Umbral crítico (desviaciones estándar) | 4.0 - 6.0 |
| `churn_dist_drift_threshold` | 0.05 | Umbral para drift en distribución de churn | 0.03 - 0.10 |
| `drift_action_on_detection` | warn | Acción: warn, stop, ignore | N/A |

**Niveles de Severidad de Drift**:
- **NONE**: Sin drift detectado
- **LOW**: Drift menor al umbral medio
- **MEDIUM**: 2-3 σ de diferencia
- **HIGH**: 3-5 σ de diferencia
- **CRITICAL**: > 5 σ de diferencia

### Configuración de Monitoreo de Rendimiento

| Variable | Valor Default | Descripción | Rango Recomendado |
|----------|---------------|-------------|-------------------|
| `performance_degradation_threshold_low` | 0.05 | Umbral bajo (5% degradación) | 0.03 - 0.07 |
| `performance_degradation_threshold_medium` | 0.10 | Umbral medio (10% degradación) | 0.07 - 0.15 |
| `performance_degradation_threshold_high` | 0.20 | Umbral alto (20% degradación) | 0.15 - 0.25 |
| `performance_min_data_points` | 5 | Puntos mínimos para análisis | 3 - 10 |

**Niveles de Severidad de Degradación**:
- **NONE**: Sin degradación detectada
- **MEDIUM**: 5-10% de degradación
- **HIGH**: 10-20% de degradación
- **CRITICAL**: > 20% de degradación

### Configuración de Umbrales de Alertas

| Variable | Valor Default | Descripción | Rango Recomendado |
|----------|---------------|-------------|-------------------|
| `alert_accuracy_threshold` | 0.80 | Umbral de alerta para accuracy | 0.75 - 0.90 |
| `alert_precision_threshold` | 0.75 | Umbral de alerta para precision | 0.70 - 0.85 |
| `alert_recall_threshold` | 0.70 | Umbral de alerta para recall | 0.65 - 0.80 |
| `alert_f1_threshold` | 0.75 | Umbral de alerta para F1-score | 0.70 - 0.85 |

### Matriz de Configuración Recomendada por Ambiente

| Ambiente | Drift Action | Drift Thresholds | Performance Thresholds | Alert Thresholds |
|----------|--------------|------------------|------------------------|------------------|
| **Desarrollo** | warn | Relajados (2.5, 4.0, 6.0) | Relajados (0.07, 0.12, 0.22) | Relajados (0.75, 0.70, 0.65, 0.70) |
| **Staging** | warn | Moderados (2.0, 3.0, 5.0) | Moderados (0.05, 0.10, 0.20) | Moderados (0.80, 0.75, 0.70, 0.75) |
| **Producción** | stop | Estrictos (1.5, 2.5, 4.0) | Estrictos (0.03, 0.07, 0.15) | Estrictos (0.85, 0.80, 0.75, 0.80) |

## 🎯 Conclusión

La configuración adecuada de variables de Airflow es crucial para el funcionamiento óptimo del pipeline de entrenamiento. Esta guía proporciona:

- ✅ Variables requeridas y opcionales (más de 20 variables)
- ✅ Comandos de configuración prácticos
- ✅ Configuraciones por ambiente (Desarrollo, Staging, Producción)
- ✅ Mejores prácticas de seguridad
- ✅ Guías de troubleshooting
- ✅ Configuración completa de monitoreo avanzado:
  - Data Drift Detection con umbrales configurables
  - Performance Monitoring con análisis de tendencias
  - Sistema de alertas multi-nivel

## 📋 Resumen de Variables por Categoría

### Variables Requeridas (5)
1. `churn_data_path` - Ruta a datos de entrenamiento
2. `model_output_path` - Directorio de salida para modelos
3. `preprocessor_output_path` - Directorio de salida para preprocesadores
4. `ml_team_email` - Email para notificaciones
5. `production_model_path` - Directorio de modelos en producción

### Variables de Entrenamiento (2)
6. `test_size` - Proporción de datos para prueba
7. `random_state` - Semilla aleatoria

### Variables de Validación (3)
8. `min_model_accuracy` - Umbral mínimo de accuracy
9. `min_model_precision` - Umbral mínimo de precision
10. `min_model_recall` - Umbral mínimo de recall

### Variables de Data Drift (5)
11. `drift_action_on_detection` - Acción ante drift (warn/stop/ignore)
12. `drift_threshold_medium` - Umbral medio de drift
13. `drift_threshold_high` - Umbral alto de drift
14. `drift_threshold_critical` - Umbral crítico de drift
15. `churn_dist_drift_threshold` - Umbral de drift en distribución de churn

### Variables de Monitoreo de Rendimiento (4)
16. `performance_degradation_threshold_low` - Umbral bajo de degradación
17. `performance_degradation_threshold_medium` - Umbral medio de degradación
18. `performance_degradation_threshold_high` - Umbral alto de degradación
19. `performance_min_data_points` - Puntos mínimos para análisis

### Variables de Alertas (4)
20. `alert_accuracy_threshold` - Umbral de alerta para accuracy
21. `alert_precision_threshold` - Umbral de alerta para precision
22. `alert_recall_threshold` - Umbral de alerta para recall
23. `alert_f1_threshold` - Umbral de alerta para F1-score

### Variables Opcionales (2)
24. `model_version` - Versión del modelo
25. `churn_data_path` - Ruta a datos (puede ser diferente en diferentes ambientes)

**Total: 25 variables configurables**

Para soporte adicional, revisar los logs del DAG o contactar al equipo de ML.