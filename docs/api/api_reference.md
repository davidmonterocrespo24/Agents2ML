# API Reference

## 🎯 Introducción

La **API REST del Sistema Multi-Agent AutoML** proporciona endpoints completos para integrar las capacidades de Machine Learning automatizado en aplicaciones externas. La API está construida con **FastAPI** y ofrece documentación interactiva automática.

## 🌐 Información General

### **Base URL**
```
http://localhost:8006
```

### **Formato de Respuesta**
Todas las respuestas utilizan formato JSON con la siguiente estructura estándar:

```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T10:30:00Z"
}
```

### **Códigos de Estado HTTP**
- `200` - Éxito
- `201` - Recurso creado
- `400` - Error en la petición
- `404` - Recurso no encontrado
- `500` - Error interno del servidor

### **Autenticación**
Actualmente el sistema no requiere autenticación. Para entornos de producción, se recomienda implementar autenticación JWT o API keys.

## 📊 Endpoints Principales

### **1. Health Check**

#### `GET /health`
Verifica el estado del sistema y todos los agentes.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "agents": {
      "DataProcessorAgent": "ready",
      "ModelBuilderAgent": "ready",
      "CodeExecutorAgent": "ready",
      "AnalystAgent": "ready",
      "PredictionAgent": "ready",
      "VisualizationAgent": "ready",
      "UserProxyAgent": "ready"
    },
    "llm_connection": "connected",
    "h2o_status": "available",
    "docker_status": "running"
  },
  "message": "System is healthy",
  "timestamp": "2024-01-01T10:30:00Z"
}
```

### **2. File Upload**

#### `POST /upload`
Sube un dataset CSV al sistema.

**Request:**
```bash
curl -X POST "http://localhost:8006/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@ventas.csv"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "file_id": "file_abc123",
    "filename": "ventas.csv",
    "size": 1024000,
    "rows_estimated": 10000,
    "upload_path": "uploads/file_abc123_ventas.csv"
  },
  "message": "File uploaded successfully",
  "timestamp": "2024-01-01T10:30:00Z"
}
```

### **3. Pipeline Management**

#### `POST /pipeline/start`
Inicia un nuevo pipeline de Machine Learning.

**Request Body:**
```json
{
  "file_path": "uploads/file_abc123_ventas.csv",
  "user_objective": "Predice las ventas futuras para los próximos 30 días",
  "pipeline_name": "prediccion_ventas_q1_2024"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "pipeline_id": "pipeline_def456",
    "status": "started",
    "estimated_duration": "15-20 minutes",
    "stages": [
      "data_analysis",
      "model_training", 
      "prediction_generation",
      "visualization_creation"
    ]
  },
  "message": "ML Pipeline started successfully",
  "timestamp": "2024-01-01T10:30:00Z"
}
```

#### `GET /pipeline/status/{pipeline_id}`
Obtiene el estado actual de un pipeline.

**Response:**
```json
{
  "success": true,
  "data": {
    "pipeline_id": "pipeline_def456",
    "status": "running",
    "current_stage": "model_training",
    "progress_percentage": 65,
    "stages": {
      "data_analysis": {
        "status": "completed",
        "duration": 45,
        "agent": "DataProcessorAgent"
      },
      "model_training": {
        "status": "in_progress", 
        "progress": 65,
        "agent": "ModelBuilderAgent",
        "current_step": "H2O AutoML training"
      },
      "prediction_generation": {
        "status": "pending",
        "agent": "PredictionAgent"
      },
      "visualization_creation": {
        "status": "pending",
        "agent": "VisualizationAgent"
      }
    },
    "start_time": "2024-01-01T10:30:00Z",
    "estimated_completion": "2024-01-01T10:50:00Z"
  },
  "message": "Pipeline status retrieved",
  "timestamp": "2024-01-01T10:42:00Z"
}
```

#### `GET /pipeline/logs/{pipeline_id}`
Obtiene los logs detallados de un pipeline.

**Query Parameters:**
- `level` (optional): `info`, `debug`, `warning`, `error`
- `agent` (optional): Filtrar por agente específico
- `limit` (optional): Número máximo de logs (default: 100)

**Response:**
```json
{
  "success": true,
  "data": {
    "pipeline_id": "pipeline_def456",
    "logs": [
      {
        "timestamp": "2024-01-01T10:30:15Z",
        "level": "info",
        "agent": "DataProcessorAgent",
        "message": "Starting dataset analysis",
        "details": {
          "file_size": "1.2MB",
          "estimated_rows": 10000
        }
      },
      {
        "timestamp": "2024-01-01T10:31:00Z",
        "level": "info",
        "agent": "DataProcessorAgent", 
        "message": "Dataset analysis completed",
        "details": {
          "columns": 5,
          "target_column": "ventas",
          "problem_type": "time_series_regression"
        }
      }
    ],
    "total_logs": 247
  },
  "message": "Pipeline logs retrieved",
  "timestamp": "2024-01-01T10:42:00Z"
}
```

#### `DELETE /pipeline/{pipeline_id}`
Cancela un pipeline en ejecución.

**Response:**
```json
{
  "success": true,
  "data": {
    "pipeline_id": "pipeline_def456",
    "status": "cancelled",
    "cleanup_performed": true
  },
  "message": "Pipeline cancelled successfully",
  "timestamp": "2024-01-01T10:42:00Z"
}
```

### **4. Results and Models**

#### `GET /pipeline/{pipeline_id}/results`
Obtiene los resultados completos de un pipeline terminado.

**Response:**
```json
{
  "success": true,
  "data": {
    "pipeline_id": "pipeline_def456",
    "status": "completed",
    "results": {
      "model_performance": {
        "algorithm": "GBM",
        "rmse": 45.23,
        "mae": 35.87,
        "r2": 0.89,
        "training_time": "8.5 minutes"
      },
      "feature_importance": [
        {"feature": "fecha", "importance": 0.45},
        {"feature": "mes", "importance": 0.25},
        {"feature": "promocion", "importance": 0.20},
        {"feature": "dia_semana", "importance": 0.10}
      ],
      "predictions": {
        "file_path": "results/pipeline_def456/predictions.csv",
        "future_periods": 30,
        "confidence_intervals": true
      },
      "visualizations": {
        "trend_chart": "results/pipeline_def456/trend_chart.png",
        "predictions_chart": "results/pipeline_def456/predictions.png",
        "feature_importance_chart": "results/pipeline_def456/feature_importance.png"
      }
    }
  },
  "message": "Results retrieved successfully",
  "timestamp": "2024-01-01T10:50:00Z"
}
```

#### `GET /models/{model_id}`
Obtiene información detallada de un modelo específico.

**Response:**
```json
{
  "success": true,
  "data": {
    "model_id": "model_ghi789",
    "pipeline_id": "pipeline_def456",
    "algorithm": "GBM",
    "version": "1.0",
    "created_at": "2024-01-01T10:45:00Z",
    "performance_metrics": {
      "rmse": 45.23,
      "mae": 35.87,
      "r2": 0.89,
      "validation_score": 0.87
    },
    "hyperparameters": {
      "ntrees": 100,
      "max_depth": 8,
      "learn_rate": 0.1,
      "sample_rate": 0.8
    },
    "training_data": {
      "rows": 8500,
      "features": 4,
      "target": "ventas"
    },
    "file_path": "models/model_ghi789.zip",
    "size_mb": 15.7
  },
  "message": "Model information retrieved",
  "timestamp": "2024-01-01T10:52:00Z"
}
```

### **5. Predictions**

#### `POST /models/{model_id}/predict`
Realiza predicciones usando un modelo entrenado.

**Request Body:**
```json
{
  "data": [
    {"mes": 1, "dia_semana": 1, "promocion": 0},
    {"mes": 1, "dia_semana": 2, "promocion": 1},
    {"mes": 1, "dia_semana": 3, "promocion": 0}
  ],
  "include_confidence": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "model_id": "model_ghi789",
    "predictions": [
      {
        "input": {"mes": 1, "dia_semana": 1, "promocion": 0},
        "prediction": 1234.56,
        "confidence_interval": {
          "lower": 1189.23,
          "upper": 1279.89,
          "confidence_level": 0.95
        }
      },
      {
        "input": {"mes": 1, "dia_semana": 2, "promocion": 1},
        "prediction": 1345.78,
        "confidence_interval": {
          "lower": 1300.45,
          "upper": 1391.11,
          "confidence_level": 0.95
        }
      }
    ],
    "prediction_time": 0.023
  },
  "message": "Predictions generated successfully",
  "timestamp": "2024-01-01T11:00:00Z"
}
```

#### `POST /models/{model_id}/batch-predict`
Realiza predicciones en lote desde un archivo CSV.

**Request:**
```bash
curl -X POST "http://localhost:8006/models/model_ghi789/batch-predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@nuevos_datos.csv"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "model_id": "model_ghi789",
    "input_rows": 1000,
    "predictions_file": "results/batch_predictions_jkl012.csv",
    "processing_time": 2.34,
    "summary": {
      "min_prediction": 856.23,
      "max_prediction": 1567.89,
      "mean_prediction": 1234.45,
      "std_prediction": 123.67
    }
  },
  "message": "Batch predictions completed",
  "timestamp": "2024-01-01T11:05:00Z"
}
```

### **6. File Downloads**

#### `GET /download/{file_type}/{identifier}`
Descarga archivos generados por el sistema.

**Parámetros:**
- `file_type`: `model`, `predictions`, `visualization`, `logs`
- `identifier`: ID del pipeline, modelo o archivo específico

**Ejemplos:**
```bash
# Descargar modelo entrenado
GET /download/model/model_ghi789

# Descargar predicciones
GET /download/predictions/pipeline_def456

# Descargar visualización
GET /download/visualization/pipeline_def456/trend_chart.png

# Descargar logs
GET /download/logs/pipeline_def456
```

### **7. System Management**

#### `GET /pipelines`
Lista todos los pipelines del sistema.

**Query Parameters:**
- `status` (optional): `running`, `completed`, `failed`, `cancelled`
- `limit` (optional): Número máximo de pipelines (default: 50)
- `offset` (optional): Offset para paginación (default: 0)

**Response:**
```json
{
  "success": true,
  "data": {
    "pipelines": [
      {
        "pipeline_id": "pipeline_def456",
        "name": "prediccion_ventas_q1_2024",
        "status": "completed",
        "created_at": "2024-01-01T10:30:00Z",
        "completed_at": "2024-01-01T10:50:00Z",
        "duration_minutes": 20,
        "user_objective": "Predice las ventas futuras para los próximos 30 días"
      }
    ],
    "total_count": 15,
    "filtered_count": 8
  },
  "message": "Pipelines retrieved successfully",
  "timestamp": "2024-01-01T11:10:00Z"
}
```

#### `GET /models`
Lista todos los modelos entrenados.

**Response:**
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "model_id": "model_ghi789",
        "pipeline_id": "pipeline_def456",
        "algorithm": "GBM",
        "performance_score": 0.89,
        "created_at": "2024-01-01T10:45:00Z",
        "size_mb": 15.7
      }
    ],
    "total_count": 23
  },
  "message": "Models retrieved successfully",
  "timestamp": "2024-01-01T11:15:00Z"
}
```

## 📡 WebSocket Endpoints

### **Real-time Pipeline Updates**

#### `WS /ws/pipeline/{pipeline_id}`
Recibe actualizaciones en tiempo real del estado del pipeline.

**Conexión:**
```javascript
const ws = new WebSocket('ws://localhost:8006/ws/pipeline/pipeline_def456');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log('Pipeline update:', update);
};
```

**Mensajes recibidos:**
```json
{
  "type": "progress_update",
  "pipeline_id": "pipeline_def456",
  "stage": "model_training",
  "progress": 75,
  "message": "Training model 15 of 20",
  "timestamp": "2024-01-01T10:42:00Z"
}

{
  "type": "stage_completed",
  "pipeline_id": "pipeline_def456", 
  "stage": "model_training",
  "duration": 540,
  "next_stage": "prediction_generation",
  "timestamp": "2024-01-01T10:45:00Z"
}

{
  "type": "pipeline_completed",
  "pipeline_id": "pipeline_def456",
  "total_duration": 1200,
  "results_available": true,
  "timestamp": "2024-01-01T10:50:00Z"
}
```

## 🔧 Cliente Python

### **Instalación**
```bash
pip install requests websocket-client
```

### **Ejemplo de Cliente**
```python
import requests
import json
from datetime import datetime

class AutoMLClient:
    def __init__(self, base_url="http://localhost:8006"):
        self.base_url = base_url
        
    def upload_file(self, file_path):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.base_url}/upload", files=files)
        return response.json()
    
    def start_pipeline(self, file_path, objective, name):
        data = {
            "file_path": file_path,
            "user_objective": objective,
            "pipeline_name": name
        }
        response = requests.post(f"{self.base_url}/pipeline/start", json=data)
        return response.json()
    
    def get_pipeline_status(self, pipeline_id):
        response = requests.get(f"{self.base_url}/pipeline/status/{pipeline_id}")
        return response.json()
    
    def get_results(self, pipeline_id):
        response = requests.get(f"{self.base_url}/pipeline/{pipeline_id}/results")
        return response.json()
    
    def predict(self, model_id, data):
        payload = {"data": data, "include_confidence": True}
        response = requests.post(f"{self.base_url}/models/{model_id}/predict", json=payload)
        return response.json()

# Uso del cliente
client = AutoMLClient()

# 1. Subir archivo
upload_result = client.upload_file("ventas.csv")
file_path = upload_result["data"]["upload_path"]

# 2. Iniciar pipeline
pipeline_result = client.start_pipeline(
    file_path=file_path,
    objective="Predice ventas futuras para 30 días",
    name="prediccion_ventas_2024"
)
pipeline_id = pipeline_result["data"]["pipeline_id"]

# 3. Monitorear progreso
import time
while True:
    status = client.get_pipeline_status(pipeline_id)
    if status["data"]["status"] == "completed":
        break
    print(f"Progreso: {status['data']['progress_percentage']}%")
    time.sleep(30)

# 4. Obtener resultados
results = client.get_results(pipeline_id)
model_id = results["data"]["model_id"]

# 5. Hacer predicciones
predictions = client.predict(model_id, [
    {"mes": 1, "dia_semana": 1, "promocion": 0}
])
print(f"Predicción: {predictions['data']['predictions'][0]['prediction']}")
```

## 🔍 Códigos de Error

### **Errores de Cliente (400-499)**

#### `400 - Bad Request`
```json
{
  "success": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required field 'user_objective'",
    "details": {
      "missing_fields": ["user_objective"],
      "provided_fields": ["file_path", "pipeline_name"]
    }
  },
  "timestamp": "2024-01-01T11:20:00Z"
}
```

#### `404 - Not Found`
```json
{
  "success": false,
  "error": {
    "code": "PIPELINE_NOT_FOUND",
    "message": "Pipeline with ID 'pipeline_xyz' not found",
    "details": {
      "pipeline_id": "pipeline_xyz",
      "suggestion": "Check pipeline ID or use GET /pipelines to list available pipelines"
    }
  },
  "timestamp": "2024-01-01T11:25:00Z"
}
```

### **Errores de Servidor (500-599)**

#### `500 - Internal Server Error`
```json
{
  "success": false,
  "error": {
    "code": "AGENT_EXECUTION_FAILED",
    "message": "ModelBuilderAgent failed to generate code",
    "details": {
      "agent": "ModelBuilderAgent",
      "stage": "code_generation",
      "error_type": "LLM_CONNECTION_ERROR",
      "retry_count": 3
    }
  },
  "timestamp": "2024-01-01T11:30:00Z"
}
```

## 📊 Rate Limiting

### **Límites Actuales**
- **File Upload**: 10 archivos por minuto por IP
- **Pipeline Creation**: 3 pipelines por minuto por IP
- **API Calls**: 100 requests por minuto por IP
- **WebSocket Connections**: 5 conexiones simultáneas por IP

### **Headers de Rate Limiting**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## 🔒 Consideraciones de Seguridad

### **Validación de Archivos**
- Máximo tamaño: 100MB por archivo
- Tipos permitidos: `.csv`, `.xlsx`, `.json`
- Validación de contenido para prevenir inyección

### **Sanitización**
- Nombres de archivos sanitizados
- Validación de parámetros de entrada
- Escape de caracteres especiales en logs

### **Recursos**
- Timeout de 30 minutos por pipeline
- Límites de memoria por ejecución
- Cleanup automático de archivos temporales

---

Esta API proporciona acceso completo a todas las capacidades del Sistema Multi-Agent AutoML, permitiendo integración seamless en aplicaciones existentes.

**Documentación Interactiva**: Disponible en `http://localhost:8006/docs` cuando el sistema está en ejecución.