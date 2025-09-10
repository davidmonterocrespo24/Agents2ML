# API Reference

## 🎯 Introduction

The **Multi-Agent AutoML System REST API** provides comprehensive endpoints for integrating automated Machine Learning capabilities into external applications. The API is built with **FastAPI** and offers automatic interactive documentation.

## 🌐 General Information

### **Base URL**
```
http://localhost:8006
```

### **Response Format**
All responses use JSON format with the following standard structure:

```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T10:30:00Z"
}
```

### **HTTP Status Codes**
- `200` - Success
- `201` - Resource created
- `400` - Request error
- `404` - Resource not found
- `500` - Internal server error

### **Authentication**
Currently the system doesn't require authentication. For production environments, it's recommended to implement JWT authentication or API keys.

## 📊 Main Endpoints

### **1. Health Check**

#### `GET /health`
Verifies system status and all agents.

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
Uploads a CSV dataset to the system.

**Request:**
```bash
curl -X POST "http://localhost:8006/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sales.csv"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "file_id": "file_abc123",
    "filename": "sales.csv",
    "size": 1024000,
    "rows_estimated": 10000,
    "upload_path": "uploads/file_abc123_sales.csv"
  },
  "message": "File uploaded successfully",
  "timestamp": "2024-01-01T10:30:00Z"
}
```

### **3. Pipeline Management**

#### `POST /pipeline/start`
Starts a new Machine Learning pipeline.

**Request Body:**
```json
{
  "file_path": "uploads/file_abc123_sales.csv",
  "user_objective": "Predict future sales for the next 30 days",
  "pipeline_name": "sales_prediction_q1_2024"
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
Gets current status of a pipeline.

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
Gets detailed logs of a pipeline.

**Query Parameters:**
- `level` (optional): `info`, `debug`, `warning`, `error`
- `agent` (optional): Filter by specific agent
- `limit` (optional): Maximum number of logs (default: 100)

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
          "target_column": "sales",
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
Cancels a running pipeline.

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
Gets complete results of a finished pipeline.

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
        {"feature": "date", "importance": 0.45},
        {"feature": "month", "importance": 0.25},
        {"feature": "promotion", "importance": 0.20},
        {"feature": "day_of_week", "importance": 0.10}
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
Gets detailed information of a specific model.

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
      "target": "sales"
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
Makes predictions using a trained model.

**Request Body:**
```json
{
  "data": [
    {"month": 1, "day_of_week": 1, "promotion": 0},
    {"month": 1, "day_of_week": 2, "promotion": 1},
    {"month": 1, "day_of_week": 3, "promotion": 0}
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
        "input": {"month": 1, "day_of_week": 1, "promotion": 0},
        "prediction": 1234.56,
        "confidence_interval": {
          "lower": 1189.23,
          "upper": 1279.89,
          "confidence_level": 0.95
        }
      },
      {
        "input": {"month": 1, "day_of_week": 2, "promotion": 1},
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
Makes batch predictions from a CSV file.

**Request:**
```bash
curl -X POST "http://localhost:8006/models/model_ghi789/batch-predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@new_data.csv"
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
Downloads files generated by the system.

**Parameters:**
- `file_type`: `model`, `predictions`, `visualization`, `logs`
- `identifier`: Pipeline, model, or specific file ID

**Examples:**
```bash
# Download trained model
GET /download/model/model_ghi789

# Download predictions
GET /download/predictions/pipeline_def456

# Download visualization
GET /download/visualization/pipeline_def456/trend_chart.png

# Download logs
GET /download/logs/pipeline_def456
```

### **7. System Management**

#### `GET /pipelines`
Lists all pipelines in the system.

**Query Parameters:**
- `status` (optional): `running`, `completed`, `failed`, `cancelled`
- `limit` (optional): Maximum number of pipelines (default: 50)
- `offset` (optional): Offset for pagination (default: 0)

**Response:**
```json
{
  "success": true,
  "data": {
    "pipelines": [
      {
        "pipeline_id": "pipeline_def456",
        "name": "sales_prediction_q1_2024",
        "status": "completed",
        "created_at": "2024-01-01T10:30:00Z",
        "completed_at": "2024-01-01T10:50:00Z",
        "duration_minutes": 20,
        "user_objective": "Predict future sales for the next 30 days"
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
Lists all trained models.

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
Receives real-time updates of pipeline status.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8006/ws/pipeline/pipeline_def456');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log('Pipeline update:', update);
};
```

**Received messages:**
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

## 🔧 Python Client

### **Installation**
```bash
pip install requests websocket-client
```

### **Client Example**
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

# Client usage
client = AutoMLClient()

# 1. Upload file
upload_result = client.upload_file("sales.csv")
file_path = upload_result["data"]["upload_path"]

# 2. Start pipeline
pipeline_result = client.start_pipeline(
    file_path=file_path,
    objective="Predict future sales for 30 days",
    name="sales_prediction_2024"
)
pipeline_id = pipeline_result["data"]["pipeline_id"]

# 3. Monitor progress
import time
while True:
    status = client.get_pipeline_status(pipeline_id)
    if status["data"]["status"] == "completed":
        break
    print(f"Progress: {status['data']['progress_percentage']}%")
    time.sleep(30)

# 4. Get results
results = client.get_results(pipeline_id)
model_id = results["data"]["model_id"]

# 5. Make predictions
predictions = client.predict(model_id, [
    {"month": 1, "day_of_week": 1, "promotion": 0}
])
print(f"Prediction: {predictions['data']['predictions'][0]['prediction']}")
```

## 🔍 Error Codes

### **Client Errors (400-499)**

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

### **Server Errors (500-599)**

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

### **Current Limits**
- **File Upload**: 10 files per minute per IP
- **Pipeline Creation**: 3 pipelines per minute per IP
- **API Calls**: 100 requests per minute per IP
- **WebSocket Connections**: 5 simultaneous connections per IP

### **Rate Limiting Headers**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## 🔒 Security Considerations

### **File Validation**
- Maximum size: 100MB per file
- Allowed types: `.csv`, `.xlsx`, `.json`
- Content validation to prevent injection

### **Sanitization**
- Sanitized filenames
- Input parameter validation
- Special character escaping in logs

### **Resources**
- 30-minute timeout per pipeline
- Memory limits per execution
- Automatic cleanup of temporary files

---

This API provides complete access to all Multi-Agent AutoML System capabilities, enabling seamless integration into existing applications.

**Interactive Documentation**: Available at `http://localhost:8006/docs` when the system is running.