# Multi-Agent AutoML System
<p align="center">
   <img src="img/Screenshot%202025-09-09%20181102.png" alt="Screenshot 1" width="600"/>
   </p>
A complete automated Machine Learning system that uses **7 specialized AI agents** to transform raw data into production-ready ML models. Users can upload datasets, define objectives, and automatically obtain trained models using H2O AutoML. The system uses the open-source model `gpt-oss:120b` via Ollama or Hugging Face API, and does not require the OpenAI API.

## 🎯 What does this system do?

**From data to model in minutes**: Upload a CSV, describe what you want to predict, and get a trained model with predictions and visualizations automatically.

## 🚀 Main Features

🤖 **ML Pipeline with 7 AI Agents**
- **DataProcessorAgent**: Automatic dataset analysis
- **ModelBuilderAgent**: Python code generation + H2O AutoML  
- **CodeExecutorAgent**: Safe execution in Docker
- **AnalystAgent**: Quality control and validation
- **PredictionAgent**: Prediction generation
- **VisualizationAgent**: Chart creation
- **UserProxyAgent**: General coordination

📊 **Complete Web Dashboard**
- Real-time training monitoring
- Interactive ML job management
- Model metrics visualization
- Detailed log viewer

🔄 **Fully Automated Workflow**
- Dataset upload (CSV/Excel)
- Natural language objective definition
- Automatic feature engineering
- Model training and validation
- Prediction generation
- Results visualization

📈 **Advanced Features**
- Job queue management
- Model versioning and storage
- Complete training history
- Model download and deployment

## Quick Start

### Prerequisites

- Python 3.8+
- Docker (for H2O execution environment)
- Ollama (for local LLM inference) **or** Hugging Face API key (for remote inference)

### Installation

1. **Clone/Download the project**
```bash
git clone https://github.com/davidmonterocrespo24/Agents2ML
cd Agent-AutoML

```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
Build the Docker image for the application (from the included `Dockerfile`). In the project root, run:
   ```powershell
      docker build -t agents2ml .
   ```


   If using Hugging Face API, create a `.env` file with:
   ```
   HF_TOKEN=your_huggingface_api_token_here
   ```
   If using Ollama, make sure the Ollama server is running locally and the `gpt-oss:120b` model is available.

4. **Start the application**
```bash
python start.py
```
The system will automatically use Ollama (local) or Hugging Face (cloud) for LLM inference with `gpt-oss:120b`.

5. **Access the web interface**
Open `http://localhost:8006` in your browser

## Architecture

### Multi-Agent System
The system uses 7 specialized agents that work together:

```mermaid
graph TD
    A[User] --> B[Web Interface]
    B --> C[Pipeline Orchestrator]
    C --> D[DataProcessorAgent]
    C --> E[ModelBuilderAgent]
    C --> F[CodeExecutorAgent]
    C --> G[AnalystAgent]
    C --> H[PredictionAgent]
    C --> I[VisualizationAgent]
    C --> J[UserProxyAgent]
    
    D --> K[H2O AutoML]
    E --> K
    F --> K
    K --> L[Trained Models]
```

### Execution Flow

1. **Data Upload & Analysis**
   - User uploads CSV file
   - DataProcessorAgent analyzes dataset structure
   - Automatic data type detection and validation

2. **Model Generation**
   - ModelBuilderAgent generates training code
   - Integration with H2O AutoML for optimal models
   - Automatic hyperparameter optimization

3. **Safe Execution**
   - CodeExecutorAgent runs code in Docker container
   - Real-time log monitoring
   - Error handling and recovery

4. **Quality Control**
   - AnalystAgent validates results
   - Model performance evaluation
   - Recommendation generation

5. **Predictions & Visualization**
   - PredictionAgent generates forecasts
   - VisualizationAgent creates charts
   - Interactive result presentation

## Usage Examples

### Example 1: Sales Prediction
```python
# 1. Upload sales data (sales.csv)
# 2. Set objective: "Predict monthly sales"
# 3. System automatically:
#    - Analyzes seasonal patterns
#    - Trains time series models
#    - Generates future predictions
#    - Creates trend visualizations
```

### Example 2: Customer Classification
```python
# 1. Upload customer data
# 2. Set objective: "Classify customer type"
# 3. System automatically:
#    - Performs feature engineering
#    - Trains classification models
#    - Evaluates accuracy
#    - Provides customer insights
```

## API Reference

### Main Endpoints

#### POST /pipeline/start
Start a new ML pipeline
```json
{
  "file_path": "path/to/dataset.csv",
  "user_objective": "Predict sales for next 3 months",
  "pipeline_name": "sales_forecast_2024"
}
```

#### GET /pipeline/status/{pipeline_id}
Get pipeline execution status
```json
{
  "status": "running",
  "current_stage": "model_training",
  "progress": 65,
  "logs": [...],
  "results": {...}
}
```

#### GET /models/{model_id}/predict
Make predictions with trained model
```json
{
  "predictions": [...],
  "confidence": 0.95,
  "feature_importance": {...}
}
```

## Configuration


### Environment Variables
Set the following variables in a `.env` file in the project root:

- `HF_TOKEN`: Hugging Face API key (required if using Hugging Face for LLM)
- `DATABASE_URL`: Path or URL to the SQLite database file (default: `automl_system.db`)
- `DOCKER_IMAGE`: Name of the Docker image for code execution (default: `my-autogen-h2o:latest`)
- `CODE_EXECUTOR_TIMEOUT`: Timeout (in ms) for code execution in Docker (default: `3000000`)
- `UPLOAD_DIR`: Directory for uploaded datasets (default: `uploads`)
- `MODELS_DIR`: Directory for storing trained models (default: `models`)
- `RESULTS_DIR`: Directory for storing results (default: `results`)
- `CODING_DIR`: Directory for temporary code execution files (default: `coding`)
- `HOST`: Server host address (default: `0.0.0.0`)
- `PORT`: Server port (default: `8006`)
- `OLLAMA_BASE_URL`: Base URL for Ollama server (default: `http://127.0.0.1:11434`)
- `OLLAMA_MODEL`: Name of the Ollama model to use (default: `gpt-oss:20b`)
- `DEBUG`: Enable debug mode (`true` or `false`, default: `false`)

Example `.env`:
```env
HF_TOKEN=your_huggingface_token
DATABASE_URL=automl_system.db
DOCKER_IMAGE=my-autogen-h2o:latest
CODE_EXECUTOR_TIMEOUT=3000000
UPLOAD_DIR=uploads
MODELS_DIR=models
RESULTS_DIR=results
CODING_DIR=coding
HOST=0.0.0.0
PORT=8006
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=gpt-oss:20b
DEBUG=false
```

### Agent Configuration
Each agent can be configured in `config.py`:
```python
AGENT_CONFIG = {
    "data_processor": {
        "max_sample_size": 10000,
        "analysis_depth": "detailed"
    },
    "model_builder": {
        "max_models": 10,
        "training_time": 300
    }
}
```

## Advanced Features

### Custom Models
The system supports custom model integration:
```python
# Register custom model
from agents.model_builder_agent import ModelBuilderAgent

agent = ModelBuilderAgent()
agent.register_custom_model("custom_xgb", custom_xgb_trainer)
```

### Pipeline Extensions
Add custom processing steps:
```python
# Custom preprocessing
def custom_preprocessor(df):
    # Your custom logic here
    return processed_df

pipeline.add_step("custom_preprocessing", custom_preprocessor)
```

## Monitoring & Debugging

### Real-time Logs
Monitor execution through the web interface or API:
```bash
curl http://localhost:8000/pipeline/logs/{pipeline_id}
```

### Performance Metrics
Access detailed performance data:
- Training time per stage
- Memory usage
- Model accuracy metrics
- Prediction confidence intervals

### Error Handling
The system includes comprehensive error handling:
- Automatic retry on transient failures
- Graceful degradation for resource limits
- Detailed error reporting and logging

## Deployment

### Docker Deployment
```dockerfile
# Use provided Dockerfile
docker build -t automl-system .
docker run -p 8000:8000 -e  automl-system
```

### Production Considerations
- Configure reverse proxy (nginx/traefik)
- Set up monitoring (Prometheus/Grafana)
- Implement backup strategies for models
- Scale Docker containers as needed

## Troubleshooting

### Common Issues


1. **Ollama API Errors**
   - Verify API is valid
   - Download the model using ollama run gpt-oss:120b
   - Make sure your hardware is adequate

2. **Docker Execution Issues**
   - Verify Docker is running
   - Check container permissions
   - Monitor disk space

3. **Memory Issues**
   - Reduce sample size for large datasets
   - Limit concurrent pipelines
   - Increase container memory limits

### Debug Mode
Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python start.py
```

## Contributing


### Adding New Agents
1. Create agent class inheriting from `BaseAgent`
2. Implement required methods
3. Add agent to pipeline configuration
4. Write tests for new functionality

## License
OSS/Apache License

## Support

- **Documentation**: Check the `/docs` folder for detailed guides
- **Issues**: Report bugs and feature requests on GitHub


## Roadmap

### Version 2.0 (Planned)
- [ ] Multi-modal data support (images, text)
- [ ] Model interpretability tools
- [ ] Advanced ensemble methods

### Version 2.1 (Future)
- [ ] Distributed training support
- [ ] Cloud deployment templates
- [ ] Integration with MLflow
- [ ] Real-time streaming predictions

---


**Built with ❤️ using AutoGen, H2O AutoML, FastAPI, Docker, and gpt-oss:120b (Ollama/Hugging Face)**

---

## Gallery

Below are screenshots of the system in action:

<p align="center">
   <img src="img/Screenshot%202025-09-09%20181102.png" alt="Screenshot 1" width="600"/>
   <img src="img/Screenshot%202025-09-09%20181117.png" alt="Screenshot 2" width="600"/>
   <img src="img/Screenshot%202025-09-09%20181237.png" alt="Screenshot 3" width="600"/>
   <img src="img/Screenshot%202025-09-09%20181245.png" alt="Screenshot 4" width="600"/>
   <img src="img/Screenshot%202025-09-09%20181257.png" alt="Screenshot 5" width="600"/>
   <img src="img/Screenshot%202025-09-09%20181311.png" alt="Screenshot 6" width="600"/>
   <img src="img/Screenshot%202025-09-09%20181321.png" alt="Screenshot 7" width="600"/>
   <img src="img/Screenshot%202025-09-09%20181355.png" alt="Screenshot 8" width="600"/>
   <img src="img/Screenshot%202025-09-09%20181557.png" alt="Screenshot 9" width="600"/>
   <img src="img/Screenshot%202025-09-09%20181608.png" alt="Screenshot 10" width="600"/>
   <img src="img/Screenshot%202025-09-09%20181628.png" alt="Screenshot 11" width="600"/>
   <img src="img/Screenshot%202025-09-09%20181714.png" alt="Screenshot 12" width="600"/>
</p>
