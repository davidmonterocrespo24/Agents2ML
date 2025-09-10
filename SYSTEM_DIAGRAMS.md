# Multi-Agent AutoML System Diagrams

## General Architecture Diagram

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Web Dashboard<br/>HTML + JavaScript]
        API[FastAPI Server<br/>app.py]
    end
    
    subgraph "Orchestration Layer"
        PIPE[Pipeline Orchestrator<br/>pipeline.py]
        CTX[Pipeline Context<br/>Shared state]
    end
    
    subgraph "Agent Layer"
        PROXY[UserProxyAgent<br/>Coordinator]
        DATA[DataProcessorAgent<br/>Data analysis]
        MODEL[ModelBuilderAgent<br/>Code generation]
        EXEC[CodeExecutorAgent<br/>Safe execution]
        ANALYST[AnalystAgent<br/>Quality control]
        PRED[PredictionAgent<br/>Predictions]
        VIZ[VisualizationAgent<br/>Charts]
    end
    
    subgraph "Execution Layer"
        DOCKER[Docker Container<br/>H2O + Python]
        H2O[H2O AutoML<br/>Model training]
    end
    
    subgraph "Storage Layer"
        DB[(SQLite Database<br/>automl_system.db)]
        FILES[File System<br/>Models, Datasets, Results]
    end
    
    %% Connections
    UI --> API
    API --> PIPE
    PIPE --> CTX
    PIPE --> PROXY
    
    PROXY --> DATA
    PROXY --> MODEL
    PROXY --> EXEC
    PROXY --> ANALYST
    PROXY --> PRED
    PROXY --> VIZ
    
    EXEC --> DOCKER
    DOCKER --> H2O
    
    PIPE --> DB
    PIPE --> FILES
    
    CTX -.-> DATA
    CTX -.-> MODEL
    CTX -.-> EXEC
    CTX -.-> ANALYST
    CTX -.-> PRED
    CTX -.-> VIZ
```

## Detailed Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant WebUI
    participant API
    participant Pipeline
    participant DataAgent as DataProcessorAgent
    participant ModelAgent as ModelBuilderAgent
    participant ExecAgent as CodeExecutorAgent
    participant AnalystAgent
    participant PredAgent as PredictionAgent
    participant VizAgent as VisualizationAgent
    participant Docker
    participant H2O
    
    User->>WebUI: Upload CSV + Define objective
    WebUI->>API: POST /pipeline/start
    API->>Pipeline: initialize_pipeline()
    
    Pipeline->>DataAgent: analyze_dataset()
    DataAgent->>DataAgent: detect_file_format()
    DataAgent->>DataAgent: extract_metadata()
    DataAgent->>Pipeline: return analysis_result
    
    Pipeline->>ModelAgent: generate_training_code()
    ModelAgent->>ModelAgent: create_h2o_script()
    ModelAgent->>Pipeline: return python_script
    
    Pipeline->>ExecAgent: execute_code()
    ExecAgent->>Docker: run_in_container()
    Docker->>H2O: train_automl_models()
    H2O->>Docker: return trained_models
    Docker->>ExecAgent: return execution_logs
    ExecAgent->>Pipeline: return results
    
    Pipeline->>AnalystAgent: validate_results()
    AnalystAgent->>AnalystAgent: check_model_quality()
    AnalystAgent->>Pipeline: return validation_report
    
    alt If validation passes
        Pipeline->>PredAgent: generate_predictions()
        PredAgent->>Docker: run_prediction_script()
        Docker->>PredAgent: return predictions
        PredAgent->>Pipeline: return prediction_results
        
        Pipeline->>VizAgent: create_visualizations()
        VizAgent->>Docker: run_visualization_script()
        Docker->>VizAgent: return charts
        VizAgent->>Pipeline: return visualization_files
    end
    
    Pipeline->>API: pipeline_complete
    API->>WebUI: return final_results
    WebUI->>User: Display results + download links
```

## Agent Communication Pattern

```mermaid
graph LR
    subgraph "Communication Pattern"
        PROXY[UserProxyAgent<br/>Orchestrator]
        
        subgraph "Specialized Agents"
            DATA[DataProcessorAgent]
            MODEL[ModelBuilderAgent]
            EXEC[CodeExecutorAgent]
            ANALYST[AnalystAgent]
            PRED[PredictionAgent]
            VIZ[VisualizationAgent]
        end
        
        subgraph "Shared Context"
            CTX[Pipeline Context<br/>- file_path<br/>- objective<br/>- analysis_results<br/>- model_info<br/>- predictions]
        end
    end
    
    PROXY -.->|"Reads/Writes"| CTX
    DATA -.->|"Updates"| CTX
    MODEL -.->|"Reads"| CTX
    EXEC -.->|"Updates"| CTX
    ANALYST -.->|"Reads"| CTX
    PRED -.->|"Reads"| CTX
    VIZ -.->|"Reads"| CTX
    
    PROXY -->|"Task assignment"| DATA
    PROXY -->|"Generate code"| MODEL
    PROXY -->|"Execute script"| EXEC
    PROXY -->|"Validate results"| ANALYST
    PROXY -->|"Make predictions"| PRED
    PROXY -->|"Create charts"| VIZ
```

## Data Flow Architecture

```mermaid
flowchart TD
    subgraph "Input Layer"
        CSV[CSV Dataset]
        OBJ[User Objective]
    end
    
    subgraph "Processing Pipeline"
        A1[Data Analysis]
        A2[Feature Engineering]
        A3[Model Training]
        A4[Model Validation]
        A5[Predictions]
        A6[Visualizations]
    end
    
    subgraph "Storage Layer"
        DB[(Database)]
        FS[File System]
        MODELS[Model Store]
    end
    
    subgraph "Output Layer"
        RESULTS[Results Dashboard]
        DOWNLOADS[Download Files]
        API_OUT[API Responses]
    end
    
    CSV --> A1
    OBJ --> A1
    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> A5
    A5 --> A6
    
    A1 --> DB
    A3 --> MODELS
    A4 --> FS
    A5 --> FS
    A6 --> FS
    
    DB --> RESULTS
    FS --> DOWNLOADS
    MODELS --> API_OUT
```

## Docker Execution Environment

```mermaid
graph TB
    subgraph "Host System"
        API[FastAPI Application]
        AGENTS[Agent System]
    end
    
    subgraph "Docker Container"
        PYTHON[Python 3.10]
        H2O_LIB[H2O AutoML Library]
        PANDAS[Pandas + NumPy]
        MATPLOTLIB[Matplotlib + Seaborn]
        WORKSPACE[/workspace<br/>Mounted Volume]
    end
    
    subgraph "Execution Flow"
        SCRIPT[Generated Script]
        EXECUTION[Script Execution]
        RESULTS[Output Files]
        LOGS[Execution Logs]
    end
    
    AGENTS -->|"Generate & Send"| SCRIPT
    SCRIPT --> EXECUTION
    EXECUTION --> RESULTS
    EXECUTION --> LOGS
    
    API -.->|"Volume Mount"| WORKSPACE
    WORKSPACE --> SCRIPT
    RESULTS --> WORKSPACE
    LOGS --> WORKSPACE
```

## State Management

```mermaid
stateDiagram-v2
    [*] --> Initializing
    
    Initializing --> DataAnalysis : Upload dataset
    DataAnalysis --> ModelGeneration : Analysis complete
    DataAnalysis --> Error : Analysis failed
    
    ModelGeneration --> CodeExecution : Code generated
    ModelGeneration --> Error : Generation failed
    
    CodeExecution --> QualityCheck : Execution successful
    CodeExecution --> Error : Execution failed
    
    QualityCheck --> PredictionGeneration : Validation passed
    QualityCheck --> ModelGeneration : Validation failed (retry)
    QualityCheck --> Error : Critical validation failure
    
    PredictionGeneration --> Visualization : Predictions ready
    PredictionGeneration --> Error : Prediction failed
    
    Visualization --> Completed : Charts created
    Visualization --> Error : Visualization failed
    
    Completed --> [*] : Results delivered
    Error --> [*] : Error reported
    
    Error --> DataAnalysis : Retry from analysis
    Error --> ModelGeneration : Retry from model
```

## Security Architecture

```mermaid
graph TB
    subgraph "Security Layers"
        AUTH[Authentication Layer]
        VALID[Input Validation]
        SANDBOX[Docker Sandbox]
        MONITOR[Process Monitoring]
    end
    
    subgraph "Threat Mitigation"
        TIMEOUT[Execution Timeouts]
        RESOURCE[Resource Limits]
        ISOLATION[Container Isolation]
        LOGGING[Security Logging]
    end
    
    USER[User Input] --> AUTH
    AUTH --> VALID
    VALID --> SANDBOX
    SANDBOX --> MONITOR
    
    MONITOR --> TIMEOUT
    MONITOR --> RESOURCE
    MONITOR --> ISOLATION
    MONITOR --> LOGGING
```

## Database Schema

```mermaid
erDiagram
    PIPELINES {
        string id PK
        string name
        string status
        datetime created_at
        datetime updated_at
        text user_objective
        text file_path
        text results
        text error_message
    }
    
    PIPELINE_LOGS {
        string id PK
        string pipeline_id FK
        string agent_name
        string level
        text message
        datetime timestamp
    }
    
    MODELS {
        string id PK
        string pipeline_id FK
        string model_name
        string model_path
        text metrics
        datetime created_at
    }
    
    PREDICTIONS {
        string id PK
        string model_id FK
        text input_data
        text predictions
        datetime created_at
    }
    
    PIPELINES ||--o{ PIPELINE_LOGS : has
    PIPELINES ||--o{ MODELS : generates
    MODELS ||--o{ PREDICTIONS : makes
```

## Performance Monitoring

```mermaid
graph TB
    subgraph "Metrics Collection"
        TIME[Execution Time]
        MEMORY[Memory Usage]
        CPU[CPU Usage]
        DISK[Disk I/O]
    end
    
    subgraph "Agent Performance"
        DATA_TIME[Data Analysis Time]
        MODEL_TIME[Model Training Time]
        PRED_TIME[Prediction Time]
        VIZ_TIME[Visualization Time]
    end
    
    subgraph "System Health"
        QUEUE[Pipeline Queue Length]
        ACTIVE[Active Containers]
        ERROR_RATE[Error Rate]
        SUCCESS_RATE[Success Rate]
    end
    
    TIME --> MONITORING[Performance Dashboard]
    MEMORY --> MONITORING
    CPU --> MONITORING
    DISK --> MONITORING
    
    DATA_TIME --> MONITORING
    MODEL_TIME --> MONITORING
    PRED_TIME --> MONITORING
    VIZ_TIME --> MONITORING
    
    QUEUE --> MONITORING
    ACTIVE --> MONITORING
    ERROR_RATE --> MONITORING
    SUCCESS_RATE --> MONITORING
```
