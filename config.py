import os
from dotenv import load_dotenv

load_dotenv()

class Config:

    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "automl_system.db")
    
    # Docker Configuration
    DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "my-autogen-h2o:latest")
    CODE_EXECUTOR_TIMEOUT = int(os.getenv("CODE_EXECUTOR_TIMEOUT", "3000000"))
    
    # File Storage
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
    MODELS_DIR = os.getenv("MODELS_DIR", "models")
    RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
    CODING_DIR = os.getenv("CODING_DIR", "coding")
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8006"))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    HF_TOKEN = os.getenv("HF_TOKEN")

    # Ollama LLM Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")