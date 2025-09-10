# Installation and Configuration

## 🎯 System Requirements

### **Minimum Requirements**
- **Operating System**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **RAM**: 8 GB (16 GB recommended)
- **Disk Space**: 10 GB free
- **Processor**: Intel i5 or AMD Ryzen 5 (4 cores minimum)
- **Internet Connection**: For initial downloads and models

### **Required Software**

#### **Mandatory**
- **Python 3.8+** with pip
- **Docker Desktop** (latest version)
- **Git** for repository cloning

#### **LLM Options (choose one)**
- **Option A**: **Ollama** (recommended for local use)
- **Option B**: **Hugging Face API Key** (for cloud use)

## 🚀 Quick Installation

### **Step 1: Clone Repository**
```bash
# Clone the project
git clone https://github.com/your-repo/Agents2ML.git
cd Agents2ML

# Verify content
ls -la
```

### **Step 2: Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify activation
which python  # Should show virtual environment path
```

### **Step 3: Install Dependencies**
```bash
# Install main dependencies
pip install -r requirements.txt

# Verify critical installation
pip show fastapi uvicorn h2o autogen-agentchat
```

### **Step 4: Configure Docker**
```bash
# Verify Docker is working
docker --version
docker run hello-world

# Verify Docker Compose
docker-compose --version
```

## 🧠 Language Model Configuration

### **Option A: Ollama (Recommended for Local)**

#### **Install Ollama**
```bash
# Windows (PowerShell as Administrator)
winget install Ollama.Ollama

# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

#### **Download Model**
```bash
# Start Ollama
ollama serve

# In another terminal, download model (may take time)
ollama run gpt-oss:120b

# Verify model installed
ollama list
```

#### **Test Connection**
```bash
# Test that Ollama works
curl http://localhost:11434/api/generate -d '{
  "model": "gpt-oss:120b",
  "prompt": "Hello, world!",
  "stream": false
}'
```

### **Option B: Hugging Face API**

#### **Get API Key**
1. Go to [Hugging Face](https://huggingface.co/)
2. Create account or sign in
3. Go to Settings → Access Tokens
4. Create new token with read permissions

#### **Configure Environment Variables**
```bash
# Create .env file
cp .env.example .env

# Edit .env with your token
echo "HF_TOKEN=your_hugging_face_token_here" > .env
```

## ⚙️ System Configuration

### **Main Configuration File**
```python
# config.py - Custom configuration

# LLM Configuration
LLM_CONFIG = {
    "primary_provider": "ollama",  # or "huggingface"
    "model_name": "gpt-oss:120b",
    "ollama_url": "http://localhost:11434",
    "max_tokens": 4000,
    "temperature": 0.1
}

# Docker Configuration
DOCKER_CONFIG = {
    "enabled": True,
    "timeout": 1800,  # 30 minutes
    "memory_limit": "2g",
    "cpu_limit": 2
}

# Application Configuration
APP_CONFIG = {
    "host": "0.0.0.0",
    "port": 8006,
    "debug": False,
    "log_level": "INFO"
}

# H2O Configuration
H2O_CONFIG = {
    "max_models": 20,
    "max_runtime_secs": 1800,
    "nfolds": 5,
    "seed": 42
}
```

### **Database Configuration**
```bash
# Initialize database
python database_init.py

# Verify tables created
sqlite3 automl_system.db ".tables"
```

## 🧪 Installation Verification

### **Verification Script**
```bash
# Create test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
import sys
import subprocess
import requests
from pathlib import Path

def check_python():
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version OK:", sys.version)
        return True
    else:
        print("❌ Python version too old:", sys.version)
        return False

def check_docker():
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker OK:", result.stdout.strip())
            return True
        else:
            print("❌ Docker not working")
            return False
    except FileNotFoundError:
        print("❌ Docker not installed")
        return False

def check_ollama():
    try:
        response = requests.get('http://localhost:11434/api/tags', 
                              timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            gpt_oss = any('gpt-oss:120b' in model.get('name', '') 
                         for model in models)
            if gpt_oss:
                print("✅ Ollama and gpt-oss:120b OK")
                return True
            else:
                print("⚠️  Ollama OK, but missing gpt-oss:120b")
                return False
        else:
            print("❌ Ollama not responding")
            return False
    except:
        print("⚠️  Ollama not working (use Hugging Face)")
        return False

def check_files():
    required_files = [
        'app.py', 'pipeline.py', 'config.py', 
        'requirements.txt', 'agents/', 'static/'
    ]
    all_present = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} missing")
            all_present = False
    return all_present

if __name__ == "__main__":
    print("🔍 Verifying installation...")
    print()
    
    checks = [
        check_python(),
        check_docker(),
        check_ollama(),
        check_files()
    ]
    
    if all(checks):
        print("\n🎉 Installation complete and correct!")
        print("Run: python start.py")
    else:
        print("\n❌ There are problems with the installation")
        print("Review items marked with ❌")
EOF

# Execute verification
python test_installation.py
```

## 🚀 First Launch

### **Start System**
```bash
# Method 1: Start script
python start.py

# Method 2: Direct with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8006 --reload
```

### **Verify It Works**
```bash
# Test API
curl http://localhost:8006/health

# Expected response:
# {"status": "healthy", "agents": 7, "llm": "connected"}
```

### **Access Web Interface**
1. Open browser
2. Go to `http://localhost:8006`
3. Should see main dashboard

## 🔧 Advanced Configuration

### **Complete Environment Variables**
```bash
# .env - Complete configuration
# LLM Configuration
LLM_PROVIDER=ollama  # or 'huggingface'
HF_TOKEN=your_token_here
OLLAMA_URL=http://localhost:11434
MODEL_NAME=gpt-oss:120b

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8006
DEBUG=False
LOG_LEVEL=INFO

# Docker Configuration
DOCKER_ENABLED=True
DOCKER_TIMEOUT=1800
DOCKER_MEMORY_LIMIT=2g
DOCKER_CPU_LIMIT=2

# H2O Configuration
H2O_MAX_MODELS=20
H2O_MAX_RUNTIME=1800
H2O_NFOLDS=5

# Database Configuration
DATABASE_URL=sqlite:///automl_system.db
DATABASE_ECHO=False

# Security
SECRET_KEY=your_secret_key_here
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8006"]
```

### **Logging Configuration**
```python
# logging_config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '[{asctime}] {levelname} in {module}: {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logs/automl_system.log',
            'formatter': 'default',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file'],
    },
}
```

## 🐳 Docker Installation

### **Docker Compose Setup**
```yaml
# docker-compose.yml
version: '3.8'

services:
  automl-system:
    build: .
    ports:
      - "8006:8006"
    volumes:
      - ./uploads:/app/uploads
      - ./models:/app/models
      - ./results:/app/results
      - ./logs:/app/logs
    environment:
      - LLM_PROVIDER=huggingface  # Recommended for Docker
      - HF_TOKEN=${HF_TOKEN}
      - DATABASE_URL=sqlite:///data/automl_system.db
    depends_on:
      - ollama
    networks:
      - automl-network

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    networks:
      - automl-network

volumes:
  ollama-data:

networks:
  automl-network:
    driver: bridge
```

### **Build and Run**
```bash
# Build and run
docker-compose up --build

# In background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🔍 Common Troubleshooting

### **Problem: Ollama doesn't start**
```bash
# Check if port is occupied
netstat -tulpn | grep 11434

# Restart Ollama
pkill ollama
ollama serve
```

### **Problem: Docker doesn't work**
```bash
# Check Docker status
systemctl status docker

# Restart Docker
sudo systemctl restart docker

# Check permissions (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### **Problem: Port 8006 occupied**
```bash
# Find process using port
lsof -i :8006

# Change port in config.py
APP_CONFIG = {
    "port": 8007  # Alternative port
}
```

### **Problem: Insufficient memory for H2O**
```python
# In config.py - Reduce memory usage
H2O_CONFIG = {
    "max_models": 5,  # Reduce models
    "max_runtime_secs": 900,  # Reduce time
    "nfolds": 3  # Reduce cross-validation
}
```

## ✅ Post-Installation Checklist

- [ ] Python 3.8+ installed and working
- [ ] Docker Desktop working
- [ ] Ollama + gpt-oss:120b OR Hugging Face token configured
- [ ] Python dependencies installed
- [ ] Database initialized
- [ ] Ports 8006 and 11434 available
- [ ] Verification script executed without errors
- [ ] Web interface accessible at http://localhost:8006
- [ ] API responding at /health endpoint

## 🎉 Installation Completed!

If all checklist items are marked, congratulations! The system is ready to use.

**Next step**: [Quick Start Guide](04_quick_start_en.md) to create your first model.

---

**Installation problems?** Check the [Troubleshooting](tutorials/troubleshooting_en.md) section for detailed solutions.