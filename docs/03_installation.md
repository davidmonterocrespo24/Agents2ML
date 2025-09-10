# Instalación y Configuración

## 🎯 Requisitos del Sistema

### **Requisitos Mínimos**
- **Sistema Operativo**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **Memoria RAM**: 8 GB (recomendado 16 GB)
- **Espacio en Disco**: 10 GB libres
- **Procesador**: Intel i5 o AMD Ryzen 5 (4 núcleos mínimo)
- **Conexión a Internet**: Para descargas iniciales y modelos

### **Software Requerido**

#### **Obligatorio**
- **Python 3.8+** con pip
- **Docker Desktop** (última versión)
- **Git** para clonación del repositorio

#### **Opciones de LLM (elegir una)**
- **Opción A**: **Ollama** (recomendado para uso local)
- **Opción B**: **Hugging Face API Key** (para uso cloud)

## 🚀 Instalación Rápida

### **Paso 1: Clonar el Repositorio**
```bash
# Clonar el proyecto
git clone https://github.com/your-repo/Agents2ML.git
cd Agents2ML

# Verificar contenido
ls -la
```

### **Paso 2: Crear Entorno Virtual**
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Verificar activación
which python  # Debe mostrar la ruta del entorno virtual
```

### **Paso 3: Instalar Dependencias**
```bash
# Instalar dependencias principales
pip install -r requirements.txt

# Verificar instalación crítica
pip show fastapi uvicorn h2o autogen-agentchat
```

### **Paso 4: Configurar Docker**
```bash
# Verificar Docker está funcionando
docker --version
docker run hello-world

# Verificar Docker Compose
docker-compose --version
```

## 🧠 Configuración del Modelo de Lenguaje

### **Opción A: Ollama (Recomendado para Local)**

#### **Instalar Ollama**
```bash
# Windows (PowerShell como Administrador)
winget install Ollama.Ollama

# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

#### **Descargar el Modelo**
```bash
# Iniciar Ollama
ollama serve

# En otra terminal, descargar el modelo (puede tomar tiempo)
ollama run gpt-oss:120b

# Verificar modelo instalado
ollama list
```

#### **Probar Conexión**
```bash
# Probar que Ollama funciona
curl http://localhost:11434/api/generate -d '{
  "model": "gpt-oss:120b",
  "prompt": "Hello, world!",
  "stream": false
}'
```

### **Opción B: Hugging Face API**

#### **Obtener API Key**
1. Ir a [Hugging Face](https://huggingface.co/)
2. Crear cuenta o iniciar sesión
3. Ir a Settings → Access Tokens
4. Crear nuevo token con permisos de lectura

#### **Configurar Variables de Entorno**
```bash
# Crear archivo .env
cp .env.example .env

# Editar .env con tu token
echo "HF_TOKEN=tu_hugging_face_token_aqui" > .env
```

## ⚙️ Configuración del Sistema

### **Archivo de Configuración Principal**
```python
# config.py - Configuración personalizada

# Configuración del LLM
LLM_CONFIG = {
    "primary_provider": "ollama",  # o "huggingface"
    "model_name": "gpt-oss:120b",
    "ollama_url": "http://localhost:11434",
    "max_tokens": 4000,
    "temperature": 0.1
}

# Configuración de Docker
DOCKER_CONFIG = {
    "enabled": True,
    "timeout": 1800,  # 30 minutos
    "memory_limit": "2g",
    "cpu_limit": 2
}

# Configuración de la aplicación
APP_CONFIG = {
    "host": "0.0.0.0",
    "port": 8006,
    "debug": False,
    "log_level": "INFO"
}

# Configuración de H2O
H2O_CONFIG = {
    "max_models": 20,
    "max_runtime_secs": 1800,
    "nfolds": 5,
    "seed": 42
}
```

### **Configuración de Base de Datos**
```bash
# Inicializar base de datos
python database_init.py

# Verificar tablas creadas
sqlite3 automl_system.db ".tables"
```

## 🧪 Verificación de la Instalación

### **Script de Verificación**
```bash
# Crear script de prueba
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
                print("✅ Ollama y gpt-oss:120b OK")
                return True
            else:
                print("⚠️  Ollama OK, pero falta gpt-oss:120b")
                return False
        else:
            print("❌ Ollama no responde")
            return False
    except:
        print("⚠️  Ollama no está funcionando (usa Hugging Face)")
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
            print(f"❌ {file} faltante")
            all_present = False
    return all_present

if __name__ == "__main__":
    print("🔍 Verificando instalación...")
    print()
    
    checks = [
        check_python(),
        check_docker(),
        check_ollama(),
        check_files()
    ]
    
    if all(checks):
        print("\n🎉 ¡Instalación completa y correcta!")
        print("Ejecutar: python start.py")
    else:
        print("\n❌ Hay problemas con la instalación")
        print("Revisar los elementos marcados con ❌")
EOF

# Ejecutar verificación
python test_installation.py
```

## 🚀 Primer Inicio

### **Iniciar el Sistema**
```bash
# Método 1: Script de inicio
python start.py

# Método 2: Directamente con uvicorn
uvicorn app:app --host 0.0.0.0 --port 8006 --reload
```

### **Verificar que Funciona**
```bash
# Probar API
curl http://localhost:8006/health

# Respuesta esperada:
# {"status": "healthy", "agents": 7, "llm": "connected"}
```

### **Acceder a la Interfaz Web**
1. Abrir navegador
2. Ir a `http://localhost:8006`
3. Debería ver el dashboard principal

## 🔧 Configuración Avanzada

### **Variables de Entorno Completas**
```bash
# .env - Configuración completa
# LLM Configuration
LLM_PROVIDER=ollama  # o 'huggingface'
HF_TOKEN=tu_token_aqui
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
SECRET_KEY=tu_secret_key_aqui
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8006"]
```

### **Configuración de Logging**
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

## 🐳 Instalación con Docker

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
      - LLM_PROVIDER=huggingface  # Recomendado para Docker
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

### **Construcción y Ejecución**
```bash
# Construir y ejecutar
docker-compose up --build

# En background
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar servicios
docker-compose down
```

## 🔍 Solución de Problemas Comunes

### **Problema: Ollama no inicia**
```bash
# Verificar si el puerto está ocupado
netstat -tulpn | grep 11434

# Reiniciar Ollama
pkill ollama
ollama serve
```

### **Problema: Docker no funciona**
```bash
# Verificar estado de Docker
systemctl status docker

# Reiniciar Docker
sudo systemctl restart docker

# Verificar permisos (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### **Problema: Puerto 8006 ocupado**
```bash
# Encontrar proceso usando el puerto
lsof -i :8006

# Cambiar puerto en config.py
APP_CONFIG = {
    "port": 8007  # Puerto alternativo
}
```

### **Problema: Falta memoria para H2O**
```python
# En config.py - Reducir uso de memoria
H2O_CONFIG = {
    "max_models": 5,  # Reducir modelos
    "max_runtime_secs": 900,  # Reducir tiempo
    "nfolds": 3  # Reducir validación cruzada
}
```

## ✅ Lista de Verificación Post-Instalación

- [ ] Python 3.8+ instalado y funcionando
- [ ] Docker Desktop funcionando
- [ ] Ollama + gpt-oss:120b OR Hugging Face token configurado
- [ ] Dependencias Python instaladas
- [ ] Base de datos inicializada
- [ ] Puertos 8006 y 11434 disponibles
- [ ] Script de verificación ejecutado sin errores
- [ ] Interfaz web accesible en http://localhost:8006
- [ ] API respondiendo en /health endpoint

## 🎉 ¡Instalación Completada!

Si todos los elementos de la lista están marcados, ¡felicitaciones! El sistema está listo para usar.

**Siguiente paso**: [Guía de Inicio Rápido](04_quick_start.md) para crear tu primer modelo.

---

**¿Problemas con la instalación?** Consulta la sección de [Troubleshooting](tutorials/troubleshooting.md) para soluciones detalladas.