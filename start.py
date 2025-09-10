#!/usr/bin/env python3
"""
AutoML Training System Startup Script
"""
import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required files and directories exist"""
    required_files = [
        "app.py",
        "pipeline.py",
        "requirements.txt",
        "static/index.html",
        "static/app.js"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Missing required file: {file_path}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["uploads", "models", "results", "coding", "static"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory created/verified: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False
    return True

def check_docker():
    """Check if Docker is available and running"""
    try:
        subprocess.check_output(["docker", "--version"])
        print("✅ Docker is available")
        
        # Check if the required Docker image exists
        result = subprocess.run(["docker", "images", "-q", "my-autogen-h2o:latest"], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ H2O Docker image found")
        else:
            print("⚠️  H2O Docker image not found. You may need to build it:")
            print("   docker build -t my-autogen-h2o:latest .")
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Docker not found or not running. Some features may not work.")
        return False
    
    return True

def check_environment():
    """Check environment configuration"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("⚠️  .env file not found. Creating from template...")
        env_file.write_text(env_example.read_text())
        print("✅ .env file created. Please update it with your configuration.")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("⚠️  OpenAI API key not configured. Please set OPENAI_API_KEY in .env file")

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting AutoML Training System...")
    print("📊 Dashboard will be available at: http://localhost:8006")
    print("📝 API documentation at: http://localhost:8006/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        import uvicorn
        from config import Config
        uvicorn.run("app:app", host=Config.HOST, port=Config.PORT, reload=Config.DEBUG)
    except ImportError:
        print("❌ uvicorn not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn[standard]"])
        import uvicorn
        from config import Config
        uvicorn.run("app:app", host=Config.HOST, port=Config.PORT, reload=Config.DEBUG)

def main():
    """Main startup function"""
    print("🤖 AutoML Training System - Startup Check")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("❌ Startup failed: Missing required files")
        return 1
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if "--skip-install" not in sys.argv:
        if not install_dependencies():
            return 1
    
    # Check Docker
    check_docker()
    
    # Check environment
    check_environment()
    
    print("\n" + "=" * 50)
    print("✅ All checks completed!")
    print("=" * 50 + "\n")
    
    # Start server
    start_server()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())