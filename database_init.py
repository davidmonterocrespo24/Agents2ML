"""
Database initialization module for AutoML Training System
Contains functions to create and initialize all database tables
"""

import sqlite3
from pathlib import Path


def init_database(db_path: str = "automl_system.db"):
    """
    Initialize the database with all required tables
    
    Args:
        db_path (str): Path to the SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Jobs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            prompt TEXT NOT NULL,
            dataset_path TEXT NOT NULL,
            status TEXT DEFAULT 'created',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            progress INTEGER DEFAULT 0,
            error_message TEXT,
            target_column TEXT,
            parent_job_id TEXT DEFAULT NULL,
            version_number INTEGER DEFAULT 1,
            is_parent BOOLEAN DEFAULT 1,
            FOREIGN KEY (parent_job_id) REFERENCES jobs (id)
        )
    """)

    # Models table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            name TEXT NOT NULL,
            model_path TEXT NOT NULL,
            metrics TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs (id)
        )
    """)

    # Migration: Add updated_at column to models table if it doesn't exist
    try:
        cursor.execute("SELECT updated_at FROM models LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, add it
        cursor.execute("ALTER TABLE models ADD COLUMN updated_at TIMESTAMP")
        # Update existing records with current timestamp
        cursor.execute("UPDATE models SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL")
        print("[DEBUG] Added updated_at column to models table")

    # Logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            message TEXT NOT NULL,
            level TEXT DEFAULT 'INFO',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs (id)
        )
    """)

    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prediction_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs (id),
            FOREIGN KEY (model_id) REFERENCES models (id)
        )
    """)

    # Agent messages table  
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            content TEXT NOT NULL,
            message_type TEXT DEFAULT 'agent',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source TEXT DEFAULT 'agent',
            FOREIGN KEY (job_id) REFERENCES jobs (id)
        )
    """)

    # Process reports table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS process_reports (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs (id)
        )
    """)

    # Generated scripts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS generated_scripts (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            script_name TEXT NOT NULL,
            script_type TEXT NOT NULL,
            script_content TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            execution_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs (id)
        )
    """)

    # Agent statistics table for tracking token usage and call frequency
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            tokens_consumed INTEGER DEFAULT 0,
            calls_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            total_execution_time REAL DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs (id),
            UNIQUE(job_id, agent_name)
        )
    """)

    # Database connections table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS database_connections (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            db_type TEXT NOT NULL,
            host TEXT NOT NULL,
            port INTEGER NOT NULL,
            database_name TEXT NOT NULL,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # SQL datasets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sql_datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            connection_id TEXT NOT NULL,
            sql_query TEXT NOT NULL,
            row_count INTEGER DEFAULT 0,
            column_count INTEGER DEFAULT 0,
            file_size_mb REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            generation_type TEXT DEFAULT 'manual',
            agent_prompt TEXT,
            FOREIGN KEY (connection_id) REFERENCES database_connections (id)
        )
    """)

    # Migration: Add new columns to sql_datasets table if they don't exist
    try:
        cursor.execute("SELECT generation_type FROM sql_datasets LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, add it
        cursor.execute("ALTER TABLE sql_datasets ADD COLUMN generation_type TEXT DEFAULT 'manual'")
        cursor.execute("ALTER TABLE sql_datasets ADD COLUMN agent_prompt TEXT")

    conn.commit()
    conn.close()


def create_directories():
    """Create necessary directories for the application"""
    directories = ["uploads", "models", "results", "coding", "datasets"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"[DEBUG] Directory '{directory}' created/verified")


def initialize_application(db_path: str = "automl_system.db"):
    """
    Initialize the complete application: database and directories
    
    Args:
        db_path (str): Path to the SQLite database file
    """
    print("[INFO] Initializing database...")
    init_database(db_path)
    
    print("[INFO] Creating necessary directories...")
    create_directories()
    
    print("[INFO] Application initialization completed successfully")


if __name__ == "__main__":
    # Allow running this script directly to initialize the database
    initialize_application()
