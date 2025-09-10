import asyncio
import base64
import hashlib
import io
import json
import os
import pandas as pd
import shutil
import sqlite3
import traceback
import uuid
import numpy as np
import uvicorn
import pymysql
import psycopg2
from cryptography.fernet import Fernet
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import List, Optional, Dict, Any

from agents.sql_dataset_agent import SQLDatasetAgent
from models import *
from database_init import initialize_application

# Import SQL Dataset Agent
try:
    from agents.sql_dataset_agent import sql_dataset_agent

    SQL_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] SQL Dataset Agent not available: {e}")
    SQL_AGENT_AVAILABLE = False

app = FastAPI(title="AutoML Training System")
# Database setup
DB_PATH = "automl_system.db"

# Encryption key for database passwords (in production, use environment variable)
ENCRYPTION_KEY = base64.urlsafe_b64encode(b"AutoMLSystemKey32BytesForEncryption!")[:32]
cipher_suite = Fernet(base64.urlsafe_b64encode(ENCRYPTION_KEY))


def encrypt_password(password: str) -> str:
    """Encrypt password for secure storage"""
    return cipher_suite.encrypt(password.encode()).decode()


def decrypt_password(encrypted_password: str) -> str:
    """Decrypt password for database connection"""
    return cipher_suite.decrypt(encrypted_password.encode()).decode()


def get_database_connection(connection_info: dict):
    """Get database connection based on connection info"""
    db_type = connection_info['db_type']

    try:
        if db_type == 'postgresql':
            import psycopg2
            from psycopg2.extras import RealDictCursor
            return psycopg2.connect(
                host=connection_info['host'],
                port=connection_info['port'],
                database=connection_info['database_name'],
                user=connection_info['username'],
                password=decrypt_password(connection_info['password']),
                cursor_factory=RealDictCursor
            )
        elif db_type == 'mysql':
            import pymysql
            return pymysql.connect(
                host=connection_info['host'],
                port=connection_info['port'],
                database=connection_info['database_name'],
                user=connection_info['username'],
                password=decrypt_password(connection_info['password']),
                cursorclass=pymysql.cursors.DictCursor
            )
        elif db_type == 'sqlite':
            import sqlite3
            sqlite3.Row.keys = lambda self: [column[0] for column in self.cursor.description]
            conn = sqlite3.connect(connection_info['database_name'])
            conn.row_factory = sqlite3.Row
            return conn
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Database driver not available: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")



# Initialize database and directories
initialize_application(DB_PATH)


@app.post("/jobs", response_model=dict)
async def create_job(job_data: JobCreate, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO jobs (id, name, prompt, dataset_path, target_column, status, parent_job_id, version_number, is_parent)
        VALUES (?, ?, ?, ?, ?, 'created', ?, ?, ?)
    """, (job_id, job_data.name, job_data.prompt, "", job_data.target_column, None, 1, True))

    conn.commit()
    conn.close()

    # Don't start the pipeline yet - wait for file upload
    return {"job_id": job_id, "status": "created", "message": "Job created successfully"}


@app.post("/jobs/{job_id}/upload")
async def upload_dataset(job_id: str, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")

    file_path = f"uploads/{job_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Update job with dataset path
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE jobs SET dataset_path = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (file_path, job_id))

    conn.commit()
    conn.close()

    # Now start the pipeline after file is uploaded
    background_tasks.add_task(process_job_pipeline, job_id)

    return {"message": "Dataset uploaded successfully", "file_path": file_path}


@app.get("/jobs", response_model=List[Job])
async def get_jobs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
    jobs = cursor.fetchall()
    conn.close()

    return [
        Job(
            id=job[0],
            name=job[1],
            prompt=job[2],
            dataset_path=job[3],
            status=job[4],
            created_at=job[5],
            updated_at=job[6],
            progress=job[7],
            error_message=job[8],
            target_column=job[9],
            parent_job_id=job[10] if len(job) > 10 else None,
            version_number=job[11] if len(job) > 11 else 1,
            is_parent=bool(job[12]) if len(job) > 12 else True
        ) for job in jobs
    ]


@app.get("/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    job = cursor.fetchone()
    conn.close()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return Job(
        id=job[0],
        name=job[1],
        prompt=job[2],
        dataset_path=job[3],
        status=job[4],
        created_at=job[5],
        updated_at=job[6],
        progress=job[7],
        error_message=job[8],
        target_column=job[9],
        parent_job_id=job[10] if len(job) > 10 else None,
        version_number=job[11] if len(job) > 11 else 1,
        is_parent=bool(job[12]) if len(job) > 12 else True
    )


@app.get("/jobs/{job_id}/logs", response_model=List[LogEntry])
async def get_job_logs(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, job_id, message, level, timestamp
        FROM logs WHERE job_id = ?
        ORDER BY timestamp DESC
    """, (job_id,))

    logs = cursor.fetchall()
    conn.close()

    return [
        LogEntry(
            id=log[0],
            job_id=log[1],
            message=log[2],
            level=log[3],
            timestamp=log[4]
        ) for log in logs
    ]


@app.get("/jobs/{job_id}/messages", response_model=List[AgentMessage])
async def get_job_messages(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, job_id, agent_name, content, message_type, timestamp, source
        FROM agent_messages WHERE job_id = ?
        ORDER BY timestamp ASC
    """, (job_id,))

    messages = cursor.fetchall()
    conn.close()

    return [
        AgentMessage(
            id=msg[0],
            job_id=msg[1],
            agent_name=msg[2],
            content=msg[3],
            message_type=msg[4],
            timestamp=msg[5],
            source=msg[6]
        ) for msg in messages
    ]


@app.post("/jobs/{job_id}/messages")
async def send_user_message(job_id: str, message_data: UserMessageInput):
    # Add user message to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO agent_messages (job_id, agent_name, content, message_type, source)
        VALUES (?, ?, ?, ?, ?)
    """, (job_id, "User", message_data.message, "user", "user"))

    # Check if the job was awaiting user input and resume it
    cursor.execute("SELECT status FROM jobs WHERE id = ?", (job_id,))
    job_result = cursor.fetchone()

    if job_result and job_result[0] == 'awaiting_user_input':
        # Reset status to processing to resume the pipeline
        cursor.execute("""
            UPDATE jobs SET status = 'processing', error_message = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (job_id,))

        # Log that user input was received
        cursor.execute("""
            INSERT INTO logs (job_id, message, level)
            VALUES (?, ?, ?)
        """, (job_id, f"User input received: {message_data.message[:100]}...", "INFO"))

    conn.commit()
    conn.close()

    # Return success message with appropriate status
    status_message = "Message sent to agents"
    if job_result and job_result[0] == 'awaiting_user_input':
        status_message += " - Pipeline will resume processing"

    return {"message": status_message}


@app.get("/jobs/{job_id}/reports", response_model=List[ProcessReport])
async def get_job_reports(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, job_id, stage, title, content, metadata, created_at
        FROM process_reports WHERE job_id = ?
        ORDER BY created_at ASC
    """, (job_id,))

    reports = cursor.fetchall()
    conn.close()

    return [
        ProcessReport(
            id=report[0] or str(uuid.uuid4()),
            job_id=report[1],
            stage=report[2],
            title=report[3],
            content=report[4],
            metadata=json.loads(report[5]) if report[5] else None,
            created_at=report[6]
        ) for report in reports if report[1] and report[2] and report[3] and report[4]
    ]


@app.get("/jobs/{job_id}/comprehensive-report")
async def get_comprehensive_report(job_id: str):
    """Get the comprehensive process analysis report for a job"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get the comprehensive report
        cursor.execute("""
            SELECT content, metadata, created_at
            FROM process_reports 
            WHERE job_id = ? AND stage = 'comprehensive_analysis'
            ORDER BY created_at DESC
            LIMIT 1
        """, (job_id,))

        result = cursor.fetchone()
        conn.close()

        if not result:
            # Generate report if it doesn't exist
            from tools.process_reporter import generate_process_report
            report = generate_process_report(job_id, DB_PATH)
            
            return {
                "job_id": job_id,
                "report": report,
                "generated_at": report.get("generated_at"),
                "status": "generated_on_demand"
            }
        
        content, metadata, created_at = result
        report = json.loads(content)
        
        return {
            "job_id": job_id,
            "report": report,
            "metadata": json.loads(metadata) if metadata else None,
            "generated_at": created_at,
            "status": "cached"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get comprehensive report: {str(e)}")


@app.get("/jobs/{job_id}/agent-analysis")
async def get_agent_analysis(job_id: str):
    """Get detailed analysis of agent performance and contributions"""
    try:
        from tools.process_reporter import ProcessReporter
        
        reporter = ProcessReporter(job_id, DB_PATH)
        contributions = reporter.analyze_agent_messages()
        
        # Format for frontend
        analysis = {}
        for agent_name, contrib in contributions.items():
            analysis[agent_name] = {
                "name": agent_name,
                "role": reporter._get_agent_role_description(agent_name),
                "statistics": {
                    "total_messages": len(contrib.messages),
                    "key_outputs": len(contrib.key_outputs),
                    "errors": len(contrib.errors),
                    "warnings": len(contrib.warnings),
                    "execution_time": contrib.execution_time,
                    "token_usage": contrib.token_usage
                },
                "performance_rating": reporter._calculate_agent_rating(contrib),
                "key_contributions": contrib.key_outputs[:5],  # Top 5
                "recent_messages": contrib.messages[-3:] if contrib.messages else []  # Last 3
            }
        
        return {
            "job_id": job_id,
            "agents": analysis,
            "total_agents": len(analysis),
            "analysis_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze agents: {str(e)}")


@app.get("/jobs/{job_id}/process-insights")
async def get_process_insights(job_id: str):
    """Get key insights and recommendations from the process analysis"""
    try:
        from tools.process_reporter import generate_process_report
        
        report = generate_process_report(job_id, DB_PATH)
        
        insights = {
            "job_id": job_id,
            "executive_summary": report.get("executive_summary", ""),
            "key_metrics": {
                "efficiency_score": report.get("performance_metrics", {}).get("efficiency_score", 0),
                "success_rate": report.get("performance_metrics", {}).get("success_rate", 0),
                "collaboration_quality": report.get("performance_metrics", {}).get("collaboration_quality", "Unknown"),
                "total_execution_time": report.get("pipeline_overview", {}).get("total_execution_time", 0),
                "total_messages": report.get("pipeline_overview", {}).get("total_messages_exchanged", 0)
            },
            "achievements": report.get("key_achievements", []),
            "challenges": report.get("challenges_and_resolutions", []),
            "recommendations": report.get("recommendations", []),
            "learning_type": report.get("pipeline_overview", {}).get("learning_type", "unknown"),
            "models_generated": report.get("pipeline_overview", {}).get("models_generated", 0),
            "process_flow": report.get("process_flow", [])
        }
        
        return insights

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get process insights: {str(e)}")


@app.get("/jobs/{job_id}/scripts", response_model=List[GeneratedScript])
async def get_job_scripts(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, job_id, script_name, script_type, script_content, agent_name, execution_result, created_at
        FROM generated_scripts WHERE job_id = ?
        ORDER BY created_at ASC
    """, (job_id,))

    scripts = cursor.fetchall()
    conn.close()

    return [
        GeneratedScript(
            id=script[0],
            job_id=script[1],
            script_name=script[2],
            script_type=script[3],
            script_content=script[4],
            agent_name=script[5],
            execution_result=script[6],
            created_at=script[7]
        ) for script in scripts
    ]


@app.get("/jobs/{job_id}/statistics", response_model=List[AgentStatistics])
async def get_job_agent_statistics(job_id: str):
    """Get agent statistics including token consumption and call frequency"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, job_id, agent_name, tokens_consumed, calls_count, 
               input_tokens, output_tokens, total_execution_time, last_updated
        FROM agent_statistics WHERE job_id = ?
        ORDER BY calls_count DESC, tokens_consumed DESC
    """, (job_id,))

    stats = cursor.fetchall()
    conn.close()

    return [
        AgentStatistics(
            id=stat[0],
            job_id=stat[1],
            agent_name=stat[2],
            tokens_consumed=stat[3],
            calls_count=stat[4],
            input_tokens=stat[5],
            output_tokens=stat[6],
            total_execution_time=stat[7],
            last_updated=stat[8]
        ) for stat in stats
    ]


@app.get("/jobs/{job_id}/models", response_model=List[Model])
async def get_job_models(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM models WHERE job_id = ?", (job_id,))
    models = cursor.fetchall()
    conn.close()

    return [
        Model(
            id=model[0],
            job_id=model[1],
            name=model[2],
            model_path=model[3],
            metrics=json.loads(model[4]) if model[4] else None,
            created_at=model[5]
        ) for model in models
    ]


@app.post("/models/{model_id}/update-metrics")
async def update_model_metrics(model_id: str):
    """Update metrics of an existing model with new extended functions"""
    try:
        print(f"[DEBUG] Attempting to update metrics for model ID: {model_id}")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # First, verify if the model exists and get all its information
        cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        model_row = cursor.fetchone()

        if not model_row:
            # Look for similar models for debugging
            cursor.execute("SELECT id, name FROM models LIMIT 5")
            existing_models = cursor.fetchall()
            print(f"[DEBUG] Model {model_id} not found. Existing models: {existing_models}")
            raise HTTPException(status_code=404, detail=f"Model not found. Model ID: {model_id}")

        # Get column names to facilitate debugging
        cursor.execute("PRAGMA table_info(models)")
        columns_info = cursor.fetchall()
        column_names = [col[1] for col in columns_info]
        print(f"[DEBUG] Models table columns: {column_names}")

        # Map model data
        model_data = dict(zip(column_names, model_row))
        print(f"[DEBUG] Found model: {model_data['name']} for job {model_data.get('job_id', 'unknown')}")

        job_id = model_data.get('job_id')
        model_name = model_data.get('name')
        model_path = model_data.get('model_path')
        created_at = model_data.get('created_at')

        # Get job information for target_column
        cursor.execute("SELECT target_column FROM jobs WHERE id = ?", (job_id,))
        job_info = cursor.fetchone()
        target_column = job_info[0] if job_info else "unknown"

        # Import extended metrics function from pipeline
        try:
            from pipeline import get_enhanced_model_metrics
            # Create a basic simulated context for the update
            class SimplePipelineContext:
                def __init__(self):
                    self.model_path_from_execution = None

            pipeline_context = SimplePipelineContext()
            pipeline_context.model_path_from_execution = model_path
        except ImportError as e:
            print(f"[ERROR] Import error: {e}")
            raise HTTPException(status_code=500, detail=f"Import error: {str(e)}")

        # Generate extended metrics (with simulated data for existing models)
        try:
            print(f"[DEBUG] Generating enhanced metrics for model: {model_name}")
            enhanced_metrics = get_enhanced_model_metrics(
                model_path=model_path,
                execution_time=0,  # Not available for existing models
                message_count=0,  # Not available for existing models
                agent_message_counts={},  # Not available for existing models
                target_column=target_column,
                pipeline_context=pipeline_context
            )
            print(f"[DEBUG] Enhanced metrics generated: {len(enhanced_metrics)} metrics")
        except Exception as e:
            print(f"[ERROR] Error generating enhanced metrics: {e}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error generating metrics: {str(e)}")

        # Mark that metrics were updated after creation
        enhanced_metrics["metrics_updated"] = True
        enhanced_metrics["metrics_update_timestamp"] = datetime.now().isoformat()
        enhanced_metrics["original_creation_date"] = created_at

        # Update in database
        try:
            cursor.execute("""
                UPDATE models SET metrics = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (json.dumps(enhanced_metrics), model_id))

            conn.commit()
            print(f"[DEBUG] Metrics successfully updated in database for model: {model_id}")
        except Exception as e:
            print(f"[ERROR] Database update failed: {e}")
            raise HTTPException(status_code=500, detail=f"Database update failed: {str(e)}")
        finally:
            conn.close()

        return {
            "message": "Model metrics updated successfully",
            "model_id": model_id,
            "updated_metrics_count": len(enhanced_metrics)
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error in update_model_metrics: {e}")
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/models/{model_id}/download")
async def download_model(model_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT model_path, name, job_id FROM models WHERE id = ?", (model_id,))
    result = cursor.fetchone()
    conn.close()

    if not result:
        raise HTTPException(status_code=404, detail="Model not found")

    model_path, model_name, job_id = result

    # Check if model exists at original path
    if os.path.exists(model_path):
        return FileResponse(model_path, filename=f"{model_name}.zip")

    # Try the new pipeline-specific path structure
    pipeline_model_path = Path("coding") / f"pipeline_{job_id}" / os.path.basename(model_path)
    if os.path.exists(pipeline_model_path):
        return FileResponse(str(pipeline_model_path), filename=f"{model_name}.zip")

    raise HTTPException(status_code=404, detail="Model file not found")


@app.get("/debug/models")
async def debug_models():
    """Debugging endpoint to view all models with their IDs"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get table structure
        cursor.execute("PRAGMA table_info(models)")
        columns_info = cursor.fetchall()

        # Get all models
        cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
        models = cursor.fetchall()

        conn.close()

        return {
            "table_structure": [{"name": col[1], "type": col[2]} for col in columns_info],
            "models_count": len(models),
            "models": [
                {
                    "id": model[0] if len(model) > 0 else None,
                    "job_id": model[1] if len(model) > 1 else None,
                    "name": model[2] if len(model) > 2 else None,
                    "model_path": model[3] if len(model) > 3 else None,
                    "created_at": model[4] if len(model) > 4 else None
                } for model in models
            ]
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/jobs/{job_id}/forecast_plot")
async def get_forecast_plot(job_id: str):
    """Serve the forecast plot image for a job"""
    # Try pipeline-specific path first
    pipeline_image_path = Path("coding") / f"pipeline_{job_id}" / "forecast_plot.png"
    if os.path.exists(pipeline_image_path):
        return FileResponse(str(pipeline_image_path), media_type="image/png")

    # Fallback to old structure
    old_image_path = Path("coding") / "forecast_plot.png"
    if os.path.exists(old_image_path):
        return FileResponse(str(old_image_path), media_type="image/png")

    raise HTTPException(status_code=404, detail="Forecast plot not found")


@app.get("/jobs/{job_id}/predictions_csv")
async def get_predictions_csv(job_id: str):
    """Serve the predictions CSV file for a job"""
    # Try pipeline-specific path first
    pipeline_csv_path = Path("coding") / f"pipeline_{job_id}" / "predictions.csv"
    if os.path.exists(pipeline_csv_path):
        return FileResponse(str(pipeline_csv_path), filename=f"predictions_{job_id}.csv", media_type="text/csv")

    # Fallback to old structure
    old_csv_path = Path("coding") / "predictions.csv"
    if os.path.exists(old_csv_path):
        return FileResponse(str(old_csv_path), filename=f"predictions_{job_id}.csv", media_type="text/csv")

    raise HTTPException(status_code=404, detail="Predictions CSV not found")


@app.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """Get information about available job results"""
    # Check what files are available for this job
    pipeline_dir = Path("coding") / f"pipeline_{job_id}"
    results = {
        "forecast_plot": False,
        "predictions_csv": False,
        "model_files": []
    }

    if pipeline_dir.exists():
        # Check for forecast plot
        forecast_path = pipeline_dir / "forecast_plot.png"
        results["forecast_plot"] = forecast_path.exists()

        # Check for predictions CSV
        predictions_path = pipeline_dir / "predictions.csv"
        results["predictions_csv"] = predictions_path.exists()

        # Get model files from database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, model_path FROM models WHERE job_id = ?", (job_id,))
        models = cursor.fetchall()
        conn.close()

        for model in models:
            model_id, model_name, model_path = model
            # Check if model file exists in pipeline directory
            pipeline_model_path = pipeline_dir / os.path.basename(model_path)
            if pipeline_model_path.exists():
                results["model_files"].append({
                    "id": model_id,
                    "name": model_name,
                    "exists": True
                })

    return results


@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if job exists and is failed
    cursor.execute("SELECT status FROM jobs WHERE id = ?", (job_id,))
    result = cursor.fetchone()

    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Job not found")

    # Reset job status to processing
    cursor.execute("""
        UPDATE jobs SET status = 'processing', progress = 0, error_message = NULL, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (job_id,))

    # Clear previous logs for this job
    cursor.execute("DELETE FROM logs WHERE job_id = ?", (job_id,))

    conn.commit()
    conn.close()

    # Restart the pipeline
    background_tasks.add_task(process_job_pipeline, job_id)

    return {"message": "Job retry initiated successfully"}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Delete job and related data
    cursor.execute("DELETE FROM predictions WHERE job_id = ?", (job_id,))
    cursor.execute("DELETE FROM models WHERE job_id = ?", (job_id,))
    cursor.execute("DELETE FROM logs WHERE job_id = ?", (job_id,))
    cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    conn.commit()
    conn.close()

    return {"message": "Job deleted successfully"}


# Update job name endpoint
@app.put("/jobs/{job_id}/name")
async def update_job_name(job_id: str, name_data: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if job exists
    cursor.execute("SELECT id FROM jobs WHERE id = ?", (job_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Job not found")

    # Update job name
    cursor.execute("UPDATE jobs SET name = ? WHERE id = ?", (name_data["name"], job_id))

    conn.commit()
    conn.close()

    return {"message": "Job name updated successfully"}


# Job versioning endpoints
@app.post("/jobs/{parent_job_id}/versions", response_model=dict)
async def create_job_version(parent_job_id: str, version_data: JobVersionCreate, background_tasks: BackgroundTasks):
    """Create a new version of an existing job with enriched prompt"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if parent job exists
        cursor.execute("SELECT * FROM jobs WHERE id = ? AND is_parent = 1", (parent_job_id,))
        parent_job = cursor.fetchone()

        if not parent_job:
            raise HTTPException(status_code=404, detail="Parent job not found or not a parent job")

        # Get the next version number
        cursor.execute("SELECT MAX(version_number) FROM jobs WHERE parent_job_id = ?", (parent_job_id,))
        max_version = cursor.fetchone()[0]
        next_version = (max_version or 1) + 1

        # Create new job version
        version_job_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO jobs (id, name, prompt, dataset_path, target_column, status, 
                            parent_job_id, version_number, is_parent)
            VALUES (?, ?, ?, ?, ?, 'created', ?, ?, ?)
        """, (version_job_id, version_data.name, version_data.prompt, parent_job[3],
              parent_job[9], parent_job_id, next_version, False))

        conn.commit()

        return {
            "job_id": version_job_id,
            "version_number": next_version,
            "status": "created",
            "message": f"Job version {next_version} created successfully"
        }

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create job version: {str(e)}")
    finally:
        conn.close()


@app.get("/jobs/{parent_job_id}/versions", response_model=List[Job])
async def get_job_versions(parent_job_id: str):
    """Get all versions of a job"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Get parent job first
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (parent_job_id,))
        parent_job = cursor.fetchone()

        if not parent_job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Get all versions (including parent)
        cursor.execute("""
            SELECT * FROM jobs 
            WHERE id = ? OR parent_job_id = ? 
            ORDER BY version_number ASC
        """, (parent_job_id, parent_job_id))

        jobs = cursor.fetchall()

        return [
            Job(
                id=job[0],
                name=job[1],
                prompt=job[2],
                dataset_path=job[3],
                status=job[4],
                created_at=job[5],
                updated_at=job[6],
                progress=job[7],
                error_message=job[8],
                target_column=job[9],
                parent_job_id=job[10] if len(job) > 10 else None,
                version_number=job[11] if len(job) > 11 else 1,
                is_parent=bool(job[12]) if len(job) > 12 else True
            ) for job in jobs
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job versions: {str(e)}")
    finally:
        conn.close()


@app.post("/jobs/versions/{version_job_id}/run")
async def run_job_version(version_job_id: str, background_tasks: BackgroundTasks):
    """Run a specific job version"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if version job exists and is not running
        cursor.execute("SELECT * FROM jobs WHERE id = ? AND is_parent = 0", (version_job_id,))
        version_job = cursor.fetchone()

        if not version_job:
            raise HTTPException(status_code=404, detail="Job version not found")

        if version_job[4] == 'processing':
            raise HTTPException(status_code=400, detail="Job version is already running")

        # Reset job status
        cursor.execute("""
            UPDATE jobs SET status = 'processing', progress = 0, error_message = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (version_job_id,))

        # Clear previous logs for this version
        cursor.execute("DELETE FROM logs WHERE job_id = ?", (version_job_id,))
        cursor.execute("DELETE FROM agent_messages WHERE job_id = ?", (version_job_id,))

        conn.commit()

        # Start the pipeline for this version
        background_tasks.add_task(process_job_pipeline, version_job_id)

        return {"message": f"Job version {version_job[11]} started successfully"}

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to run job version: {str(e)}")
    finally:
        conn.close()


@app.get("/jobs/{parent_job_id}/versions/comparison")
async def compare_job_versions(parent_job_id: str):
    """Compare results between different versions of a job"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Get all versions with their models and statistics
        cursor.execute("""
            SELECT j.id, j.name, j.prompt, j.version_number, j.status, j.created_at,
                   COUNT(m.id) as model_count,
                   GROUP_CONCAT(m.name) as model_names
            FROM jobs j
            LEFT JOIN models m ON j.id = m.job_id
            WHERE j.id = ? OR j.parent_job_id = ?
            GROUP BY j.id
            ORDER BY j.version_number ASC
        """, (parent_job_id, parent_job_id))

        versions = cursor.fetchall()

        if not versions:
            raise HTTPException(status_code=404, detail="Job not found")

        comparison_data = []
        for version in versions:
            # Get agent statistics for each version
            cursor.execute("""
                SELECT agent_name, SUM(calls_count) as total_calls, SUM(tokens_consumed) as total_tokens
                FROM agent_statistics 
                WHERE job_id = ?
                GROUP BY agent_name
            """, (version[0],))

            agent_stats = cursor.fetchall()

            comparison_data.append({
                "job_id": version[0],
                "name": version[1],
                "prompt": version[2][:100] + "..." if len(version[2]) > 100 else version[2],
                "version_number": version[3],
                "status": version[4],
                "created_at": version[5],
                "model_count": version[6],
                "model_names": version[7].split(",") if version[7] else [],
                "agent_statistics": [
                    {"agent_name": stat[0], "total_calls": stat[1], "total_tokens": stat[2]}
                    for stat in agent_stats
                ]
            })

        return {
            "parent_job_id": parent_job_id,
            "versions": comparison_data,
            "total_versions": len(comparison_data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare job versions: {str(e)}")
    finally:
        conn.close()


@app.delete("/jobs/versions/{version_job_id}")
async def delete_job_version(version_job_id: str):
    """Delete a specific job version"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if it's a version (not a parent)
        cursor.execute("SELECT is_parent FROM jobs WHERE id = ?", (version_job_id,))
        result = cursor.fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="Job version not found")

        if result[0]:  # is_parent = True
            raise HTTPException(status_code=400, detail="Cannot delete parent job using this endpoint")

        # Delete version and related data
        cursor.execute("DELETE FROM predictions WHERE job_id = ?", (version_job_id,))
        cursor.execute("DELETE FROM models WHERE job_id = ?", (version_job_id,))
        cursor.execute("DELETE FROM logs WHERE job_id = ?", (version_job_id,))
        cursor.execute("DELETE FROM agent_messages WHERE job_id = ?", (version_job_id,))
        cursor.execute("DELETE FROM agent_statistics WHERE job_id = ?", (version_job_id,))
        cursor.execute("DELETE FROM process_reports WHERE job_id = ?", (version_job_id,))
        cursor.execute("DELETE FROM generated_scripts WHERE job_id = ?", (version_job_id,))
        cursor.execute("DELETE FROM jobs WHERE id = ?", (version_job_id,))

        conn.commit()

        return {"message": "Job version deleted successfully"}

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete job version: {str(e)}")
    finally:
        conn.close()


def log_message(job_id: str, message: str, level: str = "INFO"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO logs (job_id, message, level)
        VALUES (?, ?, ?)
    """, (job_id, message, level))

    conn.commit()
    conn.close()


def update_job_status(job_id: str, status: str, progress: int = None, error_message: str = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if progress is not None and error_message is not None:
        cursor.execute("""
            UPDATE jobs SET status = ?, progress = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, progress, error_message, job_id))
    elif progress is not None:
        cursor.execute("""
            UPDATE jobs SET status = ?, progress = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, progress, job_id))
    elif error_message is not None:
        cursor.execute("""
            UPDATE jobs SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, error_message, job_id))
    else:
        cursor.execute("""
            UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, job_id))

    conn.commit()
    conn.close()


async def process_job_pipeline(job_id: str):
    try:
        log_message(job_id, "Starting job processing pipeline")
        update_job_status(job_id, "processing", 10)

        # Get job details
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        job_data = cursor.fetchone()
        conn.close()

        if not job_data:
            raise Exception("Job not found")

        dataset_path = job_data[3]
        target_column = job_data[9]
        prompt = job_data[2]

        if not dataset_path or not os.path.exists(dataset_path):
            raise Exception("Dataset file not found")

        log_message(job_id, f"Processing dataset: {dataset_path}")

        # Import and run the ML pipeline
        from pipeline import run_ml_pipeline
        await run_ml_pipeline(job_id, dataset_path, target_column, prompt)

    except Exception as e:
        log_message(job_id, f"Job failed: {str(e)}", "ERROR")
        update_job_status(job_id, "failed", error_message=str(e))


# Database Connections Endpoints
@app.post("/connections", response_model=dict)
async def create_connection(connection: DatabaseConnectionCreate):
    """Create a new database connection"""
    connection_id = str(uuid.uuid4())

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Encrypt the password
        encrypted_password = encrypt_password(connection.password)

        cursor.execute("""
            INSERT INTO database_connections 
            (id, name, db_type, host, port, database_name, username, password)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (connection_id, connection.name, connection.db_type, connection.host,
              connection.port, connection.database_name, connection.username, encrypted_password))

        conn.commit()
        conn.close()

        return {"id": connection_id, "message": "Database connection created successfully"}

    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Connection name already exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create connection: {str(e)}")


@app.get("/connections", response_model=List[DatabaseConnection])
async def get_connections():
    """Get all database connections"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, db_type, host, port, database_name, username, is_active, created_at, updated_at
            FROM database_connections WHERE is_active = 1
            ORDER BY created_at DESC
        """)

        connections = cursor.fetchall()
        conn.close()

        return [
            DatabaseConnection(
                id=conn[0],
                name=conn[1],
                db_type=conn[2],
                host=conn[3],
                port=conn[4],
                database_name=conn[5],
                username=conn[6],
                is_active=bool(conn[7]),
                created_at=conn[8],
                updated_at=conn[9]
            ) for conn in connections
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get connections: {str(e)}")


@app.post("/connections/{connection_id}/test")
async def test_connection(connection_id: str):
    """Test a database connection"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT db_type, host, port, database_name, username, password
            FROM database_connections WHERE id = ? AND is_active = 1
        """, (connection_id,))

        result = cursor.fetchone()
        conn.close()

        if not result:
            raise HTTPException(status_code=404, detail="Connection not found")

        connection_info = {
            'db_type': result[0],
            'host': result[1],
            'port': result[2],
            'database_name': result[3],
            'username': result[4],
            'password': result[5]
        }

        # Test the connection
        db_conn = get_database_connection(connection_info)

        if connection_info['db_type'] in ['postgresql', 'mysql']:
            cursor = db_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        elif connection_info['db_type'] == 'sqlite':
            cursor = db_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()

        db_conn.close()

        return {"status": "success", "message": "Connection test successful"}

    except Exception as e:
        return {"status": "error", "message": f"Connection test failed: {str(e)}"}


@app.post("/connections/test-data")
async def test_connection_data(connection: DatabaseConnectionCreate):
    """Test a database connection with form data (before saving)"""
    try:
        connection_info = {
            'db_type': connection.db_type,
            'host': connection.host,
            'port': connection.port,
            'database_name': connection.database_name,
            'username': connection.username,
            'password': connection.password
        }

        # Test the connection
        db_conn = get_database_connection(connection_info)

        if connection_info['db_type'] in ['postgresql', 'mysql']:
            cursor = db_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        elif connection_info['db_type'] == 'sqlite':
            cursor = db_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()

        db_conn.close()

        return {"status": "success",
                "message": f"Connection test successful! Connected to {connection.db_type} database '{connection.database_name}' on {connection.host}:{connection.port}"}

    except Exception as e:
        error_msg = str(e)

        # Provide more specific error messages
        if "authentication failed" in error_msg.lower() or "access denied" in error_msg.lower():
            return {"status": "error", "message": "Authentication failed. Please check your username and password."}
        elif "could not connect" in error_msg.lower() or "connection refused" in error_msg.lower():
            return {"status": "error",
                    "message": f"Could not connect to {connection.host}:{connection.port}. Please check the host and port."}
        elif "database" in error_msg.lower() and "does not exist" in error_msg.lower():
            return {"status": "error", "message": f"Database '{connection.database_name}' does not exist."}
        else:
            return {"status": "error", "message": f"Connection test failed: {error_msg}"}


@app.delete("/connections/{connection_id}")
async def delete_connection(connection_id: str):
    """Delete a database connection"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE database_connections SET is_active = 0, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (connection_id,))

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Connection not found")

        conn.commit()
        conn.close()

        return {"message": "Connection deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete connection: {str(e)}")


@app.put("/connections/{connection_id}")
async def update_connection(connection_id: str, connection: DatabaseConnectionUpdate):
    """Update an existing database connection"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Check if connection exists
        cursor.execute("SELECT password FROM database_connections WHERE id = ? AND is_active = 1", (connection_id,))
        existing = cursor.fetchone()

        if not existing:
            raise HTTPException(status_code=404, detail="Connection not found")

        # If password is not provided, keep the existing one
        if connection.password is None or connection.password == "":
            password_to_use = existing[0]  # Keep existing encrypted password
        else:
            password_to_use = encrypt_password(connection.password)

        # Update the connection
        cursor.execute("""
            UPDATE database_connections 
            SET name = ?, db_type = ?, host = ?, port = ?, 
                database_name = ?, username = ?, password = ?, 
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND is_active = 1
        """, (
            connection.name,
            connection.db_type,
            connection.host,
            connection.port,
            connection.database_name,
            connection.username,
            password_to_use,
            connection_id
        ))

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Connection not found or could not be updated")

        conn.commit()
        conn.close()

        return {"message": "Connection updated successfully", "connection_id": connection_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update connection: {str(e)}")


@app.get("/connections/{connection_id}")
async def get_connection(connection_id: str):
    """Get a specific database connection (without password for security)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, db_type, host, port, database_name, username, is_active, created_at, updated_at
            FROM database_connections 
            WHERE id = ? AND is_active = 1
        """, (connection_id,))

        connection = cursor.fetchone()

        if not connection:
            raise HTTPException(status_code=404, detail="Connection not found")

        conn.close()

        return {
            "id": connection[0],
            "name": connection[1],
            "db_type": connection[2],
            "host": connection[3],
            "port": connection[4],
            "database_name": connection[5],
            "username": connection[6],
            "is_active": connection[7],
            "created_at": connection[8],
            "updated_at": connection[9]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get connection: {str(e)}")


# SQL Query Execution Endpoints
@app.post("/sql/execute")
async def execute_sql_query(request: QueryExecuteRequest):
    """Execute SQL query and return results"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT db_type, host, port, database_name, username, password
            FROM database_connections WHERE id = ? AND is_active = 1
        """, (request.connection_id,))

        result = cursor.fetchone()
        conn.close()

        if not result:
            raise HTTPException(status_code=404, detail="Database connection not found")

        connection_info = {
            'db_type': result[0],
            'host': result[1],
            'port': result[2],
            'database_name': result[3],
            'username': result[4],
            'password': result[5]
        }

        # Execute the query
        db_conn = get_database_connection(connection_info)
        cursor = db_conn.cursor()

        # Add LIMIT to prevent large result sets
        limited_query = request.sql_query.strip()
        if not limited_query.upper().startswith('SELECT'):
            raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")

        if 'LIMIT' not in limited_query.upper():
            limited_query += f" LIMIT {request.limit}"

        cursor.execute(limited_query)

        # Get column names
        if connection_info['db_type'] == 'postgresql':
            columns = [desc[0] for desc in cursor.description]
        elif connection_info['db_type'] == 'mysql':
            columns = [desc[0] for desc in cursor.description]
        else:  # sqlite
            columns = [desc[0] for desc in cursor.description]

        rows = cursor.fetchall()

        # Convert rows to list of dictionaries
        if connection_info['db_type'] in ['postgresql', 'mysql']:
            data = [dict(row) for row in rows]
        else:  # sqlite
            data = [dict(zip(columns, row)) for row in rows]

        cursor.close()
        db_conn.close()

        return {
            "columns": columns,
            "data": data,
            "row_count": len(data),
            "column_count": len(columns)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


# SQL Datasets Endpoints
@app.post("/datasets/sql", response_model=dict)
async def create_sql_dataset(dataset: SqlDatasetCreate):
    """Create and save a SQL dataset"""
    dataset_id = str(uuid.uuid4())

    try:
        # First, execute the query to get metadata
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT db_type, host, port, database_name, username, password
            FROM database_connections WHERE id = ? AND is_active = 1
        """, (dataset.connection_id,))

        result = cursor.fetchone()

        if not result:
            conn.close()
            raise HTTPException(status_code=404, detail="Database connection not found")

        connection_info = {
            'db_type': result[0],
            'host': result[1],
            'port': result[2],
            'database_name': result[3],
            'username': result[4],
            'password': result[5]
        }

        # Execute the query to get data
        db_conn = get_database_connection(connection_info)
        db_cursor = db_conn.cursor()

        if not dataset.sql_query.strip().upper().startswith('SELECT'):
            raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")

        db_cursor.execute(dataset.sql_query)
        rows = db_cursor.fetchall()

        # Get metadata
        if connection_info['db_type'] == 'postgresql':
            columns = [desc[0] for desc in db_cursor.description]
            data = [dict(row) for row in rows]
        elif connection_info['db_type'] == 'mysql':
            columns = [desc[0] for desc in db_cursor.description]
            data = [dict(row) for row in rows]
        else:  # sqlite
            columns = [desc[0] for desc in db_cursor.description]
            data = [dict(zip(columns, row)) for row in rows]

        db_cursor.close()
        db_conn.close()

        # Convert to CSV and calculate file size
        df = pd.DataFrame(data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        file_size_mb = len(csv_content.encode('utf-8')) / (1024 * 1024)

        # Save dataset metadata
        cursor.execute("""
            INSERT INTO sql_datasets 
            (id, name, connection_id, sql_query, row_count, column_count, file_size_mb)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (dataset_id, dataset.name, dataset.connection_id, dataset.sql_query,
              len(data), len(columns), file_size_mb))

        # Save the dataset as CSV file
        datasets_dir = Path("datasets")
        datasets_dir.mkdir(exist_ok=True)
        csv_file_path = datasets_dir / f"{dataset_id}.csv"
        df.to_csv(csv_file_path, index=False)

        conn.commit()
        conn.close()

        return {
            "id": dataset_id,
            "message": "SQL dataset created successfully",
            "row_count": len(data),
            "column_count": len(columns),
            "file_size_mb": round(file_size_mb, 2)
        }

    except Exception as e:
        try:
            conn.close()
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")


@app.get("/datasets/sql", response_model=List[SqlDataset])
async def get_sql_datasets():
    """Get all SQL datasets"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT d.id, d.name, d.connection_id, d.sql_query, d.row_count, 
                   d.column_count, d.file_size_mb, d.created_at, d.updated_at,
                   d.generation_type, d.agent_prompt
            FROM sql_datasets d
            ORDER BY d.created_at DESC
        """)

        datasets = cursor.fetchall()
        conn.close()

        return [
            SqlDataset(
                id=ds[0],
                name=ds[1],
                connection_id=ds[2],
                sql_query=ds[3],
                row_count=ds[4],
                column_count=ds[5],
                file_size_mb=ds[6],
                created_at=ds[7],
                updated_at=ds[8],
                generation_type=ds[9] if ds[9] else 'manual',
                agent_prompt=ds[10]
            ) for ds in datasets
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get datasets: {str(e)}")


@app.get("/datasets/sql/{dataset_id}")
async def get_sql_dataset_data(dataset_id: str, limit: int = 100):
    """Get data from a SQL dataset"""
    try:
        # Read the CSV file
        csv_file_path = Path("datasets") / f"{dataset_id}.csv"

        if not csv_file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")

        try:
            df = pd.read_csv(csv_file_path)
        except pd.errors.EmptyDataError:
            # Handle empty CSV files gracefully
            return {
                "columns": [],
                "data": [],
                "total_rows": 0,
                "returned_rows": 0,
                "column_count": 0,
                "message": "Dataset is empty"
            }

        # Limit the results
        if limit > 0:
            df_limited = df.head(limit)
        else:
            df_limited = df

        # Convert to JSON, replacing NaN with None for JSON compatibility
        import numpy as np
        data = df_limited.replace({np.nan: None}).to_dict('records')

        return {
            "columns": list(df.columns),
            "data": data,
            "total_rows": len(df),
            "returned_rows": len(data),
            "column_count": len(df.columns)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset data: {str(e)}")


@app.get("/datasets/sql/{dataset_id}/download")
async def download_sql_dataset(dataset_id: str, format: str = "csv"):
    """Download SQL dataset in specified format"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sql_datasets WHERE id = ?", (dataset_id,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")

        dataset_name = result[0]
        csv_file_path = Path("datasets") / f"{dataset_id}.csv"

        if not csv_file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")

        if format.lower() == "csv":
            return FileResponse(
                path=str(csv_file_path),
                filename=f"{dataset_name}.csv",
                media_type="text/csv"
            )
        elif format.lower() == "excel":
            df = pd.read_csv(csv_file_path)
            excel_path = csv_file_path.with_suffix('.xlsx')
            df.to_excel(excel_path, index=False)

            return FileResponse(
                path=str(excel_path),
                filename=f"{dataset_name}.xlsx",
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            raise HTTPException(status_code=400, detail="Supported formats: csv, excel")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.delete("/datasets/sql/{dataset_id}")
async def delete_sql_dataset(dataset_id: str):
    """Delete a SQL dataset"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM sql_datasets WHERE id = ?", (dataset_id,))

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Dataset not found")

        conn.commit()
        conn.close()

        # Delete the CSV file
        csv_file_path = Path("datasets") / f"{dataset_id}.csv"
        if csv_file_path.exists():
            csv_file_path.unlink()

        # Delete Excel file if exists
        excel_file_path = Path("datasets") / f"{dataset_id}.xlsx"
        if excel_file_path.exists():
            excel_file_path.unlink()

        return {"message": "Dataset deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


@app.post("/jobs/{job_id}/use-sql-dataset")
async def use_sql_dataset_for_job(job_id: str, request: dict):
    """Use a SQL dataset for a job instead of uploading a file"""
    try:
        dataset_id = request.get("dataset_id")
        if not dataset_id:
            raise HTTPException(status_code=400, detail="Dataset ID is required")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Verify job exists
        cursor.execute("SELECT id FROM jobs WHERE id = ?", (job_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Job not found")

        # Get dataset info
        cursor.execute("SELECT name FROM sql_datasets WHERE id = ?", (dataset_id,))
        dataset_info = cursor.fetchone()
        if not dataset_info:
            raise HTTPException(status_code=404, detail="SQL dataset not found")

        dataset_name = dataset_info[0]

        # Copy the dataset CSV file to the uploads directory with the job_id as filename
        source_file = Path("datasets") / f"{dataset_id}.csv"
        target_file = Path("uploads") / f"{job_id}.csv"

        if not source_file.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")

        shutil.copy2(source_file, target_file)

        # Update job with dataset path
        cursor.execute("""
            UPDATE jobs 
            SET dataset_path = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        """, (str(target_file), job_id))

        conn.commit()
        conn.close()

        return {
            "message": f"SQL dataset '{dataset_name}' successfully applied to job",
            "dataset_path": str(target_file)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to use SQL dataset: {str(e)}")


# Dataset Editing Endpoints
@app.put("/datasets/sql/{dataset_id}/edit")
async def edit_sql_dataset(dataset_id: str, request: DatasetEditRequest):
    """Edit a SQL dataset by updating its data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Verify dataset exists
        cursor.execute("SELECT name FROM sql_datasets WHERE id = ?", (dataset_id,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Convert new data to DataFrame
        df = pd.DataFrame(request.data)

        # Save updated CSV file
        csv_file_path = Path("datasets") / f"{dataset_id}.csv"
        df.to_csv(csv_file_path, index=False)

        # Update metadata in database
        file_size_mb = csv_file_path.stat().st_size / (1024 * 1024)
        cursor.execute("""
            UPDATE sql_datasets 
            SET row_count = ?, column_count = ?, file_size_mb = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (len(df), len(df.columns), file_size_mb, dataset_id))

        conn.commit()
        conn.close()

        return {
            "message": "Dataset updated successfully",
            "row_count": len(df),
            "column_count": len(df.columns),
            "file_size_mb": round(file_size_mb, 2)
        }

    except Exception as e:
        try:
            conn.close()
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to edit dataset: {str(e)}")


@app.post("/datasets/sql/{dataset_id}/add-row")
async def add_row_to_dataset(dataset_id: str, request: DatasetAddRowRequest):
    """Add a row to a SQL dataset"""
    try:
        csv_file_path = Path("datasets") / f"{dataset_id}.csv"

        if not csv_file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")

        # Read current dataset
        df = pd.read_csv(csv_file_path)

        # Add new row
        new_row_df = pd.DataFrame([request.row_data])
        df = pd.concat([df, new_row_df], ignore_index=True)

        # Save updated file
        df.to_csv(csv_file_path, index=False)

        # Update metadata
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        file_size_mb = csv_file_path.stat().st_size / (1024 * 1024)
        cursor.execute("""
            UPDATE sql_datasets 
            SET row_count = ?, file_size_mb = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (len(df), file_size_mb, dataset_id))

        conn.commit()
        conn.close()

        return {
            "message": "Row added successfully",
            "new_row_count": len(df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add row: {str(e)}")


@app.delete("/datasets/sql/{dataset_id}/rows/{row_index}")
async def delete_row_from_dataset(dataset_id: str, row_index: int):
    """Delete a row from a SQL dataset"""
    try:
        csv_file_path = Path("datasets") / f"{dataset_id}.csv"

        if not csv_file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")

        # Read current dataset
        df = pd.read_csv(csv_file_path)

        # Validate row index
        if row_index < 0 or row_index >= len(df):
            raise HTTPException(status_code=400, detail="Invalid row index")

        # Delete row
        df = df.drop(index=row_index).reset_index(drop=True)

        # Save updated file
        df.to_csv(csv_file_path, index=False)

        # Update metadata
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        file_size_mb = csv_file_path.stat().st_size / (1024 * 1024)
        cursor.execute("""
            UPDATE sql_datasets 
            SET row_count = ?, file_size_mb = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (len(df), file_size_mb, dataset_id))

        conn.commit()
        conn.close()

        return {
            "message": "Row deleted successfully",
            "new_row_count": len(df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete row: {str(e)}")


@app.put("/datasets/sql/{dataset_id}/cell")
async def edit_dataset_cell(dataset_id: str, row_index: int, column: str, value: str = ""):
    """Edit a single cell in a dataset"""
    try:
        csv_file_path = Path("datasets") / f"{dataset_id}.csv"

        if not csv_file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")

        # Read current dataset
        df = pd.read_csv(csv_file_path)

        # Validate indices
        if row_index < 0 or row_index >= len(df):
            raise HTTPException(status_code=400, detail="Invalid row index")

        if column not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid column name")

        # Update cell
        # Try to convert to appropriate type
        if value.isdigit():
            df.loc[row_index, column] = int(value)
        elif value.replace('.', '').replace('-', '').isdigit():
            try:
                df.loc[row_index, column] = float(value)
            except ValueError:
                df.loc[row_index, column] = value
        else:
            df.loc[row_index, column] = value

        # Save updated file
        df.to_csv(csv_file_path, index=False)

        # Update file size metadata
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        file_size_mb = csv_file_path.stat().st_size / (1024 * 1024)
        cursor.execute("""
            UPDATE sql_datasets 
            SET file_size_mb = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (file_size_mb, dataset_id))

        conn.commit()
        conn.close()

        return {"message": "Cell updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to edit cell: {str(e)}")


@app.post("/datasets/sql/agent-generate", response_model=dict)
async def generate_sql_dataset_with_agent(request: SqlAgentDatasetRequest):
    """Generate a SQL dataset using the SQL Agent from natural language question"""

    if not SQL_AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="SQL Agent not available. Please install required dependencies.")

    try:
        # Prepare output directory
        output_dir = "datasets"
        os.makedirs(output_dir, exist_ok=True)

        # Get database connection info
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT db_type, host, port, database_name, username, password
            FROM database_connections WHERE id = ? AND is_active = 1
        """, (request.connection_id,))

        result_conn = cursor.fetchone()
        conn.close()

        if result_conn:
            db_type, host, port, database_name, username, encrypted_password = result_conn

            # Decrypt password
            password = decrypt_password(encrypted_password)

            # Construct the proper URI based on database type
            if db_type == 'postgresql':
                pg_uri = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
            elif db_type == 'mysql':
                pg_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_name}"
            elif db_type == 'sqlite':
                pg_uri = f"sqlite:///{database_name}"
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported database type: {db_type}")

            connection_info = {
                "pg_uri": pg_uri
            }
        else:
            connection_info = None  # Will use mock data

        # Generate dataset using the agent
        result = await sql_dataset_agent.generate_sql_dataset(
            question=request.question,
            connection_info=connection_info,
            output_dir=output_dir
        )

        if not result["success"]:
            error_detail = result.get("error", "Unknown error generating dataset")
            if result.get("warning"):
                error_detail += f". Warning: {result.get('warning')}"
            raise HTTPException(status_code=500, detail=error_detail)

        # Check if CSV was generated successfully
        if not result.get("csv_path") or not os.path.exists(result["csv_path"]):
            # If we have a SQL query but no data, provide more specific error
            if result.get("sql_query"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Dataset generation failed: SQL query was generated but no data was returned. "
                           f"Query: {result.get('sql_query')[:100]}..."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Dataset generation failed: No SQL query was generated from your question"
                )

        # Determine if this is create or update
        if request.dataset_id:
            # Updating existing dataset
            dataset_id = request.dataset_id
            new_csv_path = Path(output_dir) / f"{dataset_id}.csv"

            # Remove old CSV file if it exists
            if new_csv_path.exists():
                new_csv_path.unlink()

            # Move the generated CSV to the proper location
            original_path = result["csv_path"]
            if original_path and os.path.exists(original_path):
                shutil.move(original_path, new_csv_path)
            else:
                raise HTTPException(status_code=500, detail="Generated CSV file not found")

            # Update dataset metadata in database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE sql_datasets SET
                    name = ?, sql_query = ?, row_count = ?, column_count = ?, 
                    file_size_mb = ?, updated_at = ?, generation_type = ?, agent_prompt = ?
                WHERE id = ?
            """, (
                request.name,
                result["sql_query"],
                result["row_count"],
                result["column_count"],
                result["file_size_mb"],
                datetime.now().isoformat(),
                'agent',
                request.question,
                dataset_id
            ))
        else:
            # Creating new dataset
            dataset_id = str(uuid.uuid4())

            # Move the generated CSV to the proper location with dataset ID
            original_path = result["csv_path"]
            new_csv_path = Path(output_dir) / f"{dataset_id}.csv"
            if original_path and os.path.exists(original_path):
                shutil.move(original_path, new_csv_path)
            else:
                raise HTTPException(status_code=500, detail="Generated CSV file not found")

            # Save dataset metadata to database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # Use the provided connection_id
            connection_id = request.connection_id

            cursor.execute("""
                INSERT INTO sql_datasets (
                    id, name, connection_id, sql_query, 
                    row_count, column_count, file_size_mb, 
                    created_at, updated_at, generation_type, agent_prompt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                request.name,
                connection_id,
                result["sql_query"],
                result["row_count"],
                result["column_count"],
                result["file_size_mb"],
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                'agent',
                request.question
            ))

        conn.commit()
        conn.close()

        return {
            "id": dataset_id,
            "name": request.name,
            "sql_query": result["sql_query"],
            "csv_path": str(new_csv_path),
            "row_count": result["row_count"],
            "column_count": result["column_count"],
            "file_size_mb": result["file_size_mb"],
            "columns": result["columns"],
            "is_mock": result.get("is_mock", False),
            "generation_type": "agent",
            "agent_prompt": request.question,
            "connection_id": request.connection_id,
            "message": "Dataset generated successfully using SQL Agent" if not request.dataset_id else "Dataset updated successfully using SQL Agent"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate dataset: {str(e)}")


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_main():
    return HTMLResponse(open("static/index.html").read())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8006)
