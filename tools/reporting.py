"""
Reporting and logging tools for the ML Pipeline.
Handles process reports, script capture, and comprehensive logging.
"""

import json
import os
import sqlite3
import time
import uuid
from datetime import datetime

from config import Config

DB_PATH = Config.DATABASE_URL


def log_message(job_id: str, message: str, level: str = "INFO"):
    """Log a message to the database for a specific job"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add timestamp to message for better tracking
    formatted_message = f"[{timestamp}] {message}"

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO logs (job_id, message, level)
            VALUES (?, ?, ?)
        """, (job_id, formatted_message, level))

        conn.commit()

        # Also log to console for development
        print(f"[{level}] {job_id}: {message}")

    except Exception as e:
        print(f"Error logging message: {e}")
    finally:
        try:
            conn.close()
        except:
            pass


def save_process_report(job_id: str, stage: str, title: str, content: str, metadata: dict = None):
    """Save a process report to the database"""
    report_id = str(uuid.uuid4())

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO process_reports (id, job_id, stage, title, content, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (report_id, job_id, stage, title, content, json.dumps(metadata) if metadata else None))

        conn.commit()
        log_message(job_id, f"Process report saved: {title}", "INFO")

    except Exception as e:
        log_message(job_id, f"Error saving process report: {str(e)}", "ERROR")
    finally:
        try:
            conn.close()
        except:
            pass

    return report_id


def save_generated_script(job_id: str, script_name: str, script_type: str, script_content: str, agent_name: str,
                          execution_result: str = None):
    """Save a generated script to the database"""
    script_id = str(uuid.uuid4())

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO generated_scripts (id, job_id, script_name, script_type, script_content, agent_name, execution_result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (script_id, job_id, script_name, script_type, script_content, agent_name, execution_result))

        conn.commit()
        log_message(job_id, f"Script saved: {script_name} by {agent_name}", "INFO")

    except Exception as e:
        log_message(job_id, f"Error saving generated script: {str(e)}", "ERROR")
    finally:
        try:
            conn.close()
        except:
            pass

    return script_id


def save_model(job_id: str, name: str, model_path: str, metrics: dict = None):
    """Save model information to the database"""
    model_id = str(uuid.uuid4())

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Log model save attempt
        log_message(job_id, f"Saving model '{name}' to database with path: {model_path}", "INFO")

        # Get file size if path exists
        file_size = None
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            log_message(job_id, f"Model file size: {file_size / (1024 * 1024):.2f} MB", "DEBUG")
        else:
            log_message(job_id, f"Warning: Model path does not exist: {model_path}", "WARNING")

        # Add file size to metrics
        if metrics is None:
            metrics = {}
        if file_size:
            metrics['file_size_bytes'] = file_size
            metrics['file_size_mb'] = round(file_size / (1024 * 1024), 2)

        cursor.execute("""
            INSERT INTO models (id, job_id, name, model_path, metrics)
            VALUES (?, ?, ?, ?, ?)
        """, (model_id, job_id, name, model_path, json.dumps(metrics) if metrics else None))

        conn.commit()
        log_message(job_id, f"Model saved successfully with ID: {model_id}", "INFO")

    except Exception as e:
        log_message(job_id, f"Error saving model to database: {str(e)}", "ERROR")
    finally:
        try:
            conn.close()
        except:
            pass

    return model_id


def track_agent_call(job_id: str, agent_name: str, tokens_used: int = 0, input_tokens: int = 0, output_tokens: int = 0,
                     execution_time: float = 0):
    """Track agent call statistics including token usage"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Insert or update agent statistics
        cursor.execute("""
            INSERT INTO agent_statistics 
            (job_id, agent_name, tokens_consumed, calls_count, input_tokens, output_tokens, total_execution_time, last_updated)
            VALUES (?, ?, ?, 1, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(job_id, agent_name) DO UPDATE SET
            tokens_consumed = tokens_consumed + EXCLUDED.tokens_consumed,
            calls_count = calls_count + 1,
            input_tokens = input_tokens + EXCLUDED.input_tokens,
            output_tokens = output_tokens + EXCLUDED.output_tokens,
            total_execution_time = total_execution_time + EXCLUDED.total_execution_time,
            last_updated = CURRENT_TIMESTAMP
        """, (job_id, agent_name, tokens_used, input_tokens, output_tokens, execution_time))

        conn.commit()

        # Log the tracking
        log_message(job_id,
                    f"Tracked call for {agent_name}: {tokens_used} tokens, {execution_time:.2f}s execution time",
                    "DEBUG")

    except Exception as e:
        log_message(job_id, f"Error tracking agent call: {str(e)}", "ERROR")
    finally:
        try:
            conn.close()
        except:
            pass


def get_agent_statistics_summary(job_id: str) -> dict:
    """Get a summary of agent statistics for a job"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                COUNT(DISTINCT agent_name) as unique_agents,
                SUM(calls_count) as total_calls,
                SUM(tokens_consumed) as total_tokens,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(total_execution_time) as total_execution_time,
                MAX(last_updated) as last_activity
            FROM agent_statistics WHERE job_id = ?
        """, (job_id,))

        result = cursor.fetchone()
        conn.close()

        if result and result[0] is not None:
            return {
                'unique_agents': result[0],
                'total_calls': result[1] or 0,
                'total_tokens': result[2] or 0,
                'total_input_tokens': result[3] or 0,
                'total_output_tokens': result[4] or 0,
                'total_execution_time': result[5] or 0.0,
                'last_activity': result[6]
            }
        else:
            return {
                'unique_agents': 0,
                'total_calls': 0,
                'total_tokens': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_execution_time': 0.0,
                'last_activity': None
            }

    except Exception as e:
        log_message(job_id, f"Error getting agent statistics summary: {str(e)}", "ERROR")
        return {
            'unique_agents': 0,
            'total_calls': 0,
            'total_tokens': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_execution_time': 0.0,
            'last_activity': None
        }


def update_job_status(job_id: str, status: str, progress: int = None, error_message: str = None):
    """Update job status in the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Log status change
        status_message = f"Status changed to: {status}"
        if progress is not None:
            status_message += f" (Progress: {progress}%)"
        if error_message:
            status_message += f" - Error: {error_message}"

        log_message(job_id, status_message, "INFO" if not error_message else "ERROR")

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

    except Exception as e:
        print(f"Error updating job status: {e}")
    finally:
        try:
            conn.close()
        except:
            pass


class MLPipelineLogger:
    """Custom logger that captures agent outputs and saves to database"""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.start_time = time.time()
        self.step_times = {}
        self.user_input_requested = False
        self.process_data = {
            'dataset_info': {},
            'preprocessing_info': {},
            'training_info': {},
            'model_info': {},
            'predictions_info': {},
            'scripts_generated': []
        }

    def log_agent_message(self, agent_name: str, message: str, level: str = "INFO", tokens_used: int = 0,
                          input_tokens: int = 0, output_tokens: int = 0, execution_time: float = 0):
        # Add timing information
        elapsed = time.time() - self.start_time
        formatted_message = f"[{agent_name}] [T+{elapsed:.1f}s] {message}"
        log_message(self.job_id, formatted_message, level)

        # Track agent call if tokens were used (indicating actual agent activity)
        if tokens_used > 0 or input_tokens > 0 or output_tokens > 0:
            track_agent_call(self.job_id, agent_name, tokens_used, input_tokens, output_tokens, execution_time)

        # Check if user input is requested
        self.check_for_user_input_request(message, agent_name)

        # Also save to agent_messages table for chat interface
        self.save_agent_message(agent_name, message)

    def save_agent_message(self, agent_name: str, content: str):
        """Save agent message to chat interface table"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO agent_messages (job_id, agent_name, content, message_type, source)
                VALUES (?, ?, ?, ?, ?)
            """, (self.job_id, agent_name, content, "agent", "agent"))

            conn.commit()

        except Exception as e:
            print(f"Error saving agent message: {e}")
        finally:
            try:
                conn.close()
            except:
                pass

    def log_step_start(self, step_name: str):
        """Log the start of a pipeline step"""
        self.step_times[step_name] = time.time()
        self.log_agent_message("Pipeline", f"Starting step: {step_name}", "INFO")

    def log_step_end(self, step_name: str):
        """Log the end of a pipeline step"""
        if step_name in self.step_times:
            duration = time.time() - self.step_times[step_name]
            self.log_agent_message("Pipeline", f"Completed step: {step_name} in {duration:.2f}s", "INFO")
        else:
            self.log_agent_message("Pipeline", f"Completed step: {step_name} (no start time recorded)", "WARNING")

    def log_file_operation(self, operation: str, file_path: str, success: bool = True):
        """Log file operations"""
        status = "SUCCESS" if success else "FAILED"
        file_size = ""
        if success and os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)
            file_size = f" ({size_bytes / (1024 * 1024):.2f} MB)"

        message = f"File operation {operation}: {file_path}{file_size} - {status}"
        self.log_agent_message("FileOps", message, "INFO" if success else "ERROR")

    def check_for_user_input_request(self, message: str, agent_name: str):
        """Check if an agent is requesting user input and mark input area as visible"""
        user_input_indicators = [
            "Enter your response:",
            "Please provide",
            "I need your input",
            "Could you help me",
            "What would you like",
            "Please specify",
            "How would you like to proceed",
            "Do you want me to",
            "Should I proceed",
            "Would you like me to",
            "USER_INPUT_REQUIRED",
            "WAITING_FOR_USER",
            "Please clarify"
        ]

        # Check if any user input indicator is present in the message
        if any(indicator.lower() in message.lower() for indicator in user_input_indicators):
            self.user_input_requested = True
            self.log_agent_message("System", f"🔔 User input requested by {agent_name}", "WARNING")

            # Update job status to indicate user input is needed
            update_job_status(self.job_id, "awaiting_user_input", error_message="Agent is waiting for user input")

            # Save a special message to highlight that input is needed
            self.save_user_input_request_message(agent_name, message)

    def save_user_input_request_message(self, agent_name: str, original_message: str):
        """Save a special highlighted message when user input is requested"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            highlighted_message = f"🔔 **USER INPUT REQUESTED** 🔔\n\n{original_message}\n\n**Please respond using the chat input below.**"

            cursor.execute("""
                INSERT INTO agent_messages (job_id, agent_name, content, message_type, source)
                VALUES (?, ?, ?, ?, ?)
            """, (self.job_id, f"{agent_name} (Input Request)", highlighted_message, "user_input_request", "system"))

            conn.commit()

        except Exception as e:
            print(f"Error saving user input request message: {e}")
        finally:
            try:
                conn.close()
            except:
                pass

    def log_dataset_info(self, dataset_path: str, rows: int, columns: int, target_column: str, file_size_mb: float):
        """Log dataset information"""
        self.process_data['dataset_info'] = {
            'path': dataset_path,
            'rows': rows,
            'columns': columns,
            'target_column': target_column,
            'file_size_mb': file_size_mb,
            'timestamp': time.time() - self.start_time
        }

        content = f"""
## Dataset Analysis

**File**: {dataset_path}
**Rows**: {rows:,}
**Columns**: {columns}
**Target Column**: {target_column}
**File Size**: {file_size_mb:.2f} MB

Dataset loaded and analyzed successfully at T+{self.process_data['dataset_info']['timestamp']:.1f}s
        """

        save_process_report(self.job_id, "dataset", "Dataset Information", content.strip(),
                            self.process_data['dataset_info'])

    def log_script_generated(self, script_name: str, script_type: str, script_content: str, agent_name: str,
                             execution_result: str = None):
        """Log a generated script"""
        script_info = {
            'name': script_name,
            'type': script_type,
            'agent': agent_name,
            'size': len(script_content),
            'timestamp': time.time() - self.start_time
        }

        self.process_data['scripts_generated'].append(script_info)

        # Save to database
        save_generated_script(self.job_id, script_name, script_type, script_content, agent_name, execution_result)

        self.log_agent_message("ScriptCapture",
                               f"Script captured: {script_name} ({len(script_content)} chars) by {agent_name}")

    def generate_final_report(self):
        """Generate a comprehensive final report"""
        total_time = time.time() - self.start_time

        report_sections = []

        # Summary section
        summary = f"""
# ML Pipeline Execution Report

**Job ID**: {self.job_id}
**Total Execution Time**: {total_time:.2f} seconds
**Completed**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---
        """
        report_sections.append(summary.strip())

        # Dataset section
        if self.process_data['dataset_info']:
            dataset_info = self.process_data['dataset_info']
            dataset_section = f"""
## Dataset Summary
- **File**: {dataset_info.get('path', 'N/A')}
- **Dimensions**: {dataset_info.get('rows', 0):,} rows × {dataset_info.get('columns', 0)} columns
- **Target**: {dataset_info.get('target_column', 'N/A')}
- **Size**: {dataset_info.get('file_size_mb', 0):.2f} MB
            """
            report_sections.append(dataset_section.strip())

        # Scripts section
        if self.process_data['scripts_generated']:
            scripts_section = "## Generated Scripts\n"
            for script in self.process_data['scripts_generated']:
                scripts_section += f"- **{script['name']}** ({script['type']}) - {script['size']:,} chars by {script['agent']}\n"
            report_sections.append(scripts_section.strip())

        final_content = "\n\n".join(report_sections)

        # Save final report
        save_process_report(self.job_id, "summary", "Complete Pipeline Report", final_content, {
            'total_time': total_time,
            'scripts_count': len(self.process_data['scripts_generated']),
            'completion_time': datetime.now().isoformat()
        })
