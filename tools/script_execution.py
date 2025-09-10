"""
Script execution tools for the ML Pipeline.
Handles automatic package installation and script execution in Docker containers.
"""

import asyncio
import re
import time
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import Config

# Configuration
PIP_NAME_MAP = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "cv2": "opencv-python",
}

# Initialize code executor
code_executor = DockerCommandLineCodeExecutor(
    work_dir=Config.CODING_DIR,
    image=Config.DOCKER_IMAGE,
    timeout=Config.CODE_EXECUTOR_TIMEOUT
)


def check_file_persistence(pipeline_name: str) -> bool:
    """Check if files are properly persisting from container to host"""
    try:
        base_work_dir = Path(Config.CODING_DIR)
        pipeline_dir = base_work_dir / pipeline_name if pipeline_name else base_work_dir
        return pipeline_dir.exists()
    except Exception as e:
        print(f"[PERSISTENCE] Error checking persistence: {e}")
        return False


def module_to_pip_name(module: str) -> str:
    """Map module name to pip package name"""
    return PIP_NAME_MAP.get(module, module)


async def execute_with_auto_install(
        script: str, filename: str = "script.py", args: str = "", max_retries: int = 3, pipeline_name: str = None
) -> str:
    """Execute script with automatic package installation"""
    start_time = time.time()

    # Create pipeline-specific subdirectory
    base_work_dir = Path(Config.CODING_DIR)
    if pipeline_name:
        work_dir = base_work_dir / pipeline_name
    else:
        work_dir = base_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: Save the script in the pipeline directory on the HOST
    # This ensures that the script persists and is available after execution
    script_path = work_dir / filename
    script_path.write_text(script, encoding="utf-8")
    print(f"[EXEC] Script saved to host directory: {script_path}")

    # NEW: Modify the script to ensure output files are saved in the correct location
    # Add persistence instructions at the beginning of the script
    if "predictions.csv" in script or "forecast_plot.png" in script or ".png" in script or ".csv" in script:
        persistence_header = f'''import shutil
import os
from pathlib import Path

# Ensure the output directory exists
output_dir = Path("/workspace/{pipeline_name}" if "{pipeline_name}" else "/workspace")
output_dir.mkdir(parents=True, exist_ok=True)

# Function to copy files to the mounted directory
def ensure_file_persistence(filename):
    """Copy file to the host mounted directory"""
    if os.path.exists(filename):
        dest_path = output_dir / filename
        shutil.copy2(filename, dest_path)
        print(f"File persisted to: {{dest_path}}")
        return str(dest_path)
    else:
        print(f"Warning: File {{filename}} not found for persistence")
    return None

# Hook for matplotlib to ensure plots are saved
import matplotlib
import matplotlib.pyplot as plt
original_savefig = plt.savefig

def persistent_savefig(filename, *args, **kwargs):
    """Wrapper for plt.savefig that ensures persistence"""
    result = original_savefig(filename, *args, **kwargs)
    ensure_file_persistence(filename)
    return result

plt.savefig = persistent_savefig

'''
        script = persistence_header + script

        # Add persistence calls at the end of the script for specific files
        persistence_footer = '''
# List current files for debugging
print("\\n=== DEBUG: Files in current directory ===")
import os
for item in os.listdir("."):
    if os.path.isfile(item):
        size = os.path.getsize(item)
        print(f"File: {item} ({size} bytes)")

# Ensure persistence of common output files
import glob
persisted_files = []
for file_pattern in ['*.csv', '*.png', '*.jpg', '*.jpeg', '*.pdf', '*.html']:
    for filename in glob.glob(file_pattern):
        result = ensure_file_persistence(filename)
        if result:
            persisted_files.append(filename)

if persisted_files:
    print(f"\\n=== Files persisted successfully: {', '.join(persisted_files)} ===")
else:
    print("\\n=== WARNING: No output files found to persist ===")
'''
        script += persistence_footer

    print(f"[EXEC] Starting execution of {filename} with args: {args}")
    print(f"[EXEC] Script size: {len(script)} characters")

    attempt = 0
    last_output = ""

    while attempt < max_retries:
        attempt += 1
        attempt_start = time.time()
        print(f"[EXEC] Attempt {attempt}/{max_retries}")

        cancel = CancellationToken()
        corrected_args = args
        if pipeline_name:
            # The correct path INSIDE the Docker container
            correct_docker_path = f"/workspace/{pipeline_name}"

            # Use a regular expression to find and replace the --pipeline-dir value
            # This corrects any incorrect value like '/pipelines/abc' that the agent might generate.
            pattern = r"(--pipeline-dir\s+)\S+"
            if re.search(pattern, corrected_args):
                corrected_args = re.sub(pattern, rf"\g<1>{correct_docker_path}", corrected_args)
                print(f"[EXEC] Corrected --pipeline-dir argument to: {correct_docker_path}")

            # Keep the 'cd' command to ensure the shell's working directory is correct
            safe_pipeline_name = pipeline_name.replace("'", "'\\''")
            command = f"cd '{safe_pipeline_name}' && python {filename} {corrected_args}"
        else:
            command = f"python {filename} {args}"

        run_block = CodeBlock(code=command, language="bash")

        try:
            print(f"[EXEC] Executing command: {command}")
            run_result = await code_executor.execute_code_blocks([run_block], cancel)
        except Exception as e:
            error_msg = f"Error executing in Docker (execute_code_blocks): {e}"
            print(f"[EXEC] Docker execution error: {e}")
            return error_msg

        exit_code = getattr(run_result, "exit_code", None)
        stdout = (
                getattr(run_result, "output", None)
                or getattr(run_result, "stdout", "")
                or ""
        )
        stderr = getattr(run_result, "stderr", "") or ""
        combined = (stdout or "") + "\n" + (stderr or "")

        attempt_time = time.time() - attempt_start
        print(f"[EXEC] Attempt {attempt} completed in {attempt_time:.2f}s, exit code: {exit_code}")

        if exit_code == 0 or (exit_code is None and "Traceback" not in combined):
            total_time = time.time() - start_time

            # Verify persistence of specific files after success
            if pipeline_name:
                host_pipeline_dir = Path(Config.CODING_DIR) / pipeline_name
                print(f"[EXEC] Checking file persistence in: {host_pipeline_dir}")

                # Verify specific files
                expected_files = []
                if "forecast_plot.png" in script or ".png" in script:
                    expected_files.append("forecast_plot.png")
                if "predictions.csv" in script or ".csv" in script:
                    expected_files.append("predictions.csv")

                persistence_status = []
                for expected_file in expected_files:
                    file_path = host_pipeline_dir / expected_file
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        persistence_status.append(f"✓ {expected_file} ({file_size} bytes)")
                        print(f"[EXEC] File persisted successfully: {file_path} ({file_size} bytes)")
                    else:
                        persistence_status.append(f"✗ {expected_file} (missing)")
                        print(f"[EXEC] WARNING: Expected file not found: {file_path}")

                if persistence_status:
                    persistence_report = "\n[File Persistence Status]\n" + "\n".join(persistence_status)
                    success_msg = f"=== Execution successful (attempt {attempt}/{max_retries}) in {total_time:.2f}s ===\n{stdout}\n{persistence_report}"
                else:
                    success_msg = f"=== Execution successful (attempt {attempt}/{max_retries}) in {total_time:.2f}s ===\n{stdout}"
            else:
                success_msg = f"=== Execution successful (attempt {attempt}/{max_retries}) in {total_time:.2f}s ===\n{stdout}"

            print(f"[EXEC] SUCCESS: {filename} executed successfully")
            return success_msg

        last_output = combined
        print(f"[EXEC] Attempt {attempt} failed, checking for missing modules...")

        # Detect missing module
        m = re.search(r"No module named ['\"]?([^'\"\n]+)['\"]?", combined)
        if not m:
            print(f"[EXEC] Error is not a missing module issue")
            return f"Execution error (not missing module). Attempts: {attempt}\n{combined}"

        missing_module = m.group(1).strip()
        pip_name = module_to_pip_name(missing_module)
        print(f"[EXEC] Missing module detected: {missing_module} -> installing {pip_name}")

        # Install with pip
        install_start = time.time()
        install_block = CodeBlock(
            code=f"pip install {pip_name} --no-input", language="bash"
        )
        try:
            install_result = await code_executor.execute_code_blocks(
                [install_block], cancel
            )
        except Exception as e:
            error_msg = f"Error installing {pip_name}: {e}"
            print(f"[EXEC] Installation error: {e}")
            return error_msg

        install_time = time.time() - install_start
        install_out = (
                getattr(install_result, "output", "")
                or getattr(install_result, "stdout", "")
                or ""
        )
        install_err = getattr(install_result, "stderr", "") or ""
        install_combined = install_out + "\n" + install_err

        print(f"[EXEC] Package installation completed in {install_time:.2f}s")

        if getattr(install_result, "exit_code", None) not in (0, None):
            print(f"[EXEC] Installation failed for {pip_name}")
            if attempt >= max_retries:
                return f"Could not install '{pip_name}'. Output:\n{install_combined}"
        else:
            print(f"[EXEC] Successfully installed {pip_name}")

    total_time = time.time() - start_time
    print(f"[EXEC] FAILED: Maximum retries reached after {total_time:.2f}s")
    return f"Maximum retries reached ({max_retries}). Last error:\n{last_output}"


def create_script_execution_wrapper(pipeline_name: str, logger):
    """Create a script execution wrapper with pipeline context and logging"""

    def execute_script_in_pipeline(script: str, filename: str = "script.py", args: str = "",
                                   max_retries: int = 3) -> str:
        """
        Synchronous wrapper that runs the async executor in a separate thread.
        This avoids calling asyncio.run() inside an already-running loop.
        Returns the stdout/stderr summary string.
        """

        def runner() -> str:
            # Capture the script before execution
            script_type = "unknown"
            if "train" in filename.lower() or "model" in filename.lower():
                script_type = "training"
            elif "predict" in filename.lower():
                script_type = "prediction"
            elif "visual" in filename.lower() or "plot" in filename.lower():
                script_type = "visualization"
            elif "preprocess" in filename.lower() or "clean" in filename.lower():
                script_type = "preprocessing"
            elif "analys" in filename.lower():
                script_type = "analysis"

            # NEW: Save script in the pipeline directory with metadata before executing
            base_work_dir = Path(Config.CODING_DIR)
            work_dir = base_work_dir / pipeline_name
            work_dir.mkdir(parents=True, exist_ok=True)

            # Save the script with timestamp and type
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            script_with_metadata = f"""# {script_type.upper()} SCRIPT
# Generated on: {datetime.now().isoformat()}
# Pipeline: {pipeline_name}
# Filename: {filename}
# Arguments: {args}
# Script Type: {script_type}

{script}
"""
            # Save with descriptive name
            if script_type != "unknown":
                persistent_filename = f"{timestamp}_{script_type}_{filename}"
            else:
                persistent_filename = f"{timestamp}_{filename}"

            persistent_script_path = work_dir / persistent_filename
            persistent_script_path.write_text(script_with_metadata, encoding="utf-8")
            print(f"[WRAPPER] Script saved persistently to: {persistent_script_path}")

            # Capture the script in the logger
            logger.log_script_generated(filename, script_type, script, "CodeExecutorAgent")

            # run the coroutine in an isolated event loop in a worker thread
            result = asyncio.run(execute_with_auto_install(script, filename, args, max_retries, pipeline_name))

            # Update the script with execution result
            logger.log_script_generated(filename, script_type, script, "CodeExecutorAgent",
                                        result[:1000] if result else None)

            # NEW: Save execution result alongside the script
            result_filename = f"{timestamp}_{script_type}_{filename.replace('.py', '_result.txt')}"
            result_path = work_dir / result_filename
            result_path.write_text(f"Execution Result for {filename}\n{'-' * 50}\n{result}", encoding="utf-8")
            print(f"[WRAPPER] Execution result saved to: {result_path}")

            return result

        # Run the coroutine in a separate thread to avoid interfering with the main event loop.
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(runner)
            return future.result()

    return execute_script_in_pipeline


async def start_code_executor():
    """Start the code executor"""
    await code_executor.start()


async def stop_code_executor():
    """Stop the code executor"""
    await code_executor.stop()
