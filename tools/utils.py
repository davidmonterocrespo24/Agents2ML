"""
Utility functions for the ML Pipeline.
Contains common utilities and helper functions.
"""

import psutil

from .reporting import log_message


def log_system_resources(job_id: str):
    """Log current system resource usage"""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')

        resource_info = (
            f"System Resources - "
            f"CPU: {cpu_percent}%, "
            f"Memory: {memory.percent}% ({memory.used // (1024 ** 3):.1f}GB/{memory.total // (1024 ** 3):.1f}GB), "
            f"Disk: {disk.percent}% ({disk.free // (1024 ** 3):.1f}GB free)"
        )

        log_message(job_id, resource_info, "DEBUG")
    except Exception as e:
        log_message(job_id, f"Error getting system resources: {str(e)}", "WARNING")


class PipelineContext:
    """A simple object to share state."""

    def __init__(self):
        self.model_path_from_execution = None
