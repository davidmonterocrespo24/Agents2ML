"""
Tools module for the ML Pipeline.
Provides organized utilities for script execution, file analysis, and reporting.
"""

# File analysis tools
from .file_analysis import (
    detect_decimal_separator,
    try_read_csv_with_options,
    analyze_dataframe,
    read_and_analyze_csv,
    get_file_sample,
    check_generated_files,
    create_file_analysis_wrappers
)
# Reporting and logging tools
from .reporting import (
    log_message,
    save_process_report,
    save_generated_script,
    save_model,
    update_job_status,
    track_agent_call,
    get_agent_statistics_summary,
    MLPipelineLogger
)
# Script execution tools
from .script_execution import (
    execute_with_auto_install,
    create_script_execution_wrapper,
    start_code_executor,
    stop_code_executor,
    module_to_pip_name
)
# Utility tools
from .utils import (
    log_system_resources,
    PipelineContext
)

__all__ = [
    # Script execution
    'execute_with_auto_install',
    'create_script_execution_wrapper',
    'start_code_executor',
    'stop_code_executor',
    'module_to_pip_name',

    # File analysis
    'detect_decimal_separator',
    'try_read_csv_with_options',
    'analyze_dataframe',
    'read_and_analyze_csv',
    'get_file_sample',
    'check_generated_files',
    'create_file_analysis_wrappers',

    # Reporting and logging
    'log_message',
    'save_process_report',
    'save_generated_script',
    'save_model',
    'update_job_status',
    'track_agent_call',
    'get_agent_statistics_summary',
    'MLPipelineLogger',

    # Utilities
    'log_system_resources',
    'PipelineContext'
]
