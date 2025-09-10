# Tools Module

This module contains organized utilities for the ML Pipeline, separated into focused modules for better maintainability.

## Structure

```
tools/
├── __init__.py              # Main imports and exports
├── script_execution.py      # Script execution and Docker management
├── file_analysis.py         # CSV analysis and file operations
├── reporting.py             # Logging, reporting, and database operations
├── utils.py                 # Utility functions and helpers
└── README.md               # This documentation
```

## Modules

### script_execution.py

- **Purpose**: Handles script execution with automatic package installation
- **Key Functions**:
    - `execute_with_auto_install()`: Execute scripts with auto pip install
    - `create_script_execution_wrapper()`: Create pipeline-specific execution wrappers
    - `start_code_executor()`, `stop_code_executor()`: Docker container management

### file_analysis.py

- **Purpose**: File reading, CSV analysis, and data processing
- **Key Functions**:
    - `read_and_analyze_csv()`: Comprehensive CSV analysis with multiple encodings/separators
    - `get_file_sample()`: Read file samples for analysis
    - `analyze_dataframe()`: Extract metadata from pandas DataFrames
    - `create_file_analysis_wrappers()`: Create pipeline-specific file analysis tools

### reporting.py

- **Purpose**: Logging, reporting, and database operations
- **Key Classes & Functions**:
    - `MLPipelineLogger`: Comprehensive pipeline logging and reporting
    - `save_process_report()`, `save_generated_script()`: Database persistence
    - `update_job_status()`: Job status management
    - `log_message()`: General purpose logging

### utils.py

- **Purpose**: Common utilities and helper functions
- **Key Functions**:
    - `log_system_resources()`: System resource monitoring
    - `PipelineContext`: Context object for pipeline state

## Usage

Import tools from the main module:

```python
from tools import (
    MLPipelineLogger,
    create_script_execution_wrapper,
    create_file_analysis_wrappers,
    log_system_resources
)
```

Or import specific modules:

```python
from tools.script_execution import execute_with_auto_install
from tools.reporting import MLPipelineLogger
```

## Benefits of This Structure

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Maintainability**: Easier to modify and extend individual components
3. **Testability**: Each module can be tested independently
4. **Reusability**: Tools can be used across different parts of the system
5. **Clean Imports**: Pipeline.py is much cleaner and focused on orchestration

## Migration

The original `pipeline.py` had all these functions mixed together. They have been:

- ✅ Extracted to focused modules
- ✅ Organized by functionality
- ✅ Imported cleanly in the main pipeline
- ✅ Tested for compatibility