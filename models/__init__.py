"""
Pydantic models for the AutoML Training System
"""

from .agent_models import (
    LogEntry,
    AgentMessage,
    UserMessageInput,
    ProcessReport,
    GeneratedScript,
    AgentStatistics
)
from .database_models import (
    DatabaseConnectionCreate,
    DatabaseConnectionUpdate,
    DatabaseConnection,
    SqlDatasetCreate,
    SqlDataset,
    QueryExecuteRequest,
    SqlAgentDatasetRequest,
    DatasetEditRequest,
    DatasetAddRowRequest,
    DatasetDeleteRowRequest
)
from .job_models import (
    JobCreate,
    JobVersionCreate,
    Job,
    Model
)

__all__ = [
    # Job models
    "JobCreate",
    "JobVersionCreate",
    "Job",
    "Model",

    # Agent models
    "LogEntry",
    "AgentMessage",
    "UserMessageInput",
    "ProcessReport",
    "GeneratedScript",
    "AgentStatistics",

    # Database models
    "DatabaseConnectionCreate",
    "DatabaseConnectionUpdate",
    "DatabaseConnection",
    "SqlDatasetCreate",
    "SqlDataset",
    "QueryExecuteRequest",
    "SqlAgentDatasetRequest",
    "DatasetEditRequest",
    "DatasetAddRowRequest",
    "DatasetDeleteRowRequest"
]
