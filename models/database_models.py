"""
Database and SQL dataset-related Pydantic models for the AutoML Training System
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class DatabaseConnectionCreate(BaseModel):
    name: str
    db_type: str
    host: str
    port: int
    database_name: str
    username: str
    password: str


class DatabaseConnectionUpdate(BaseModel):
    name: str
    db_type: str
    host: str
    port: int
    database_name: str
    username: str
    password: Optional[str] = None  # Optional to allow keeping existing password


class DatabaseConnection(BaseModel):
    id: str
    name: str
    db_type: str
    host: str
    port: int
    database_name: str
    username: str
    is_active: bool
    created_at: str
    updated_at: str


class SqlDatasetCreate(BaseModel):
    name: str
    connection_id: str
    sql_query: str


class SqlDataset(BaseModel):
    id: str
    name: str
    connection_id: str
    sql_query: str
    row_count: int
    column_count: int
    file_size_mb: float
    created_at: str
    updated_at: str
    generation_type: Optional[str] = 'manual'
    agent_prompt: Optional[str] = None


class QueryExecuteRequest(BaseModel):
    connection_id: str
    sql_query: str
    limit: Optional[int] = 100


class SqlAgentDatasetRequest(BaseModel):
    name: str
    question: str
    connection_id: str
    dataset_id: Optional[str] = None  # For editing existing datasets


class DatasetEditRequest(BaseModel):
    data: List[Dict[str, Any]]


class DatasetAddRowRequest(BaseModel):
    row_data: Dict[str, Any]


class DatasetDeleteRowRequest(BaseModel):
    row_index: int
