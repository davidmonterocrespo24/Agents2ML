"""
Job-related Pydantic models for the AutoML Training System
"""

from pydantic import BaseModel
from typing import Optional


class JobCreate(BaseModel):
    name: str
    prompt: str
    target_column: Optional[str] = None


class JobVersionCreate(BaseModel):
    parent_job_id: str
    name: str
    prompt: str


class Job(BaseModel):
    id: str
    name: str
    prompt: str
    dataset_path: str
    status: str
    created_at: str
    updated_at: str
    progress: int
    error_message: Optional[str] = None
    target_column: str
    parent_job_id: Optional[str] = None
    version_number: int = 1
    is_parent: bool = True


class Model(BaseModel):
    id: str
    job_id: str
    name: str
    model_path: str
    metrics: Optional[dict] = None
    created_at: str
