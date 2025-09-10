"""
Agent-related Pydantic models for the AutoML Training System
"""

from pydantic import BaseModel
from typing import Optional


class LogEntry(BaseModel):
    id: int
    job_id: str
    message: str
    level: str
    timestamp: str


class AgentMessage(BaseModel):
    id: int
    job_id: str
    agent_name: str
    content: str
    message_type: str
    timestamp: str
    source: str


class UserMessageInput(BaseModel):
    message: str


class ProcessReport(BaseModel):
    id: str
    job_id: str
    stage: str
    title: str
    content: str
    metadata: Optional[dict] = None
    created_at: str


class GeneratedScript(BaseModel):
    id: str
    job_id: str
    script_name: str
    script_type: str
    script_content: str
    agent_name: str
    execution_result: Optional[str] = None
    created_at: str


class AgentStatistics(BaseModel):
    id: int
    job_id: str
    agent_name: str
    tokens_consumed: int
    calls_count: int
    input_tokens: int
    output_tokens: int
    total_execution_time: float
    last_updated: str
