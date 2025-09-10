"""
Prompts package - Contains all system prompts for agents
"""

from .data_processor_prompt import DATA_PROCESSOR_PROMPT
from .model_builder_prompt import MODEL_BUILDER_PROMPT
from .code_executor_prompt import CODE_EXECUTOR_PROMPT
from .analyst_prompt import ANALYST_PROMPT
from .prediction_agent_prompt import PREDICTION_AGENT_PROMPT
from .visualization_agent_prompt import VISUALIZATION_AGENT_PROMPT

__all__ = [
    'DATA_PROCESSOR_PROMPT',
    'MODEL_BUILDER_PROMPT',
    'CODE_EXECUTOR_PROMPT',
    'ANALYST_PROMPT',
    'PREDICTION_AGENT_PROMPT',
    'VISUALIZATION_AGENT_PROMPT',
]