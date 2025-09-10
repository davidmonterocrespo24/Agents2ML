"""
CodeExecutorAgent definition and configuration
"""

from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool

from agents.base_agent import create_model_client
from prompts.code_executor_prompt import CODE_EXECUTOR_PROMPT


def create_code_executor_agent(execute_script_func):
    """
    Create and return the CodeExecutorAgent
    
    Args:
        execute_script_func: Function that executes scripts in pipeline context
    
    Returns:
        AssistantAgent: Configured CodeExecutorAgent
    """
    model_client = create_model_client()

    return AssistantAgent(
        name="CodeExecutorAgent",
        model_client=model_client,
        tools=[FunctionTool(
            execute_script_func,
            description="Execute script and auto-install missing packages.",
        )],
        system_message=CODE_EXECUTOR_PROMPT,
    )
