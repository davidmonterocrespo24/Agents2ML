"""
AnalystAgent definition and configuration
"""

from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool

from agents.base_agent import create_model_client
from prompts.analyst_prompt import ANALYST_PROMPT


def create_analyst_agent(set_model_path_tool, check_files_tool):
    """
    Create and return the AnalystAgent
    
    Args:
        set_model_path_tool: FunctionTool for setting model path
        check_files_tool: FunctionTool for checking generated files
    
    Returns:
        AssistantAgent: Configured AnalystAgent
    """
    model_client = create_model_client()

    return AssistantAgent(
        name="AnalystAgent",
        model_client=model_client,
        tools=[set_model_path_tool, check_files_tool],
        description="Data analyst that reviews and improves ML models",
        system_message=ANALYST_PROMPT,
    )
