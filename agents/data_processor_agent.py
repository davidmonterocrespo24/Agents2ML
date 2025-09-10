"""
DataProcessorAgent definition and configuration
"""

from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool

from agents.base_agent import create_model_client
from prompts.data_processor_prompt import DATA_PROCESSOR_PROMPT


def create_data_processor_agent(get_sample_func, read_and_analyze_func):
    """
    Creates and returns the DataProcessorAgent with specialized tools.
    
    Args:
        get_sample_func: Function to get a text sample from the file.
        read_and_analyze_func: Function to read and analyze the CSV with specific parameters.
    
    Returns:
        AssistantAgent: The configured DataProcessorAgent.
    """
    model_client = create_model_client()

    return AssistantAgent(
        name="DataProcessorAgent",
        model_client=model_client,
        description="Data processing expert that analyzes CSV files in a two-step process.",
        tools=[
            FunctionTool(
                get_sample_func,
                description="Use this tool FIRST to get a small text sample from the file. This helps you inspect the format (column separator, decimal separator) before trying to read the complete file."
            ),
            FunctionTool(
                read_and_analyze_func,
                description="Use this tool AFTER having inspected the sample. Provide the correct column separator ('separator') and decimal separator ('decimal') that you identified in the sample to read and get a complete analysis of the file."
            )
        ],
        system_message=DATA_PROCESSOR_PROMPT,
    )
