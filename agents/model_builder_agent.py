"""
ModelBuilderAgent definition and configuration
"""

from autogen_agentchat.agents import AssistantAgent

from agents.base_agent import create_model_client
from prompts.model_builder_prompt import MODEL_BUILDER_PROMPT


def create_model_builder_agent():
    """
    Create and return the ModelBuilderAgent
    
    Returns:
        AssistantAgent: Configured ModelBuilderAgent
    """
    model_client = create_model_client()

    return AssistantAgent(
        name="ModelBuilderAgent",
        model_client=model_client,
        description="Machine Learning expert that generates Python scripts for H2O AutoML",
        system_message=MODEL_BUILDER_PROMPT,
    )
