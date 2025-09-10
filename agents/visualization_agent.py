"""
VisualizationAgent definition and configuration
"""

from autogen_agentchat.agents import AssistantAgent

from agents.base_agent import create_model_client
from prompts.visualization_agent_prompt import VISUALIZATION_AGENT_PROMPT


def create_visualization_agent():
    """
    Create and return the VisualizationAgent
    
    Returns:
        AssistantAgent: Configured VisualizationAgent
    """
    model_client = create_model_client()

    return AssistantAgent(
        name="VisualizationAgent",
        model_client=model_client,
        description="Specialist that creates scripts to plot results.",
        system_message=VISUALIZATION_AGENT_PROMPT,
    )
