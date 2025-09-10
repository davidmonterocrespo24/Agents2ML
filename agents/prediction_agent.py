"""
PredictionAgent definition and configuration
"""

from autogen_agentchat.agents import AssistantAgent

from agents.base_agent import create_model_client
from prompts.prediction_agent_prompt import PREDICTION_AGENT_PROMPT


def create_prediction_agent():
    """
    Create and return the PredictionAgent
    
    Returns:
        AssistantAgent: Configured PredictionAgent
    """
    model_client = create_model_client()

    return AssistantAgent(
        name="PredictionAgent",
        model_client=model_client,
        description="Expert that generates a Python script to make predictions.",
        system_message=PREDICTION_AGENT_PROMPT,
    )
