"""
Base agent configuration and utilities
"""

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import Config


def create_model_client():
    """Create and return the OpenAI model client"""

    return OpenAIChatCompletionClient(
        model="gpt-oss:120b",
        model_info={
            "family": "ollama",
            "vision": False,
            "function_calling": True,
            "json_output": True,
        },
        base_url="http://127.0.0.1:11434/v1",
    )

# Uncomment the following lines to use the Hugging Face API

    # return OpenAIChatCompletionClient(
    #    model="openai/gpt-oss-120b:cerebras",
    #    api_key=Config.HF_TOKEN,
    #    model_info={
    #        "vision": False,
    #        "function_calling": True,
    #       "json_output": True,
    #       "family": "openai",


#     },
#    base_url="https://router.huggingface.co/v1",
# )


def create_user_proxy_agent():
    """Create and return the UserProxy agent"""
    return UserProxyAgent(
        name="Admin",
        description="An administrator that coordinates data analysis and ML model construction",
    )
