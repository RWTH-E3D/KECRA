"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated

from langchain_core.runnables import ensure_config
from langgraph.config import get_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    # task_parse_system_prompt: str = field(default=prompts.TASK_PARSE_SYSTEM_PROMPT)
    task_parse_user_prompt:   str = field(default=prompts.TASK_PARSE_USER_PROMPT)
    # action_parse_system_prompt: str = field(default=prompts.ACTION_PARSE_SYSTEM_PROMPT)
    # action_parse_user_prompt:   str = field(default=prompts.ACTION_PARSE_USER_PROMPT)
    # pose_gen_system_prompt:     str = field(default=prompts.POSE_GEN_SYSTEM_PROMPT)
    pose_gen_user_prompt: str = field(default=prompts.POSE_GEN_USER_PROMPT)
    bad_pose_fix_user_prompt: str = field(default=prompts.BAD_POSE_FIX_USER_PROMPT)

    # model: str = field(default="deepseek-chat")
    # model: str = field(default="deepseek-reasoner")
    # model: str = field(default="gpt-4o")

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        # default="anthropic/claude-3-5-sonnet-20240620",
        default="openai/gpt-4o",
        # default="deepseek/deepseek-chat",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = 10

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})

