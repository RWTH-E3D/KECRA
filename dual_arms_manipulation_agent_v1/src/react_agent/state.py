"""Define the shared values."""

from __future__ import annotations
from typing import List, Any, Optional

from dataclasses import dataclass, field

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


@dataclass(kw_only=True)
class State:
    messages: Annotated[List[AnyMessage], add_messages] = field(default_factory=list)

    tasks:   List[str] = field(default_factory=list)
    # task_idx: int = 0

    # actions: List[str] = field(default_factory=list)
    # act_idx:  int = 0
    feedback: Optional[str] = None
    bad_poses: List[List[float]] = field(default_factory=list)
    poses: List[List[float]] = field(default_factory=list)



__all__ = [
    "State",
]