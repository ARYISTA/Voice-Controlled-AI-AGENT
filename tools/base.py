"""
tools/base.py — Abstract base class for all agent tools.

Every tool returns a ToolResult that captures success/failure uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """
    Uniform result object returned by every tool.

    Fields
    ------
    success      : True if the tool completed without error.
    output       : The primary result (file path, generated text, etc.)
    action_taken : Human-readable description of what the tool did.
    metadata     : Extra key-value data (file size, language, etc.)
    error        : Error message if success is False.
    """
    success: bool
    output: str = ""
    action_taken: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.success:
            return f"✅ {self.action_taken}\n\n{self.output}"
        return f"❌ {self.error}"


class BaseTool(ABC):
    """
    Abstract tool interface.

    Usage::

        class MyTool(BaseTool):
            @property
            def name(self) -> str:
                return "my_tool"

            def run(self, entities: dict, llm, raw_text: str) -> ToolResult:
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""

    @abstractmethod
    def run(self, entities: dict, llm, raw_text: str) -> ToolResult:
        """
        Execute the tool.

        Parameters
        ----------
        entities : dict   — Entities extracted by the LLM intent classifier.
        llm               — The active BaseLLM instance (for code/text generation).
        raw_text : str    — The original user utterance.

        Returns
        -------
        ToolResult
        """

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"
