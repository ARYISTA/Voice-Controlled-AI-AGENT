"""
tools/chat_tool.py — General-purpose conversational tool.

Handles intent: general_chat

Falls through here when no specialised tool matches.
"""

from tools.base import BaseTool, ToolResult
from utils.logger import get_logger

logger = get_logger(__name__)

CHAT_SYSTEM_PROMPT = """You are a helpful, conversational AI assistant.
Answer the user's question clearly and concisely.
Be friendly but professional. Use markdown formatting where helpful."""


class ChatTool(BaseTool):
    """Return a conversational LLM response."""

    @property
    def name(self) -> str:
        return "general_chat"

    def run(self, entities: dict, llm, raw_text: str) -> ToolResult:
        logger.info("Running general chat for: %s…", raw_text[:60])

        response = llm.chat(prompt=raw_text, system_prompt=CHAT_SYSTEM_PROMPT)

        return ToolResult(
            success=True,
            output=response,
            action_taken="Responded conversationally.",
            metadata={"topic": entities.get("topic", "general")},
        )
