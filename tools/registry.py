"""
tools/registry.py — Tool registry and dispatcher.

Maps intent strings → tool instances.
Adding a new tool = register it in TOOL_REGISTRY.
"""

from llm.base import Intent
from tools.base import BaseTool, ToolResult
from tools.file_tool import FileTool
from tools.code_tool import CodeTool
from tools.summarize_tool import SummarizeTool
from tools.chat_tool import ChatTool
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────
# Maps intent value → tool instance
TOOL_REGISTRY: dict[str, BaseTool] = {
    Intent.CREATE_FILE:    FileTool(),
    Intent.WRITE_CODE:     CodeTool(),
    Intent.SUMMARIZE_TEXT: SummarizeTool(),
    Intent.GENERAL_CHAT:   ChatTool(),
}


def dispatch(intent: str, entities: dict, llm, raw_text: str) -> ToolResult:
    """
    Look up the correct tool for `intent` and execute it.

    Parameters
    ----------
    intent   : str  — Intent value (e.g. "write_code")
    entities : dict — Entities from the LLM classifier
    llm           — Active BaseLLM instance
    raw_text : str  — Original user utterance

    Returns
    -------
    ToolResult
    """
    tool = TOOL_REGISTRY.get(intent)

    if tool is None:
        logger.warning("No tool registered for intent '%s'; using chat.", intent)
        tool = TOOL_REGISTRY[Intent.GENERAL_CHAT]

    logger.info("Dispatching to tool: %s", tool.name)
    return tool.run(entities=entities, llm=llm, raw_text=raw_text)


def list_tools() -> list[str]:
    """Return the names of all registered tools."""
    return list(TOOL_REGISTRY.keys())
