"""
tools/file_tool.py — File creation tool (sandboxed to ./output/).

Handles intent: create_file
Also used as a post-step by the code tool to persist generated files.
"""

from pathlib import Path

from tools.base import BaseTool, ToolResult
from utils.file_safety import safe_output_path, SafePathError
from utils.logger import get_logger

logger = get_logger(__name__)


class FileTool(BaseTool):
    """
    Create or overwrite a file inside ./output/.

    Expected entities
    -----------------
    filename : str   — target file name (e.g. "notes.txt", "data/readme.md")
    content  : str   — text content to write (default: empty string)
    """

    @property
    def name(self) -> str:
        return "create_file"

    def run(self, entities: dict, llm, raw_text: str) -> ToolResult:
        filename = entities.get("filename") or entities.get("file_name") or "output.txt"
        content  = entities.get("content", "")

        # If no content was extracted, ask the LLM to draft it
        if not content and raw_text:
            logger.info("No content in entities; prompting LLM to draft file content.")
            content = llm.chat(
                prompt=f"Draft the content for a file called '{filename}' "
                       f"based on: {raw_text}",
                system_prompt=(
                    "You are a helpful assistant. Write clean, well-structured "
                    "content for a text file. Output only the file content — "
                    "no explanations, no markdown fences."
                ),
            )

        return self.write_file(filename, content)

    # ── Public helper (also called by CodeTool) ────────────────────────────

    def write_file(self, filename: str, content: str) -> ToolResult:
        """
        Write `content` to `filename` inside the output sandbox.
        Returns a ToolResult describing what happened.
        """
        try:
            target: Path = safe_output_path(filename)
        except SafePathError as exc:
            return ToolResult(
                success=False,
                error=f"Security: {exc}",
                action_taken=f"Blocked write to '{filename}'",
            )

        try:
            target.write_text(content, encoding="utf-8")
            size_kb = target.stat().st_size / 1024
            logger.info("File written: %s (%.1f KB)", target, size_kb)

            return ToolResult(
                success=True,
                output=str(target),
                action_taken=f"Created file: {target.relative_to(target.parent.parent)}",
                metadata={"path": str(target), "size_kb": round(size_kb, 2)},
            )
        except OSError as exc:
            logger.error("Failed to write file %s: %s", filename, exc)
            return ToolResult(
                success=False,
                error=str(exc),
                action_taken=f"Failed to write '{filename}'",
            )
