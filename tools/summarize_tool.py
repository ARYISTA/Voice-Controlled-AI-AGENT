"""
tools/summarize_tool.py — Text summarization tool.

Handles intent: summarize_text

The text to summarise can arrive via:
  a) entities["text"]   — extracted directly from speech
  b) entities["file"]   — path to a file whose contents to summarise
  c) raw_text fallback  — the user's full utterance
"""

from pathlib import Path

from tools.base import BaseTool, ToolResult
from utils.logger import get_logger

logger = get_logger(__name__)


class SummarizeTool(BaseTool):
    """Summarize user-provided text using the active LLM."""

    @property
    def name(self) -> str:
        return "summarize_text"

    def run(self, entities: dict, llm, raw_text: str) -> ToolResult:
        text = self._resolve_text(entities, raw_text)

        if not text or len(text.split()) < 10:
            return ToolResult(
                success=False,
                error="Not enough text to summarise (need at least 10 words).",
                action_taken="Summarisation skipped — insufficient text.",
            )

        word_count = len(text.split())
        logger.info("Summarising %d words…", word_count)

        system_prompt = (
            "You are an expert summariser. Produce a concise, accurate summary "
            "that captures all key points. Use clear language. "
            "Format: a short paragraph followed by 3-5 bullet points."
        )

        prompt = (
            f"Summarise the following text:\n\n"
            f"---\n{text}\n---\n\n"
            "Provide a summary paragraph and key bullet points."
        )

        summary = llm.chat(prompt=prompt, system_prompt=system_prompt)
        logger.info("Summary generated (%d chars).", len(summary))

        return ToolResult(
            success=True,
            output=summary,
            action_taken=f"Summarised {word_count}-word text.",
            metadata={"original_word_count": word_count, "summary_chars": len(summary)},
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _resolve_text(self, entities: dict, raw_text: str) -> str:
        """
        Determine the text to summarise from multiple possible sources.
        Priority: entities["text"] > entities["file"] content > raw_text
        """
        if entities.get("text"):
            return entities["text"]

        if entities.get("file"):
            file_path = Path(entities["file"])
            if file_path.exists():
                logger.info("Reading file for summarisation: %s", file_path)
                return file_path.read_text(encoding="utf-8")
            else:
                logger.warning("File not found: %s", file_path)

        # Fall back to raw utterance (strip common preambles)
        text = raw_text
        for prefix in ("summarise", "summarize", "summarize this", "summarise this",
                        "can you summarise", "please summarise"):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()

        return text
