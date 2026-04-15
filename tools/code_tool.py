"""
tools/code_tool.py — Code generation + file persistence tool.

Handles intent: write_code

Flow:
  1. Build a precise coding prompt from entities + raw_text.
  2. Call the LLM to generate clean code.
  3. Strip markdown fences from the response.
  4. Save the code to ./output/<filename> via FileTool.
"""

import re
from pathlib import Path

from tools.base import BaseTool, ToolResult
from tools.file_tool import FileTool
from utils.logger import get_logger

logger = get_logger(__name__)

# Language → file extension mapping
LANG_EXTENSIONS = {
    "python":     ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "java":       ".java",
    "c":          ".c",
    "cpp":        ".cpp",
    "c++":        ".cpp",
    "go":         ".go",
    "rust":       ".rs",
    "bash":       ".sh",
    "shell":      ".sh",
    "sql":        ".sql",
    "html":       ".html",
    "css":        ".css",
    "json":       ".json",
    "yaml":       ".yaml",
    "markdown":   ".md",
}


class CodeTool(BaseTool):
    """
    Generate code using the LLM and save it to ./output/.

    Expected entities
    -----------------
    language : str   — programming language (default: "python")
    task     : str   — description of what the code should do
    filename : str   — target filename (auto-generated if absent)
    """

    @property
    def name(self) -> str:
        return "write_code"

    def run(self, entities: dict, llm, raw_text: str) -> ToolResult:
        language = (entities.get("language") or "python").lower()
        task     = entities.get("task") or raw_text
        filename = entities.get("filename") or self._infer_filename(task, language)

        system_prompt = (
            f"You are an expert {language} developer. "
            "Write clean, production-ready, well-commented code. "
            "Output ONLY the raw code — no explanations, no markdown fences. "
            "Include docstrings and type hints where appropriate."
        )

        prompt = (
            f"Write {language} code for the following task:\n\n"
            f"{task}\n\n"
            "Requirements:\n"
            "- Clean, readable, production-quality code\n"
            "- Comprehensive docstring / module comment\n"
            "- Error handling where appropriate\n"
            "- Example usage at the bottom (if applicable)\n"
        )

        logger.info("Generating %s code for: %s", language, task[:80])
        raw_code = llm.chat(prompt=prompt, system_prompt=system_prompt)

        # Strip markdown fences that some models insist on including
        code = self._strip_fences(raw_code)

        # Persist to output directory
        file_tool = FileTool()
        file_result = file_tool.write_file(filename, code)

        if not file_result.success:
            return file_result

        return ToolResult(
            success=True,
            output=code,
            action_taken=(
                f"Generated {language} code and saved to "
                f"{file_result.output}"
            ),
            metadata={
                "language": language,
                "filename": filename,
                "path":     file_result.output,
                "lines":    code.count("\n") + 1,
            },
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _infer_filename(self, task: str, language: str) -> str:
        """Generate a reasonable filename from the task description."""
        # Grab the first 3 meaningful words from the task
        words = re.findall(r"[a-zA-Z]+", task)[:3]
        stem  = "_".join(w.lower() for w in words) or "generated_code"
        ext   = LANG_EXTENSIONS.get(language, ".txt")
        return stem + ext

    def _strip_fences(self, text: str) -> str:
        """Remove ```language ... ``` markdown code fences."""
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
        text = re.sub(r"\n?```$", "", text.strip())
        return text.strip()
