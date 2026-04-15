"""
tests/test_agent.py — Unit + integration tests for the Voice AI Agent.

Run with:
    pytest tests/ -v

Tests use mock LLM and STT backends to avoid requiring Ollama/Whisper.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUT_DIR
from llm.base import IntentResult, Intent
from stt.base import TranscriptionResult
from tools.base import ToolResult
from tools.file_tool import FileTool
from tools.code_tool import CodeTool
from tools.summarize_tool import SummarizeTool
from tools.chat_tool import ChatTool
from tools.registry import dispatch
from utils.file_safety import safe_output_path, SafePathError, is_safe_path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """A mock LLM that returns predictable responses."""
    llm = MagicMock()
    llm.chat.return_value = "Mock LLM response for testing."
    llm.classify_intent.return_value = IntentResult(
        intent=Intent.GENERAL_CHAT,
        entities={"topic": "test"},
        confidence=0.9,
        raw_text="test input",
    )
    return llm


@pytest.fixture
def mock_stt():
    """A mock STT that returns a fixed transcription."""
    stt = MagicMock()
    stt.transcribe.return_value = TranscriptionResult(
        text="Create a Python file with a retry function",
        language="en",
        backend="mock",
    )
    return stt


# ── File Safety Tests ─────────────────────────────────────────────────────────

class TestFileSafety:

    def test_safe_path_inside_output(self):
        path = safe_output_path("test_output.txt")
        assert str(OUTPUT_DIR.resolve()) in str(path)

    def test_safe_path_creates_parent_dirs(self):
        path = safe_output_path("subdir/nested/file.txt")
        assert path.parent.exists()

    def test_path_traversal_blocked(self):
        with pytest.raises(SafePathError):
            safe_output_path("../../etc/passwd")

    def test_path_traversal_double_dot(self):
        with pytest.raises(SafePathError):
            safe_output_path("../config.py")

    def test_is_safe_path_returns_false_on_traversal(self):
        assert is_safe_path("../../secret") is False

    def test_is_safe_path_returns_true_for_valid(self):
        assert is_safe_path("safe_file.txt") is True


# ── FileTool Tests ────────────────────────────────────────────────────────────

class TestFileTool:

    def test_write_simple_file(self, mock_llm):
        tool = FileTool()
        result = tool.run(
            entities={"filename": "test_simple.txt", "content": "Hello, World!"},
            llm=mock_llm,
            raw_text="create a file",
        )
        assert result.success
        target = OUTPUT_DIR / "test_simple.txt"
        assert target.exists()
        assert target.read_text() == "Hello, World!"

    def test_write_file_no_content_calls_llm(self, mock_llm):
        tool = FileTool()
        result = tool.run(
            entities={"filename": "llm_generated.txt"},
            llm=mock_llm,
            raw_text="create a readme file",
        )
        assert result.success
        mock_llm.chat.assert_called_once()

    def test_path_traversal_blocked_in_tool(self, mock_llm):
        tool = FileTool()
        result = tool.run(
            entities={"filename": "../../evil.txt", "content": "bad"},
            llm=mock_llm,
            raw_text="",
        )
        assert not result.success
        assert "Security" in result.error

    def test_write_file_returns_metadata(self, mock_llm):
        tool = FileTool()
        result = tool.write_file("meta_test.txt", "content for metadata test")
        assert result.success
        assert "path" in result.metadata
        assert "size_kb" in result.metadata


# ── CodeTool Tests ────────────────────────────────────────────────────────────

class TestCodeTool:

    def test_generates_python_code(self, mock_llm):
        mock_llm.chat.return_value = "def retry(func):\n    pass\n"
        tool = CodeTool()
        result = tool.run(
            entities={"language": "python", "task": "retry decorator", "filename": "retry.py"},
            llm=mock_llm,
            raw_text="create a Python retry decorator",
        )
        assert result.success
        assert result.metadata["language"] == "python"
        target = Path(result.metadata["path"])
        assert target.exists()
        assert "retry" in target.read_text()

    def test_strips_markdown_fences(self, mock_llm):
        mock_llm.chat.return_value = "```python\ndef hello():\n    pass\n```"
        tool = CodeTool()
        result = tool.run(
            entities={"language": "python", "task": "hello function"},
            llm=mock_llm,
            raw_text="write a hello function",
        )
        assert result.success
        assert "```" not in result.output

    def test_infers_filename_from_task(self, mock_llm):
        mock_llm.chat.return_value = "def bubble_sort(arr): pass"
        tool = CodeTool()
        result = tool.run(
            entities={"language": "python", "task": "bubble sort algorithm"},
            llm=mock_llm,
            raw_text="write bubble sort",
        )
        assert result.success
        # filename should contain words from task
        fname = Path(result.metadata["path"]).name
        assert fname.endswith(".py")


# ── SummarizeTool Tests ───────────────────────────────────────────────────────

class TestSummarizeTool:

    def test_summarizes_provided_text(self, mock_llm):
        mock_llm.chat.return_value = "Summary: This is about AI."
        tool = SummarizeTool()
        long_text = "Artificial intelligence " * 20  # 40 words
        result = tool.run(
            entities={"text": long_text},
            llm=mock_llm,
            raw_text="summarize this text",
        )
        assert result.success
        assert "Summary" in result.output

    def test_fails_on_too_short_text(self, mock_llm):
        tool = SummarizeTool()
        result = tool.run(
            entities={},
            llm=mock_llm,
            raw_text="hi",
        )
        assert not result.success
        assert "insufficient" in result.error.lower() or "enough" in result.error.lower()

    def test_strips_summarize_prefix(self, mock_llm):
        mock_llm.chat.return_value = "A summary of the provided text."
        tool = SummarizeTool()
        long_text = "please summarize " + ("the quick brown fox jumps " * 5)
        result = tool.run(
            entities={},
            llm=mock_llm,
            raw_text=long_text,
        )
        assert result.success


# ── ChatTool Tests ────────────────────────────────────────────────────────────

class TestChatTool:

    def test_returns_llm_response(self, mock_llm):
        mock_llm.chat.return_value = "The capital of France is Paris."
        tool = ChatTool()
        result = tool.run(
            entities={"topic": "geography"},
            llm=mock_llm,
            raw_text="What is the capital of France?",
        )
        assert result.success
        assert "Paris" in result.output


# ── Registry / Dispatch Tests ─────────────────────────────────────────────────

class TestRegistry:

    def test_dispatch_write_code(self, mock_llm):
        mock_llm.chat.return_value = "print('hello')"
        result = dispatch(
            intent=Intent.WRITE_CODE,
            entities={"language": "python", "task": "hello world", "filename": "hello.py"},
            llm=mock_llm,
            raw_text="write hello world in python",
        )
        assert result.success

    def test_dispatch_general_chat(self, mock_llm):
        result = dispatch(
            intent=Intent.GENERAL_CHAT,
            entities={},
            llm=mock_llm,
            raw_text="What time is it?",
        )
        assert result.success

    def test_dispatch_unknown_intent_falls_back_to_chat(self, mock_llm):
        result = dispatch(
            intent="nonexistent_intent",
            entities={},
            llm=mock_llm,
            raw_text="mystery input",
        )
        assert result.success


# ── Agent Integration Tests ───────────────────────────────────────────────────

class TestAgentIntegration:

    def test_process_text_write_code(self, mock_llm, mock_stt):
        from agent import VoiceAgent
        mock_llm.classify_intent.return_value = IntentResult(
            intent=Intent.WRITE_CODE,
            entities={"language": "python", "task": "fibonacci sequence", "filename": "fib.py"},
            confidence=0.95,
            raw_text="write a fibonacci function in python",
        )
        mock_llm.chat.return_value = "def fib(n): return n if n < 2 else fib(n-1)+fib(n-2)"

        agent = VoiceAgent(stt_backend=mock_stt, llm_backend=mock_llm)
        response = agent.process_text("write a fibonacci function in python")

        assert response.intent_result.intent == Intent.WRITE_CODE
        assert response.tool_result.success
        assert len(agent.history) == 1

    def test_process_text_general_chat(self, mock_llm, mock_stt):
        from agent import VoiceAgent
        mock_llm.classify_intent.return_value = IntentResult(
            intent=Intent.GENERAL_CHAT,
            entities={},
            confidence=0.8,
            raw_text="hello",
        )
        mock_llm.chat.return_value = "Hello! How can I help you today?"

        agent = VoiceAgent(stt_backend=mock_stt, llm_backend=mock_llm)
        response = agent.process_text("hello")

        assert response.success
        assert "Hello" in response.tool_result.output

    def test_history_accumulates(self, mock_llm, mock_stt):
        from agent import VoiceAgent
        agent = VoiceAgent(stt_backend=mock_stt, llm_backend=mock_llm)
        agent.process_text("first command")
        agent.process_text("second command")
        assert len(agent.history) == 2


# ── Intent JSON Parsing Tests ─────────────────────────────────────────────────

class TestIntentParsing:

    def _make_llm(self):
        from llm.ollama_llm import OllamaLLM
        llm = OllamaLLM.__new__(OllamaLLM)
        return llm

    def test_valid_json_parsed(self):
        llm = self._make_llm()
        raw = '{"intent": "write_code", "entities": {"language": "python"}, "confidence": 0.9}'
        result = llm._parse_intent_json(raw, "test")
        assert result.intent == "write_code"
        assert result.confidence == 0.9

    def test_markdown_fence_stripped(self):
        llm = self._make_llm()
        raw = '```json\n{"intent": "general_chat", "entities": {}, "confidence": 0.7}\n```'
        result = llm._parse_intent_json(raw, "test")
        assert result.intent == "general_chat"

    def test_invalid_intent_falls_back_to_chat(self):
        llm = self._make_llm()
        raw = '{"intent": "make_coffee", "entities": {}, "confidence": 0.5}'
        result = llm._parse_intent_json(raw, "test")
        assert result.intent == Intent.GENERAL_CHAT

    def test_no_json_returns_error_result(self):
        llm = self._make_llm()
        result = llm._parse_intent_json("Sorry, I cannot help.", "test")
        assert result.intent == Intent.GENERAL_CHAT
        assert result.error is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
