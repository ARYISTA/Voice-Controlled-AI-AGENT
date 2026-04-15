"""
run_tests.py — Standalone test runner (no pytest needed).
Run: python run_tests.py
"""
import sys
sys.path.insert(0, '.')

from unittest.mock import MagicMock
from config import OUTPUT_DIR
from utils.file_safety import safe_output_path, SafePathError, is_safe_path
from utils.history import SessionHistory, AgentTurn
from tools.file_tool import FileTool
from tools.code_tool import CodeTool
from tools.summarize_tool import SummarizeTool
from tools.chat_tool import ChatTool
from tools.registry import dispatch
from llm.base import Intent
from llm.ollama_llm import OllamaLLM

passed = failed = 0

def ok(msg):
    global passed
    passed += 1
    print(f"  PASS  {msg}")

def fail(msg, err=""):
    global failed
    failed += 1
    print(f"  FAIL  {msg}  [{err}]")

print("\n=== File Safety ===")
try:
    p = safe_output_path("test.txt")
    assert str(OUTPUT_DIR.resolve()) in str(p)
    ok("safe path inside output")
except Exception as e:
    fail("safe path inside output", e)

try:
    safe_output_path("../../etc/passwd")
    fail("path traversal blocked — should have raised")
except SafePathError:
    ok("path traversal blocked")

try:
    assert is_safe_path("good.txt") is True
    assert is_safe_path("../../bad.txt") is False
    ok("is_safe_path correct")
except Exception as e:
    fail("is_safe_path", e)

print("\n=== FileTool ===")
mock_llm = MagicMock()
mock_llm.chat.return_value = "Generated content"
tool = FileTool()

try:
    r = tool.run({"filename": "unit_test.txt", "content": "hello"}, mock_llm, "test")
    assert r.success
    assert (OUTPUT_DIR / "unit_test.txt").read_text() == "hello"
    ok("FileTool writes file")
except Exception as e:
    fail("FileTool writes file", e)

try:
    r = tool.run({"filename": "../../evil.txt", "content": "bad"}, mock_llm, "")
    assert not r.success
    ok("FileTool blocks traversal")
except Exception as e:
    fail("FileTool blocks traversal", e)

try:
    r = tool.run({"filename": "no_content.txt"}, mock_llm, "make a readme")
    assert r.success
    mock_llm.chat.assert_called()
    ok("FileTool calls LLM when no content")
except Exception as e:
    fail("FileTool calls LLM when no content", e)

print("\n=== CodeTool ===")
mock_llm.chat.return_value = "def retry(func): pass"
code_tool = CodeTool()

try:
    r = code_tool.run(
        {"language": "python", "task": "retry function", "filename": "retry_test.py"},
        mock_llm, "write retry"
    )
    assert r.success
    assert "```" not in r.output
    ok("CodeTool generates and strips fences")
except Exception as e:
    fail("CodeTool generates and strips fences", e)

try:
    r = code_tool.run({"language": "python", "task": "bubble sort"}, mock_llm, "write bubble sort")
    assert r.success
    assert r.metadata["filename"].endswith(".py")
    ok("CodeTool infers filename")
except Exception as e:
    fail("CodeTool infers filename", e)

print("\n=== SummarizeTool ===")
mock_llm.chat.return_value = "Summary: AI is transforming technology."
sum_tool = SummarizeTool()

try:
    long_text = "Artificial intelligence " * 15
    r = sum_tool.run({"text": long_text}, mock_llm, "summarize")
    assert r.success
    ok("SummarizeTool summarises text")
except Exception as e:
    fail("SummarizeTool summarises text", e)

try:
    r = sum_tool.run({}, mock_llm, "hi")
    assert not r.success
    ok("SummarizeTool rejects short text")
except Exception as e:
    fail("SummarizeTool rejects short text", e)

print("\n=== ChatTool ===")
mock_llm.chat.return_value = "Paris is the capital of France."
chat_tool = ChatTool()

try:
    r = chat_tool.run({}, mock_llm, "What is the capital of France?")
    assert r.success and "Paris" in r.output
    ok("ChatTool returns LLM response")
except Exception as e:
    fail("ChatTool returns LLM response", e)

print("\n=== Registry ===")
mock_llm.chat.return_value = 'print("hello")'

try:
    r = dispatch(Intent.WRITE_CODE, {"language": "python", "task": "hello", "filename": "hi.py"}, mock_llm, "write hello")
    assert r.success
    ok("Registry dispatches write_code")
except Exception as e:
    fail("Registry dispatches write_code", e)

try:
    r = dispatch("unknown_intent", {}, mock_llm, "test")
    assert r.success
    ok("Registry falls back to chat for unknown intent")
except Exception as e:
    fail("Registry falls back to chat for unknown intent", e)

print("\n=== Intent JSON Parsing ===")
llm_obj = object.__new__(OllamaLLM)

try:
    raw = '{"intent": "write_code", "entities": {"language": "python"}, "confidence": 0.9}'
    r = llm_obj._parse_intent_json(raw, "test")
    assert r.intent == "write_code" and r.confidence == 0.9
    ok("Valid JSON parsed")
except Exception as e:
    fail("Valid JSON parsed", e)

try:
    raw = "```json\n{\"intent\": \"general_chat\", \"entities\": {}, \"confidence\": 0.7}\n```"
    r = llm_obj._parse_intent_json(raw, "test")
    assert r.intent == "general_chat"
    ok("Markdown fences stripped")
except Exception as e:
    fail("Markdown fences stripped", e)

try:
    raw = '{"intent": "make_coffee", "entities": {}, "confidence": 0.5}'
    r = llm_obj._parse_intent_json(raw, "test")
    assert r.intent == Intent.GENERAL_CHAT
    ok("Unknown intent falls back to general_chat")
except Exception as e:
    fail("Unknown intent falls back to general_chat", e)

try:
    r = llm_obj._parse_intent_json("Sorry I cannot help.", "test")
    assert r.error is not None
    ok("No JSON populates error field")
except Exception as e:
    fail("No JSON populates error field", e)

print("\n=== SessionHistory ===")
try:
    hist = SessionHistory()
    hist.add(AgentTurn(raw_text="hello", intent="general_chat"))
    hist.add(AgentTurn(raw_text="write code", intent="write_code"))
    assert len(hist) == 2
    assert len(hist.recent(1)) == 1
    hist.clear()
    assert len(hist) == 0
    ok("SessionHistory accumulates and clears")
except Exception as e:
    fail("SessionHistory accumulates and clears", e)

total = passed + failed
print(f"\n{'='*40}")
print(f"Results: {passed}/{total} passed  {'✅ ALL PASS' if failed == 0 else f'❌ {failed} FAILED'}")
print(f"{'='*40}\n")
sys.exit(0 if failed == 0 else 1)
