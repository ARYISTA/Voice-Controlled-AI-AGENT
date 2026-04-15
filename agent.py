"""
agent.py — Core Agent Pipeline Orchestrator.

Wires all modules together:
  Audio → STT → LLM Intent → Tool Dispatch → TTS → History

The Agent class is the single public interface the UI talks to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import LLM_BACKEND, STT_BACKEND, TTS_ENABLED
from llm.base import BaseLLM, IntentResult
from llm.factory import get_llm_backend
from stt.base import BaseSTT, TranscriptionResult
from stt.factory import get_stt_backend
from tools.base import ToolResult
from tools.registry import dispatch
from utils.audio_utils import bytes_to_temp_wav, ensure_wav, validate_audio_file
from utils.history import AgentTurn, SessionHistory
from utils.logger import get_logger
from utils.tts import speak

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """Full pipeline output returned to the UI."""
    transcription: TranscriptionResult
    intent_result: IntentResult
    tool_result: ToolResult
    turn: AgentTurn
    tts_audio_path: Optional[str] = None
    total_elapsed: float = 0.0

    @property
    def success(self) -> bool:
        return self.tool_result.success


class VoiceAgent:
    """
    Voice-Controlled AI Agent.

    Usage::

        agent = VoiceAgent()
        response = agent.process_audio_file("recording.wav")
        print(response.tool_result.output)
    """

    def __init__(
        self,
        stt_backend: Optional[BaseSTT] = None,
        llm_backend: Optional[BaseLLM] = None,
    ):
        self.stt: BaseSTT = stt_backend or get_stt_backend(STT_BACKEND)
        self.llm: BaseLLM = llm_backend or get_llm_backend(LLM_BACKEND)
        self.history = SessionHistory()
        logger.info(
            "VoiceAgent ready | STT=%s | LLM=%s",
            self.stt.__class__.__name__,
            self.llm.__class__.__name__,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process_audio_file(self, audio_path: str) -> AgentResponse:
        """
        Full pipeline: audio file → transcription → intent → tool → response.

        Parameters
        ----------
        audio_path : str — Path to an audio file (WAV / MP3 / etc.)
        """
        if not validate_audio_file(audio_path):
            transcription = TranscriptionResult(
                text="", backend=STT_BACKEND,
                error=f"Invalid or missing audio file: {audio_path}"
            )
            return self._error_response(transcription)

        t0 = time.perf_counter()

        # ── Step 1: STT ───────────────────────────────────────────────────────
        logger.info("Step 1/3: Transcribing audio…")
        transcription = self.stt.transcribe(audio_path)

        if not transcription.success:
            return self._error_response(transcription)

        return self._process_text(transcription, t0)

    def process_audio_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> AgentResponse:
        """
        Convenience method for Streamlit's UploadedFile.read() output.
        Writes bytes to a temp file then runs the standard pipeline.
        """
        tmp_path = bytes_to_temp_wav(audio_bytes, suffix=suffix)
        return self.process_audio_file(tmp_path)

    def process_text(self, text: str) -> AgentResponse:
        """
        Skip STT and run intent + tool pipeline directly on `text`.
        Useful for keyboard input or testing.
        """
        transcription = TranscriptionResult(
            text=text,
            backend="text_input",
            language="en",
        )
        return self._process_text(transcription, time.perf_counter())

    # ── Private Pipeline ──────────────────────────────────────────────────────

    def _process_text(
        self, transcription: TranscriptionResult, t0: float
    ) -> AgentResponse:
        text = transcription.text

        # ── Step 2: Intent Classification ─────────────────────────────────────
        logger.info("Step 2/3: Classifying intent for: %s…", text[:80])
        intent_result: IntentResult = self.llm.classify_intent(text)

        # ── Step 3: Tool Dispatch ─────────────────────────────────────────────
        logger.info(
            "Step 3/3: Dispatching → intent=%s confidence=%.2f",
            intent_result.intent,
            intent_result.confidence,
        )
        tool_result: ToolResult = dispatch(
            intent=intent_result.intent,
            entities=intent_result.entities,
            llm=self.llm,
            raw_text=text,
        )

        elapsed = time.perf_counter() - t0

        # ── TTS (voice feedback) ──────────────────────────────────────────────
        tts_audio = None
        if TTS_ENABLED and tool_result.success:
            tts_text = self._tts_summary(intent_result, tool_result)
            tts_audio = speak(tts_text)

        # ── History ───────────────────────────────────────────────────────────
        turn = AgentTurn(
            raw_text=text,
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            entities=intent_result.entities,
            action_taken=tool_result.action_taken,
            output=tool_result.output[:500],   # truncate for storage
            success=tool_result.success,
            error=tool_result.error,
            stt_backend=transcription.backend,
            llm_backend=self.llm.__class__.__name__,
        )
        self.history.add(turn)

        logger.info(
            "Pipeline complete in %.2fs | success=%s", elapsed, tool_result.success
        )

        return AgentResponse(
            transcription=transcription,
            intent_result=intent_result,
            tool_result=tool_result,
            turn=turn,
            tts_audio_path=tts_audio,
            total_elapsed=elapsed,
        )

    def _error_response(self, transcription: TranscriptionResult) -> AgentResponse:
        """Build a failed AgentResponse when STT itself fails."""
        from llm.base import IntentResult, Intent

        intent_result = IntentResult(
            intent=Intent.GENERAL_CHAT,
            confidence=0.0,
            error="STT failed",
        )
        tool_result = ToolResult(
            success=False,
            error=transcription.error or "Unknown STT error",
            action_taken="Pipeline aborted at STT stage.",
        )
        turn = AgentTurn(
            raw_text="",
            intent="error",
            success=False,
            error=transcription.error,
        )
        self.history.add(turn)
        return AgentResponse(
            transcription=transcription,
            intent_result=intent_result,
            tool_result=tool_result,
            turn=turn,
        )

    def _tts_summary(self, intent: IntentResult, result: ToolResult) -> str:
        """Generate a brief speech summary of what the agent did."""
        summaries = {
            "write_code":     f"Done! I generated the code and saved it.",
            "create_file":    f"Done! I created the file for you.",
            "summarize_text": f"Here's your summary.",
            "general_chat":   result.output[:200],
        }
        return summaries.get(intent.intent, result.action_taken)
