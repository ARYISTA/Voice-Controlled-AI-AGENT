"""
stt/groq_stt.py — Cloud STT via Groq's Whisper API.

Use when local Whisper is not feasible (e.g. very limited RAM).
Requires: pip install groq
Set env var: GROQ_API_KEY=<your-key>
"""

import time
from pathlib import Path

from stt.base import BaseSTT, TranscriptionResult
from utils.audio_utils import ensure_wav
from utils.logger import get_logger
from config import GROQ_API_KEY

logger = get_logger(__name__)


class GroqSTT(BaseSTT):
    """
    Transcribe audio using Groq's cloud Whisper API.
    Groq offers free-tier access with very fast inference.
    """

    def __init__(self, api_key: str = GROQ_API_KEY, model: str = "whisper-large-v3"):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Export it or set STT_BACKEND=whisper_local to use local Whisper."
            )
        self.api_key = api_key
        self.model = model

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        try:
            from groq import Groq  # pip install groq
        except ImportError:
            raise ImportError("groq not installed. Run: pip install groq")

        try:
            wav_path = ensure_wav(audio_path)
            client = Groq(api_key=self.api_key)

            logger.info("Sending audio to Groq Whisper API: %s", Path(audio_path).name)
            t0 = time.perf_counter()

            with open(wav_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    file=(Path(wav_path).name, f),
                    model=self.model,
                    response_format="verbose_json",
                )

            elapsed = time.perf_counter() - t0
            text = response.text.strip() if hasattr(response, "text") else ""
            language = getattr(response, "language", None)

            logger.info(
                "Groq transcription done in %.2fs | lang=%s", elapsed, language
            )
            return TranscriptionResult(
                text=text,
                language=language,
                backend="groq",
                duration_seconds=elapsed,
            )

        except Exception as exc:
            logger.exception("GroqSTT failed: %s", exc)
            return TranscriptionResult(text="", backend="groq", error=str(exc))
