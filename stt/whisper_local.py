"""
stt/whisper_local.py — Local Whisper STT using OpenAI's `whisper` library.

The model is downloaded once and cached by the whisper library in ~/.cache/whisper.
Model sizes (speed ↔ accuracy trade-off):
    tiny   ~39M params  | fastest, least accurate
    base   ~74M params  | good balance  ← default
    small  ~244M params | better accuracy
    medium ~769M params | high accuracy
    large  ~1.5B params | best accuracy, slow
"""

import time
from pathlib import Path

from stt.base import BaseSTT, TranscriptionResult
from utils.audio_utils import ensure_wav
from utils.logger import get_logger
from config import WHISPER_MODEL_SIZE

logger = get_logger(__name__)


class WhisperLocalSTT(BaseSTT):
    """
    Transcribe audio using a locally-downloaded Whisper model.
    The model is loaded lazily on first use.
    """

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self._model = None  # lazy-loaded

    def _load_model(self):
        """Load Whisper model (called once, then cached on self._model)."""
        if self._model is None:
            try:
                import whisper  # openai-whisper package
            except ImportError:
                raise ImportError(
                    "openai-whisper not installed. Run: pip install openai-whisper"
                )
            logger.info("Loading Whisper model: %s", self.model_size)
            self._model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully.")
        return self._model

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Run local Whisper inference on the given audio file."""
        try:
            # Ensure we have a WAV file
            wav_path = ensure_wav(audio_path)

            model = self._load_model()

            logger.info("Transcribing: %s", Path(audio_path).name)
            t0 = time.perf_counter()

            result = model.transcribe(
                wav_path,
                language=None,       # auto-detect language
                fp16=False,          # safe default for CPU
                verbose=False,
            )

            elapsed = time.perf_counter() - t0
            text = result.get("text", "").strip()
            language = result.get("language")

            logger.info(
                "Transcription done in %.2fs | lang=%s | chars=%d",
                elapsed, language, len(text),
            )

            return TranscriptionResult(
                text=text,
                language=language,
                backend="whisper_local",
                duration_seconds=elapsed,
            )

        except Exception as exc:
            logger.exception("WhisperLocalSTT failed: %s", exc)
            return TranscriptionResult(
                text="",
                backend="whisper_local",
                error=str(exc),
            )
