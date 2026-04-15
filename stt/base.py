"""
stt/base.py — Abstract base class for all Speech-to-Text backends.

Every STT implementation must subclass BaseSTT and implement `transcribe`.
This allows the rest of the system to swap backends without any code changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranscriptionResult:
    """Value object returned by every STT backend."""
    text: str                          # The transcribed text
    language: Optional[str] = None     # Detected language code (e.g. "en")
    confidence: Optional[float] = None # 0-1 confidence score if available
    backend: str = "unknown"           # Which backend produced this result
    duration_seconds: Optional[float] = None  # Audio length
    error: Optional[str] = None        # Populated on failure

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.text)


class BaseSTT(ABC):
    """
    Abstract Speech-to-Text interface.

    Usage::

        class MySTT(BaseSTT):
            def transcribe(self, audio_path: str) -> TranscriptionResult:
                ...
    """

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Convert an audio file to text.

        Parameters
        ----------
        audio_path : str
            Path to the audio file (WAV preferred).

        Returns
        -------
        TranscriptionResult
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
