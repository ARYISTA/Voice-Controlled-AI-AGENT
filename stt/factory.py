"""
stt/factory.py — STT backend factory.

Reads STT_BACKEND from config and returns the appropriate BaseSTT instance.
Adding a new backend = one new elif branch here.
"""

from config import STT_BACKEND
from stt.base import BaseSTT
from utils.logger import get_logger

logger = get_logger(__name__)


def get_stt_backend(backend: str = STT_BACKEND) -> BaseSTT:
    """
    Return an instantiated STT backend.

    Parameters
    ----------
    backend : str
        One of: "whisper_local" | "groq" | "openai"

    Returns
    -------
    BaseSTT subclass instance
    """
    backend = backend.lower().strip()
    logger.info("Initialising STT backend: %s", backend)

    if backend == "whisper_local":
        from stt.whisper_local import WhisperLocalSTT
        return WhisperLocalSTT()

    elif backend == "groq":
        from stt.groq_stt import GroqSTT
        return GroqSTT()

    else:
        logger.warning(
            "Unknown STT backend '%s', falling back to whisper_local.", backend
        )
        from stt.whisper_local import WhisperLocalSTT
        return WhisperLocalSTT()
