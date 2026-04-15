"""
utils/tts.py — Text-to-Speech (voice feedback) module.

Supports two engines:
  - pyttsx3 : offline, no API key needed (default)
  - gtts    : Google TTS, requires internet

Engine is selected via TTS_ENGINE env var.
TTS can be disabled entirely by setting TTS_ENABLED=false.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from utils.logger import get_logger
from config import TTS_ENABLED, TTS_ENGINE

logger = get_logger(__name__)


def speak(text: str, engine: str = TTS_ENGINE) -> Optional[str]:
    """
    Speak `text` aloud using the configured TTS engine.

    Parameters
    ----------
    text   : str  — Text to convert to speech
    engine : str  — "pyttsx3" | "gtts"

    Returns
    -------
    str | None  — Path to generated audio file (if applicable), else None
    """
    if not TTS_ENABLED:
        logger.debug("TTS disabled; skipping speak().")
        return None

    if not text or not text.strip():
        return None

    # Truncate very long responses for TTS
    tts_text = text[:500].strip()
    if len(text) > 500:
        tts_text += "… (response truncated for speech)"

    engine = engine.lower()

    try:
        if engine == "pyttsx3":
            return _speak_pyttsx3(tts_text)
        elif engine == "gtts":
            return _speak_gtts(tts_text)
        else:
            logger.warning("Unknown TTS engine: %s; using pyttsx3.", engine)
            return _speak_pyttsx3(tts_text)
    except Exception as exc:
        logger.warning("TTS failed (%s): %s", engine, exc)
        return None


# ── pyttsx3 (offline) ─────────────────────────────────────────────────────────

def _speak_pyttsx3(text: str) -> None:
    """Speak using pyttsx3 (offline, cross-platform)."""
    try:
        import pyttsx3
    except ImportError:
        raise ImportError("pyttsx3 not installed. Run: pip install pyttsx3")

    engine = pyttsx3.init()
    engine.setProperty("rate", 165)   # words per minute
    engine.setProperty("volume", 0.9)

    # Try to pick a natural-sounding voice
    voices = engine.getProperty("voices")
    for v in voices:
        if "english" in v.name.lower():
            engine.setProperty("voice", v.id)
            break

    logger.debug("pyttsx3 speaking: %s…", text[:50])
    engine.say(text)
    engine.runAndWait()
    return None


# ── gTTS (online, higher quality) ────────────────────────────────────────────

def _speak_gtts(text: str) -> str:
    """
    Synthesise speech using Google TTS and return the path to the MP3 file.
    The caller is responsible for playing and cleaning up the file.
    """
    try:
        from gtts import gTTS
    except ImportError:
        raise ImportError("gtts not installed. Run: pip install gtts")

    tts = gTTS(text=text, lang="en", slow=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(tmp.name)
    logger.debug("gTTS saved to: %s", tmp.name)
    return tmp.name
