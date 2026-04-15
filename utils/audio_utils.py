"""
utils/audio_utils.py — Audio pre-processing helpers.

Responsibilities:
  - Validate uploaded audio files
  - Convert mp3 → wav (Whisper needs WAV)
  - Capture live microphone audio and return a temp WAV path
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


# ── File Validation ───────────────────────────────────────────────────────────

def validate_audio_file(file_path: str) -> bool:
    """Return True if the file exists and has a supported extension."""
    path = Path(file_path)
    if not path.exists():
        logger.warning("Audio file not found: %s", file_path)
        return False
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning("Unsupported audio format: %s", path.suffix)
        return False
    return True


# ── Format Conversion ─────────────────────────────────────────────────────────

def convert_to_wav(input_path: str) -> str:
    """
    Convert any supported audio file to a temporary WAV file.
    Returns the path to the temporary WAV file.
    Requires: pydub + ffmpeg
    """
    try:
        from pydub import AudioSegment  # lazy import
    except ImportError:
        logger.error("pydub not installed. Run: pip install pydub")
        raise

    path = Path(input_path)
    logger.info("Converting %s → WAV", path.name)

    audio = AudioSegment.from_file(str(path))

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    logger.debug("Saved converted WAV to: %s", tmp.name)
    return tmp.name


def ensure_wav(file_path: str) -> str:
    """
    If the file is already WAV, return it unchanged.
    Otherwise convert and return the temp WAV path.
    """
    if Path(file_path).suffix.lower() == ".wav":
        return file_path
    return convert_to_wav(file_path)


# ── Live Microphone Recording ─────────────────────────────────────────────────

def record_microphone(duration_seconds: int = 5, sample_rate: int = 16_000) -> str:
    """
    Record audio from the default microphone for `duration_seconds` seconds.
    Returns the path to a temporary WAV file.
    Requires: sounddevice + scipy
    """
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write as wav_write
        import numpy as np
    except ImportError:
        logger.error(
            "sounddevice / scipy not installed. "
            "Run: pip install sounddevice scipy"
        )
        raise

    logger.info("Recording %d seconds from microphone…", duration_seconds)
    audio_data = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    logger.info("Recording complete.")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_write(tmp.name, sample_rate, audio_data)
    logger.debug("Saved microphone recording to: %s", tmp.name)
    return tmp.name


# ── Bytes → Temp File ─────────────────────────────────────────────────────────

def bytes_to_temp_wav(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """
    Write raw audio bytes to a temp file and return the path.
    Useful for handling Streamlit's UploadedFile objects.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(audio_bytes)
    tmp.flush()
    logger.debug("Written %d bytes to temp file: %s", len(audio_bytes), tmp.name)
    return tmp.name
