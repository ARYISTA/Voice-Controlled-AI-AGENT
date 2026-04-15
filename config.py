"""
config.py — Central configuration for the Voice Agent.
All environment variables, model paths, and tunable knobs live here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # ✅ THIS LINE IS MISSING


# ── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR    = BASE_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── STT Configuration ────────────────────────────────────────────────────────
# Options: "whisper_local" | "groq" | "openai"
STT_BACKEND = os.getenv("STT_BACKEND", "whisper_local")

# Local Whisper model size: tiny | base | small | medium | large
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# API keys (used only when STT_BACKEND != whisper_local)
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── LLM Configuration ────────────────────────────────────────────────────────
# Options: "ollama" | "openai" | "groq"
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")

# OpenAI / Groq fallback model
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama3-8b-8192")

# ── TTS Configuration ────────────────────────────────────────────────────────
TTS_ENABLED = os.getenv("TTS_ENABLED", "true").lower() == "true"
# Options: "pyttsx3" | "gtts"
TTS_ENGINE  = os.getenv("TTS_ENGINE", "pyttsx3")

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = LOG_DIR / "agent.log"

# ── Safety ───────────────────────────────────────────────────────────────────
# All file writes are sandboxed to OUTPUT_DIR (enforced in tools layer)
ALLOWED_WRITE_ROOT = OUTPUT_DIR

print("MODEL:", OLLAMA_MODEL)