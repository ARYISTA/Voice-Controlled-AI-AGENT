# 🎙️ Voice-Controlled Local AI Agent

A **production-ready**, modular voice agent that converts speech to text, understands intent, and executes local tools — all with a clean Streamlit UI.

```
Audio ──► STT ──► LLM Intent ──► Tool Dispatch ──► Output + TTS
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **STT** | Local Whisper (offline) or Groq API (cloud fallback) |
| **LLM** | Ollama (local, private) or Groq API |
| **Intents** | `write_code` · `create_file` · `summarize_text` · `general_chat` |
| **Tools** | Code generation · File creation · Summarisation · Chat |
| **TTS** | pyttsx3 (offline) or gTTS voice feedback |
| **UI** | Streamlit — mic recording, file upload, text input |
| **Safety** | All file writes sandboxed to `./output/` |
| **History** | In-session log panel + persistent JSONL file |

---

## 📁 Project Structure

```
voice_agent/
├── app.py                  # Entry point
├── agent.py                # Core pipeline orchestrator
├── config.py               # Central configuration
├── requirements.txt
├── .env.example
│
├── stt/                    # Speech-to-Text backends
│   ├── base.py             # Abstract interface + TranscriptionResult
│   ├── whisper_local.py    # Local OpenAI Whisper
│   ├── groq_stt.py         # Groq cloud Whisper
│   └── factory.py          # Backend selector
│
├── llm/                    # Language Model backends
│   ├── base.py             # Abstract interface + IntentResult + Intent enum
│   ├── ollama_llm.py       # Local Ollama (Llama 3, Mistral, etc.)
│   ├── groq_llm.py         # Groq cloud API
│   └── factory.py          # Backend selector
│
├── tools/                  # Agent tools
│   ├── base.py             # Abstract interface + ToolResult
│   ├── file_tool.py        # File creation (sandboxed)
│   ├── code_tool.py        # Code generation + save
│   ├── summarize_tool.py   # Text summarisation
│   ├── chat_tool.py        # Conversational response
│   └── registry.py         # Intent → Tool dispatcher
│
├── ui/
│   └── streamlit_app.py    # Streamlit UI
│
├── utils/
│   ├── logger.py           # Centralised logging
│   ├── audio_utils.py      # Audio conversion + mic recording
│   ├── file_safety.py      # Sandbox path enforcement
│   ├── tts.py              # Text-to-Speech
│   └── history.py          # Session history tracker
│
├── tests/
│   └── test_agent.py       # Unit + integration tests
│
├── output/                 # All generated files go here (git-ignored)
└── logs/                   # Log files (git-ignored)
```

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Python 3.9+
python --version

# ffmpeg (required by Whisper for audio conversion)
# Ubuntu/Debian:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg
```

### 2. Clone & Install

```bash
git clone <your-repo-url>
cd voice_agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your settings
```

**Minimum config for fully local setup (no API keys needed):**
```env
STT_BACKEND=whisper_local
WHISPER_MODEL_SIZE=base
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3
TTS_ENGINE=pyttsx3
```

### 4. Start Ollama (for local LLM)

```bash
# Install Ollama: https://ollama.com/download
ollama serve                    # Start Ollama server
ollama pull llama3              # Download Llama 3 model (~4 GB)
```

### 5. Launch the UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔧 Configuration Reference

| Variable | Default | Options |
|---|---|---|
| `STT_BACKEND` | `whisper_local` | `whisper_local`, `groq` |
| `WHISPER_MODEL_SIZE` | `base` | `tiny`, `base`, `small`, `medium`, `large` |
| `LLM_BACKEND` | `ollama` | `ollama`, `groq` |
| `OLLAMA_MODEL` | `llama3` | Any model pulled in Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `GROQ_API_KEY` | — | Your Groq API key |
| `TTS_ENABLED` | `true` | `true`, `false` |
| `TTS_ENGINE` | `pyttsx3` | `pyttsx3`, `gtts` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## 🗣️ Example Commands

| Voice / Text Input | Intent | What Happens |
|---|---|---|
| *"Create a Python file with a retry decorator"* | `write_code` | Generates `retry_decorator.py` in `./output/` |
| *"Write a JavaScript async fetch wrapper and save it"* | `write_code` | Generates `async_fetch_wrapper.js` |
| *"Create a file called notes.txt with meeting agenda"* | `create_file` | Creates `notes.txt` with LLM-drafted content |
| *"Summarise this: [long text]"* | `summarize_text` | Returns paragraph summary + bullets |
| *"What is the difference between TCP and UDP?"* | `general_chat` | Returns conversational explanation |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ -v --cov=. --cov-report=term-missing

# Run a specific test class
pytest tests/test_agent.py::TestFileSafety -v
```

**Test categories:**
- `TestFileSafety` — Path traversal prevention, sandbox enforcement
- `TestFileTool` — File creation, metadata, error cases
- `TestCodeTool` — Code generation, markdown stripping, filename inference
- `TestSummarizeTool` — Text summarisation, edge cases
- `TestChatTool` — Conversational response
- `TestRegistry` — Intent dispatch, unknown intent fallback
- `TestAgentIntegration` — Full pipeline with mocked STT + LLM
- `TestIntentParsing` — JSON parsing, fence stripping, invalid intents

---

## 🔌 Using Groq (Cloud Fallback — Free Tier)

1. Sign up at [console.groq.com](https://console.groq.com) (free)
2. Generate an API key
3. Update `.env`:
   ```env
   STT_BACKEND=groq
   LLM_BACKEND=groq
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   ```

Groq provides very fast inference on Llama 3 and Whisper models at no cost on the free tier.

---

## 🛡️ Security

- **All file writes are sandboxed** to `./output/` — enforced by `utils/file_safety.py`
- Path traversal attempts (`../../etc/passwd`) are detected and blocked
- API keys are never logged or exposed in the UI
- No external network calls unless cloud backends are explicitly configured

---

## ➕ Adding a New Tool

1. Create `tools/my_new_tool.py` inheriting from `BaseTool`
2. Implement `name` property and `run()` method
3. Add a new `Intent` value in `llm/base.py` if needed
4. Register in `tools/registry.py`:
   ```python
   TOOL_REGISTRY["my_new_intent"] = MyNewTool()
   ```

Done — no other changes needed.

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                         │
│    [Upload] [Microphone] [Text Input] [History Panel]    │
└───────────────────────┬─────────────────────────────────┘
                        │ audio bytes / text
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   VoiceAgent (agent.py)                  │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   STT    │───►│  LLM Intent  │───►│ Tool Registry │  │
│  │ (Whisper │    │  Classifier  │    │   Dispatch    │  │
│  │  /Groq)  │    │ (Ollama/Groq)│    │               │  │
│  └──────────┘    └──────────────┘    └───────┬───────┘  │
│                                              │           │
│                          ┌───────────────────┤           │
│                          │                   │           │
│                   ┌──────▼──┐ ┌──────────┐  │           │
│                   │FileTool │ │CodeTool  │  │           │
│                   │         │ │          │  │           │
│                   └─────────┘ └──────────┘  │           │
│                   ┌──────────┐ ┌──────────┐ │           │
│                   │Summarize │ │ChatTool  │ │           │
│                   │  Tool    │ │          │◄┘           │
│                   └──────────┘ └──────────┘             │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  SessionHistory  │  TTS  │  Logger  │  Safety    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                        │
                    ./output/
              (sandboxed file writes)
```

---

## 📝 License

MIT — use freely for learning, portfolio, or production projects.
