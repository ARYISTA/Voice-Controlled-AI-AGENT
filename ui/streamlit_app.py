"""
ui/streamlit_app.py — Streamlit UI for the Voice-Controlled AI Agent.

Design: Industrial-terminal aesthetic with amber-on-dark palette.
Features:
  - Microphone recording (via st-audiorec)
  - Audio file upload
  - Text fallback input
  - Live pipeline status
  - Confidence score visualisation
  - Collapsible history panel
  - TTS audio playback
"""

import os
import sys
import time
from pathlib import Path
import html


# Ensure project root on sys.path when run via `streamlit run ui/streamlit_app.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from config import LLM_BACKEND, STT_BACKEND, TTS_ENABLED, OUTPUT_DIR

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #0d0d0f;
    color: #e8e0d0;
}

/* ── Header ── */
.agent-header {
    background: linear-gradient(135deg, #1a1a1f 0%, #0f1117 100%);
    border: 1px solid #f5a623;
    border-radius: 4px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.agent-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #f5a623, #ff6b35, #f5a623);
}
.agent-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    color: #f5a623;
    margin: 0;
    letter-spacing: -0.03em;
}
.agent-header p {
    color: #8a8070;
    margin: 0.25rem 0 0 0;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Cards ── */
.pipeline-card {
    background: #12121a;
    border: 1px solid #2a2a35;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.pipeline-card h4 {
    font-family: 'Syne', sans-serif;
    color: #f5a623;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 0 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.pipeline-card .value {
    font-size: 1rem;
    color: #e8e0d0;
    line-height: 1.5;
}

/* ── Intent Badge ── */
.intent-badge {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 2px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.intent-write_code     { background: #1a3a1a; color: #4dff91; border: 1px solid #2a5a2a; }
.intent-create_file    { background: #1a2a3a; color: #4db8ff; border: 1px solid #2a4a6a; }
.intent-summarize_text { background: #3a1a3a; color: #ff4dff; border: 1px solid #5a2a5a; }
.intent-general_chat   { background: #2a2a1a; color: #f5a623; border: 1px solid #4a4a2a; }

/* ── Confidence Bar ── */
.conf-bar-container {
    background: #1a1a22;
    border-radius: 2px;
    height: 8px;
    width: 100%;
    margin: 0.5rem 0;
    overflow: hidden;
    border: 1px solid #2a2a35;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}

/* ── Output Box ── */
.output-box {
    background: #0a0a10;
    border: 1px solid #2a2a35;
    border-left: 3px solid #f5a623;
    border-radius: 2px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    line-height: 1.7;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
    color: #c8c0b0;
}

/* ── History Item ── */
.history-item {
    border-bottom: 1px solid #1a1a22;
    padding: 0.6rem 0;
    font-size: 0.78rem;
    display: grid;
    grid-template-columns: 60px 1fr auto;
    gap: 0.5rem;
    align-items: start;
}
.history-time  { color: #4a4a5a; font-size: 0.7rem; }
.history-text  { color: #a8a090; }
.history-ok    { color: #4dff91; }
.history-fail  { color: #ff4d4d; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0a10 !important;
    border-right: 1px solid #1a1a22 !important;
}
[data-testid="stSidebar"] label {
    color: #8a8070 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #f5a623 !important;
    color: #0d0d0f !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #ffc050 !important;
    transform: translateY(-1px) !important;
}

/* ── Streamlit Overrides ── */
.stTextArea textarea, .stTextInput input {
    background: #12121a !important;
    border: 1px solid #2a2a35 !important;
    color: #e8e0d0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 2px !important;
}
.stSelectbox > div > div {
    background: #12121a !important;
    border: 1px solid #2a2a35 !important;
    color: #e8e0d0 !important;
}
.stFileUploader {
    border: 1px dashed #3a3a45 !important;
    border-radius: 4px !important;
    background: #0f0f17 !important;
}
div[data-testid="stStatusWidget"] { display: none; }
.stSpinner > div { border-top-color: #f5a623 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0f; }
::-webkit-scrollbar-thumb { background: #2a2a35; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def confidence_bar_html(confidence: float) -> str:
    pct = int(confidence * 100)
    if pct >= 75:
        color = "#4dff91"
    elif pct >= 50:
        color = "#f5a623"
    else:
        color = "#ff6b35"
    return f"""
    <div style="margin:0.3rem 0">
        <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#8a8070;margin-bottom:3px">
            <span>CONFIDENCE</span><span>{pct}%</span>
        </div>
        <div class="conf-bar-container">
            <div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div>
        </div>
    </div>"""


def intent_badge_html(intent: str) -> str:
    cls = f"intent-{intent.replace(' ', '_')}"
    label = intent.replace("_", " ").upper()
    return f'<span class="intent-badge {cls}">{label}</span>'


def get_agent():
    """Cache the VoiceAgent in Streamlit session state (init once per session)."""
    if "agent" not in st.session_state:
        with st.spinner("⚙️ Initialising AI Agent…"):
            from agent import VoiceAgent
            st.session_state.agent = VoiceAgent()
    return st.session_state.agent


def run_pipeline(agent, audio_path: str = None, text: str = None):
    """
    Execute the agent pipeline and render results.
    Accepts either an audio path OR a direct text input.
    """
    from agent import AgentResponse

    with st.spinner("🔄 Processing pipeline…"):
        t0 = time.perf_counter()

        if audio_path:
            response: AgentResponse = agent.process_audio_file(audio_path)
        elif text:
            response: AgentResponse = agent.process_text(text)
        else:
            st.error("No input provided.")
            return

    # ── Pipeline Output Cards ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="pipeline-card"><h4>📡 Pipeline Complete</h4></div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ── STT Result ────────────────────────────────────────────────────────────
    with col1:
        st.markdown(f"""
        <div class="pipeline-card">
            <h4>🎙️ Transcription</h4>
            <div class="value">{response.transcription.text or "—"}</div>
        </div>""", unsafe_allow_html=True)

    # ── Intent ────────────────────────────────────────────────────────────────
    with col2:
        intent_value = str(response.intent_result.intent)
        if "." in intent_value:
            intent_value = intent_value.split(".")[-1]
            badge = intent_badge_html(intent_value.lower())
            bar = confidence_bar_html(response.intent_result.confidence)
            
            st.markdown(f"""
                        <div class="pipeline-card">
                        <h4>🧠 Intent &amp; Entities</h4>
                        {badge} {bar}
                        </div>""", unsafe_allow_html=True)

    # ── Action Taken ──────────────────────────────────────────────────────────
    status_icon = "✅" if response.tool_result.success else "❌"
    action_color = "#4dff91" if response.tool_result.success else "#ff4d4d"
    st.markdown(f"""
    <div class="pipeline-card">
        <h4>⚡ Action Taken</h4>
        <div class="value" style="color:{action_color}">
            {status_icon} {response.tool_result.action_taken or response.tool_result.error}
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Output ────────────────────────────────────────────────────────────────
    if response.tool_result.output:
        safe_output = html.escape(str(response.tool_result.output))
        st.markdown(f"""
                    <div class="pipeline-card">
                    <h4>📄 Output</h4>
                    <div class="output-box">{safe_output}</div>
                    </div>""", unsafe_allow_html=True)

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = response.tool_result.metadata
    if meta:
        meta_items = "  |  ".join(f"{k}: `{v}`" for k, v in meta.items())
        st.caption(f"ℹ️ {meta_items}  |  ⏱ {response.total_elapsed:.2f}s")

    # ── TTS Audio Playback ────────────────────────────────────────────────────
    if response.tts_audio_path and Path(response.tts_audio_path).exists():
        with open(response.tts_audio_path, "rb") as f:
            st.audio(f.read(), format="audio/mp3")

    # ── Error ─────────────────────────────────────────────────────────────────
    if response.tool_result.error and not response.tool_result.success:
        st.error(f"⚠️ {response.tool_result.error}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(agent):
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        st.caption(f"**STT Backend:** `{STT_BACKEND}`")
        st.caption(f"**LLM Backend:** `{LLM_BACKEND}`")
        st.caption(f"**TTS Enabled:** `{TTS_ENABLED}`")
        st.caption(f"**Output Dir:** `{OUTPUT_DIR}`")

        st.markdown("---")

        # ── Output Files ──────────────────────────────────────────────────────
        st.markdown("### 📁 Output Files")
        output_files = list(OUTPUT_DIR.glob("**/*"))
        output_files = [f for f in output_files if f.is_file()]

        if output_files:
            for f in sorted(output_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                size = f.stat().st_size
                label = f"{f.name} ({size} B)"
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                st.download_button(
                    label=f"⬇ {f.name}",
                    data=content,
                    file_name=f.name,
                    mime="text/plain",
                    key=f"dl_{f.name}_{f.stat().st_mtime}",
                )
        else:
            st.caption("_No files generated yet._")

        st.markdown("---")

        # ── History Panel ─────────────────────────────────────────────────────
        st.markdown("### 📜 Session History")
        turns = agent.history.all()

        if turns:
            for turn in reversed(turns[-8:]):
                ok_icon = "✅" if turn.success else "❌"
                intent_short = turn.intent.replace("_", " ")
                st.markdown(
                    f"`{turn.timestamp_str}` {ok_icon} **{intent_short}**  \n"
                    f"<span style='color:#8a8070;font-size:0.75rem'>"
                    f"{turn.raw_text[:60]}{'…' if len(turn.raw_text)>60 else ''}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
            if st.button("🗑 Clear History"):
                agent.history.clear()
                st.rerun()
        else:
            st.caption("_No interactions yet._")


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="agent-header">
        <h1>🎙️ Voice AI Agent</h1>
        <p>Local · Modular · Production-Ready</p>
    </div>
    """, unsafe_allow_html=True)

    # Init agent
    agent = get_agent()
    render_sidebar(agent)

    # ── Input Tabs ────────────────────────────────────────────────────────────
    tab_upload, tab_mic, tab_text = st.tabs(
        ["📂 Upload Audio", "🎙️ Microphone", "⌨️ Text Input"]
    )

    # ── Tab 1: File Upload ────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("Upload a `.wav` or `.mp3` file to transcribe and process.")
        uploaded = st.file_uploader(
            "Choose audio file",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.audio(uploaded)
            if st.button("🚀 Process Audio", key="btn_upload"):
                import tempfile
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                run_pipeline(agent, audio_path=tmp_path)

    # ── Tab 2: Microphone ─────────────────────────────────────────────────────
    with tab_mic:
        st.markdown(
            "Record directly from your microphone. "
            "*(Requires `st-audiorec`: `pip install streamlit-audiorec`)*"
        )
        try:
            from st_audiorec import st_audiorec
            wav_audio_data = st_audiorec()
            if wav_audio_data is not None:
                if st.button("🚀 Process Recording", key="btn_mic"):
                    run_pipeline(agent, audio_path=None)
                    # Write bytes to temp file
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(wav_audio_data)
                        tmp_path = tmp.name
                    run_pipeline(agent, audio_path=tmp_path)
        except ImportError:
            st.warning(
                "**st-audiorec not installed.** "
                "Run `pip install streamlit-audiorec` to enable live microphone input.\n\n"
                "Use the **Upload Audio** or **Text Input** tab for now."
            )
            st.code("pip install streamlit-audiorec", language="bash")

    # ── Tab 3: Text Input ─────────────────────────────────────────────────────
    with tab_text:
        st.markdown(
            "Type your command directly. Great for testing without a microphone."
        )
        example_commands = [
            "— pick an example —",
            "Create a Python file with a retry decorator function",
            "Write a JavaScript async fetch wrapper and save it",
            "Summarize this: Artificial intelligence is the simulation of human intelligence...",
            "What is the difference between supervised and unsupervised learning?",
            "Create a file called notes.txt with today's meeting agenda",
        ]
        selected = st.selectbox("Try an example:", example_commands, key="example_select")
        user_text = st.text_area(
            "Your command:",
            value="" if selected.startswith("—") else selected,
            height=100,
            placeholder="e.g. Create a Python file with a retry function…",
            key="text_input",
        )
        if st.button("🚀 Run Command", key="btn_text") and user_text.strip():
            run_pipeline(agent, text=user_text.strip())

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#3a3a4a;font-size:0.7rem;"
        "letter-spacing:0.1em;text-transform:uppercase'>"
        "Voice AI Agent · All file ops sandboxed to ./output/ · "
        f"Session turns: {len(agent.history)}"
        "</div>",
        unsafe_allow_html=True,
    )
