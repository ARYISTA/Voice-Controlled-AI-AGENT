"""
Voice-Controlled Local AI Agent
================================
Entry point: launches the Streamlit UI.

Usage:
    streamlit run app.py
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from ui.streamlit_app import main

if __name__ == "__main__":
    main()
