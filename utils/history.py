"""
utils/history.py — In-session interaction history tracker.

Stores a list of AgentTurn objects in memory (and optionally to a JSONL log
file) so the Streamlit UI can display a history panel.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from config import LOG_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

HISTORY_FILE = LOG_DIR / "history.jsonl"


@dataclass
class AgentTurn:
    """One complete user→agent interaction."""
    timestamp: float        = field(default_factory=time.time)
    raw_text: str           = ""
    intent: str             = ""
    confidence: float       = 0.0
    entities: dict          = field(default_factory=dict)
    action_taken: str       = ""
    output: str             = ""
    success: bool           = True
    error: Optional[str]    = None
    stt_backend: str        = ""
    llm_backend: str        = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def timestamp_str(self) -> str:
        import datetime
        return datetime.datetime.fromtimestamp(self.timestamp).strftime(
            "%H:%M:%S"
        )


class SessionHistory:
    """
    Maintains the list of AgentTurns for the current session.
    Thread-safe for Streamlit's single-threaded model.
    """

    def __init__(self):
        self._turns: List[AgentTurn] = []

    def add(self, turn: AgentTurn) -> None:
        self._turns.append(turn)
        self._append_to_file(turn)
        logger.debug("History: added turn #%d", len(self._turns))

    def all(self) -> List[AgentTurn]:
        return list(self._turns)

    def recent(self, n: int = 10) -> List[AgentTurn]:
        return self._turns[-n:]

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def _append_to_file(self, turn: AgentTurn) -> None:
        """Persist turn to JSONL file for post-session analysis."""
        try:
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(turn.to_dict()) + "\n")
        except OSError as exc:
            logger.warning("Could not write history file: %s", exc)
