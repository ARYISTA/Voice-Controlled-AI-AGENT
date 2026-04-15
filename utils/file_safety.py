"""
utils/file_safety.py — Sandbox enforcement for all file operations.

Rules:
  1. All writes MUST be inside OUTPUT_DIR (./output/).
  2. Paths are resolved to their absolute form before comparison.
  3. Path traversal attempts ("../../etc/passwd") are blocked.
"""

from pathlib import Path
from config import ALLOWED_WRITE_ROOT
from utils.logger import get_logger

logger = get_logger(__name__)


class SafePathError(Exception):
    """Raised when a requested path violates the sandbox policy."""


def safe_output_path(relative_name: str) -> Path:
    """
    Given a filename (or relative sub-path), return a fully-resolved Path
    that is guaranteed to sit inside ALLOWED_WRITE_ROOT.

    Raises SafePathError if the resolved path would escape the sandbox.
    """
    # Resolve against the allowed root
    target = (ALLOWED_WRITE_ROOT / relative_name).resolve()
    allowed = ALLOWED_WRITE_ROOT.resolve()

    try:
        # Python 3.9+: is_relative_to
        target.relative_to(allowed)
    except ValueError:
        logger.error(
            "Path traversal attempt blocked: %s → %s", relative_name, target
        )
        raise SafePathError(
            f"'{relative_name}' resolves outside the allowed output directory."
        )

    # Create parent directories as needed
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def is_safe_path(relative_name: str) -> bool:
    """Return True if the path is safe, False otherwise (no exception)."""
    try:
        safe_output_path(relative_name)
        return True
    except SafePathError:
        return False
