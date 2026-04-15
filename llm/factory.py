"""
llm/factory.py — LLM backend factory.
"""

from config import LLM_BACKEND
from llm.base import BaseLLM
from utils.logger import get_logger

logger = get_logger(__name__)


def get_llm_backend(backend: str = LLM_BACKEND) -> BaseLLM:
    """
    Return an instantiated LLM backend.

    Parameters
    ----------
    backend : str
        One of: "ollama" | "groq" | "openai"
    """
    backend = backend.lower().strip()
    logger.info("Initialising LLM backend: %s", backend)

    if backend == "ollama":
        from llm.ollama_llm import OllamaLLM
        return OllamaLLM()

    elif backend == "groq":
        from llm.groq_llm import GroqLLM
        return GroqLLM()

    elif backend == "openai":
        # Extend here if needed
        raise NotImplementedError(
            "OpenAI LLM backend not yet implemented. Use ollama or groq."
        )

    else:
        logger.warning(
            "Unknown LLM backend '%s', falling back to ollama.", backend
        )
        from llm.ollama_llm import OllamaLLM
        return OllamaLLM()
