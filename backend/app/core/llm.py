from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel

from backend.app.core.config import settings
from backend.app.core.logging import logger

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """Unified LLM Factory returning live ChatGoogleGenerativeAI, ChatOpenAI, or fallback models."""

    # 1. Groq (fastest, free tier) - llama-3.3-70b-versatile
    if settings.GROQ_API_KEY and HAS_OPENAI:
        try:
            logger.debug("Instantiating Live Groq LLM via OpenAI compatible client")
            return ChatOpenAI(
                model="llama-3.3-70b-versatile",
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to instantiate Groq LLM: {e}")

    # 2. Google Gemini Flash
    if settings.GEMINI_API_KEY and HAS_GEMINI:
        target_model = model_name if model_name and "gemini" in model_name else "gemini-2.0-flash"
        try:
            logger.debug(f"Instantiating Live Gemini LLM: {target_model}")
            return ChatGoogleGenerativeAI(
                model=target_model,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to instantiate ChatGoogleGenerativeAI: {e}")

    # 3. OpenAI GPT-4o / GPT-4o-mini
    if settings.OPENAI_API_KEY and HAS_OPENAI:
        target_model = model_name if model_name and "gpt" in model_name else "gpt-4o-mini"
        try:
            logger.debug(f"Instantiating Live OpenAI LLM: {target_model}")
            return ChatOpenAI(
                model=target_model,
                api_key=settings.OPENAI_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to instantiate ChatOpenAI: {e}")

    # Fallback error if no live LLM keys are configured
    raise RuntimeError(
        "No active LLM provider found! Please ensure GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY is set in backend/.env"
    )
