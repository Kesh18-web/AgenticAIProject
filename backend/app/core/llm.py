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
    """Unified LLM Factory returning live ChatGoogleGenerativeAI with seamless Groq fallback on Rate Limits."""

    groq_llm = None
    if settings.GROQ_API_KEY and HAS_OPENAI:
        try:
            groq_llm = ChatOpenAI(
                model="llama-3.3-70b-versatile",
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to instantiate Groq LLM: {e}")

    # 1. Google Gemini (Primary Choice)
    if settings.GEMINI_API_KEY and HAS_GEMINI:
        target_model = model_name if model_name and ("gemini" in model_name or "gpt" in model_name) else "gemini-2.0-flash"
        if target_model in ["gemini-1.5-flash", "gpt-4o-mini", "mock-reasoning-model"]:
            target_model = "gemini-2.0-flash"
        try:
            logger.debug(f"Instantiating Live Gemini LLM: {target_model}")
            gemini_llm = ChatGoogleGenerativeAI(
                model=target_model,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            # Attach Groq as auto-fallback if Gemini hits 429 rate limits or errors
            if groq_llm:
                return gemini_llm.with_fallbacks([groq_llm])
            return gemini_llm
        except Exception as e:
            logger.error(f"Failed to instantiate ChatGoogleGenerativeAI [{target_model}]: {e}")

    # 2. Return Groq directly if Gemini not configured
    if groq_llm:
        return groq_llm

    # 3. OpenAI GPT-4o / GPT-4o-mini fallback
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

    raise RuntimeError(
        "No active LLM provider found! Please ensure GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY is set in backend/.env"
    )
