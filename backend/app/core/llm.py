"""
LLM Factory — Direct dispatch using verified active Google AI Studio models & content extraction utilities.
"""
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
    from langchain_openai import ChatOpenAI  # used for Groq (OpenAI-compatible)
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def extract_text_content(content: Any) -> str:
    """
    Safely extract plain text from an LLM response.content property.
    Handles plain strings as well as multi-part list-of-dicts (e.g. [{'type': 'text', 'text': '...'}, ...]).
    """
    if not content:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(item["text"])
                elif "text" in item:
                    text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        if text_parts:
            return "\n".join(text_parts).strip()
    return str(content).strip()


def _build_groq_llm(temperature: float = 0.0, max_tokens: Optional[int] = None) -> Optional[BaseChatModel]:
    """Helper to build Groq Llama 3.3 70B model if configured."""
    if settings.GROQ_API_KEY and HAS_OPENAI:
        try:
            return ChatOpenAI(
                model="llama-3.3-70b-versatile",
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=1,
            )
        except Exception as e:
            logger.error(f"[LLM Factory] Failed to build Groq LLM: {e}")
    return None


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """
    Unified LLM Factory with automatic rate-limit fallback.
    Uses verified active model: 'gemini-2.5-flash' (1,500 RPD free tier limit).
    """
    model = (model_name or "gemini-2.5-flash").lower()
    groq_fallback = _build_groq_llm(temperature=temperature, max_tokens=max_tokens)

    # ── 1. Groq Explicit Selection ──────────────────────────────────────────
    if "groq" in model or "llama" in model:
        if groq_fallback:
            logger.debug("[LLM Factory] Dispatching → Groq / llama-3.3-70b-versatile")
            return groq_fallback
        logger.warning("[LLM Factory] Groq requested but unavailable — falling back to Gemini Flash.")
        return _get_gemini_flash("gemini-2.5-flash", temperature, max_tokens)

    # ── 2. Gemini Pro Selection ──────────────────────────────────────────────
    if "pro" in model:
        if settings.GEMINI_API_KEY and HAS_GEMINI:
            logger.debug("[LLM Factory] Dispatching → gemini-2.5-pro")
            primary = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=settings.GEMINI_API_KEY,
                temperature=temperature,
                max_output_tokens=max_tokens,
                max_retries=0,
            )
            if groq_fallback:
                return primary.with_fallbacks([groq_fallback], exceptions_to_handle=(Exception,))
            return primary

    # ── 3. Gemini Flash (Default: gemini-2.5-flash) ─────────────────────────
    if settings.GEMINI_API_KEY and HAS_GEMINI:
        logger.debug("[LLM Factory] Dispatching → gemini-2.5-flash")
        primary = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=temperature,
            max_output_tokens=max_tokens,
            max_retries=0,
        )
        if groq_fallback:
            return primary.with_fallbacks([groq_fallback], exceptions_to_handle=(Exception,))
        return primary

    # If Gemini API key is missing, return Groq directly
    if groq_fallback:
        logger.warning("[LLM Factory] Gemini key missing — using Groq as primary.")
        return groq_fallback

    raise RuntimeError("No active LLM provider found! Please check GEMINI_API_KEY or GROQ_API_KEY in backend/.env")


def _get_gemini_flash(
    model_id: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """Internal helper: returns Gemini Flash model with robust Groq fallback."""
    groq_fallback = _build_groq_llm(temperature=temperature, max_tokens=max_tokens)
    if settings.GEMINI_API_KEY and HAS_GEMINI:
        primary = ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=temperature,
            max_output_tokens=max_tokens,
            max_retries=0,
        )
        if groq_fallback:
            return primary.with_fallbacks([groq_fallback], exceptions_to_handle=(Exception,))
        return primary

    if groq_fallback:
        return groq_fallback

    raise RuntimeError("No LLM provider available.")
