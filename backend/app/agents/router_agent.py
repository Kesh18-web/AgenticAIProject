import json
from typing import Any, Dict
from backend.app.core.config import settings
from backend.app.core.llm import get_llm
from backend.app.core.logging import logger, logger_timer
from backend.app.core.state import AnalystState


class ModelRouterAgent:
    """Model Router Agent allocating the optimal LLM based on task complexity, cost, and latency."""

    def select_model(self, task_type: str, state: AnalystState) -> str:
        """Route to optimal model (Gemini Flash, Groq, Gemini Pro, GPT-4o) using Live LLM complexity rating."""
        trace_id = state.get("trace_id", "N/A")
        query = state.get("user_query", "")

        with logger_timer("ModelRouterAgent: Route Allocation", trace_id=trace_id) as log:
            # 1. Try Live LLM Complexity Classification
            try:
                llm = get_llm(temperature=0.0)
                prompt = (
                    f"User Query: '{query}'\n"
                    f"Target Task Type: '{task_type}'\n\n"
                    "You are an AI System Model Router. Rate query complexity from 1 to 10 and allocate a model tier.\n"
                    "Tier Guidelines:\n"
                    "- FAST_TIER (complexity 1-6): For simple policy lookups, keyword extractions, fast tasks -> gemini-1.5-flash or groq/llama-3-70b\n"
                    "- REASONING_TIER (complexity 7-10): For multi-document synthesis, complex governance, solution architecture -> gemini-1.5-pro or gpt-4o\n\n"
                    "Respond ONLY with a JSON object:\n"
                    '{"complexity_score": 4, "recommended_tier": "FAST_TIER", "selected_model": "gemini-1.5-flash", "reason": "Simple lookup"}'
                )

                response = llm.invoke(prompt)
                text = str(response.content).strip()

                if text.startswith("```json"):
                    text = text.replace("```json", "").replace("```", "").strip()
                elif text.startswith("```"):
                    text = text.replace("```", "").strip()

                data = json.loads(text)
                selected = data.get("selected_model", "gemini-1.5-flash")
                log.info(
                    f"Live LLM Router Allocation -> Model: [{selected}] | Tier: {data.get('recommended_tier')} | Complexity: {data.get('complexity_score')}/10"
                )
                return selected

            except Exception as e:
                # Explicit Fallback Warning
                log.warning(
                    f"[FALLBACK_TRIGGERED] Live LLM Router call unavailable ({e}). Reverting to rule-based fallback routing."
                )

                # Rule-based fallback matrix
                if task_type in ["query_rewrite", "guardrail", "routing"]:
                    if settings.GROQ_API_KEY:
                        selected = "groq/llama-3-70b"
                    elif settings.GEMINI_API_KEY:
                        selected = "gemini-1.5-flash"
                    else:
                        selected = "mock-fast-model"
                elif task_type in ["analysis", "reflection", "solution_architect"]:
                    if settings.OPENAI_API_KEY:
                        selected = "gpt-4o"
                    elif settings.GEMINI_API_KEY:
                        selected = "gemini-1.5-pro"
                    else:
                        selected = "mock-reasoning-model"
                else:
                    selected = "gemini-1.5-flash"

                log.info(f"Fallback Route Allocated -> [{selected}]")
                return selected


# Global singleton instance
model_router_agent = ModelRouterAgent()

