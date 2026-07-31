from typing import Any, Dict, Optional
from backend.app.core.logging import logger

# Model Pricing per 1 Million Tokens (USD)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-1.5-flash": {"prompt_per_1m": 0.075, "completion_per_1m": 0.30},
    "groq-llama-3-70b": {"prompt_per_1m": 0.59, "completion_per_1m": 0.79},
    "gpt-4o": {"prompt_per_1m": 2.50, "completion_per_1m": 10.00},
    "claude-3-5-sonnet": {"prompt_per_1m": 3.00, "completion_per_1m": 15.00},
    "mock-reasoning-model": {"prompt_per_1m": 0.00, "completion_per_1m": 0.00},
}


class TelemetryEngine:
    """Telemetry & Financial Observability Engine tracking Tokens, Millisecond Latencies, and USD Cost ($)."""

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count based on standard 4 characters per token heuristic."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def calculate_telemetry(
        self,
        model_name: str,
        prompt_text: str,
        completion_text: str,
        node_latencies: Dict[str, float],
        is_cache_hit: bool = False,
    ) -> Dict[str, Any]:
        """Compute token counts, dollar cost ($), and per-node latencies for an execution run."""
        prompt_tokens = self.estimate_tokens(prompt_text)
        completion_tokens = self.estimate_tokens(completion_text)
        total_tokens = prompt_tokens + completion_tokens

        # Zero cost if request was served by Cache
        if is_cache_hit:
            total_cost_usd = 0.0
            prompt_cost = 0.0
            completion_cost = 0.0
        else:
            rates = MODEL_PRICING.get(
                model_name.lower(),
                {"prompt_per_1m": 0.10, "completion_per_1m": 0.40},
            )
            prompt_cost = (prompt_tokens / 1_000_000) * rates["prompt_per_1m"]
            completion_cost = (completion_tokens / 1_000_000) * rates["completion_per_1m"]
            total_cost_usd = round(prompt_cost + completion_cost, 6)

        total_latency_ms = round(sum(node_latencies.values()), 2)

        telemetry_result = {
            "model_allocated": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd,
            "formatted_cost": f"${total_cost_usd:.6f}" if total_cost_usd > 0 else "$0.000000 (Cached)",
            "node_latencies": node_latencies,
            "total_latency_ms": total_latency_ms,
            "is_cache_hit": is_cache_hit,
        }

        logger.info(
            f"[Telemetry] Model: {model_name} | Tokens: {total_tokens} | Cost: ${total_cost_usd:.6f} | Latency: {total_latency_ms}ms"
        )
        return telemetry_result


# Singleton Telemetry Engine Instance
telemetry_engine = TelemetryEngine()
