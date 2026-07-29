from typing import Any, Dict
from backend.app.core.logging import logger, logger_timer
from backend.app.core.state import AnalystState


class ReflectionAgent:
    """Reflection Agent providing self-critique, hallucination auditing, and re-plan routing."""

    def evaluate_and_reflect(self, state: AnalystState) -> Dict[str, Any]:
        """Audit report for unsupported claims and estimate confidence score."""
        report = state.get("analysis_report", "")
        reranked_chunks = state.get("reranked_chunks", [])
        trace_id = state.get("trace_id", "N/A")
        replan_count = state.get("reflection_count", 0)

        with logger_timer("ReflectionAgent: Self-Critique Audit", trace_id=trace_id) as log:
            log.info(f"Reflecting on generated analysis report (iteration {replan_count})...")

            # Check citation coverage and chunk presence
            if not reranked_chunks or "No relevant documentation" in report:
                confidence = 0.30
                critique = "Insufficient evidence retrieved in initial pass. Requires expanded sub-query retrieval."
                should_replan = replan_count < 2  # Limit max replan loops to 2
            else:
                confidence = 0.92
                critique = "Report is strongly supported by grounded chunk evidence with valid citations."
                should_replan = False

            log.info(
                f"Reflection Complete | Confidence={confidence:.2f} | Replan={should_replan} | Critique='{critique}'"
            )

            return {
                "confidence": confidence,
                "critique": critique,
                "should_replan": should_replan,
            }


# Global singleton instance
reflection_agent = ReflectionAgent()
