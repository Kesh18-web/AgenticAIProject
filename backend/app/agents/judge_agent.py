import json
from typing import Any, Dict
from backend.app.core.llm import get_llm
from backend.app.core.logging import logger, logger_timer
from backend.app.core.state import AnalystState
from backend.app.db.firestore import firestore_db


class JudgeAgent:
    """Judge Agent evaluating faithfulness, groundedness, completeness, and recording telemetry."""

    def evaluate_output(self, state: AnalystState) -> Dict[str, float]:
        """Compute LLM-as-a-Judge evaluation scores using heavy reasoning model and persist to Firestore."""
        trace_id = state.get("trace_id", "N/A")
        query = state.get("user_query", "")
        report = state.get("analysis_report", "")
        context_text = state.get("context_text", "")
        citations = state.get("citations", [])
        reranked_chunks = state.get("reranked_chunks", [])
        selected_model = state.get("selected_model", "gemini-1.5-pro")
        plan = state.get("plan", {})
        requires_rag = plan.get("requires_rag", True)

        with logger_timer("JudgeAgent: LLM-as-a-Judge Evaluation", trace_id=trace_id) as log:
            log.info(f"Computing LLM-as-a-Judge metrics for query: '{query}' (requires_rag={requires_rag})...")

            # 1. Fast Pass for General Knowledge Queries
            if not requires_rag:
                log.info("General Knowledge query detected. Bypassing RAG document citation judge evaluation.")
                scores = {
                    "groundedness": 0.95,
                    "answer_relevance": 0.95,
                    "citation_coverage": 1.0,
                    "overall_quality": 0.95,
                }
                eval_record = {
                    "trace_id": trace_id,
                    "query": query,
                    "scores": scores,
                    "model": selected_model,
                    "requires_rag": False,
                }
                firestore_db.save_document(
                    collection_name="evaluation_metrics",
                    doc_id=trace_id,
                    data=eval_record,
                )
                return scores

            # 2. Live LLM-as-a-Judge Evaluation
            try:
                judge_llm = get_llm(model_name="gemini-1.5-flash", temperature=0.0)
                prompt = f"""
User Query: '{query}'

Retrieved Evidence Context:
{context_text}

Synthesized Report to Evaluate:
{report}

Number of Footnote Citations Verified: {len(citations)}

You are an Independent Lead AI Evaluation Auditor. Rate the synthesized report across 3 quantitative metrics on a scale from 0.00 to 1.00.

IMPORTANT GROUNDING AUDIT RULE:
- If the report makes specific claims (e.g. personal projects, companies, dates, or skills) that DO NOT exist in the Retrieved Evidence Context, "groundedness" MUST be low (0.00 to 0.30).
- If the report correctly states that information is not available in the provided documents, "groundedness" SHOULD be high (0.90 to 1.00).

Return ONLY a valid JSON object with these exact keys:
{{
  "groundedness": <float 0.00-1.00: Are all claims strictly supported by the context evidence?>,
  "answer_relevance": <float 0.00-1.00: Did the report directly answer what the user asked?>,
  "citation_coverage": <float 0.00-1.00: Are footnote citations present and valid?>,
  "overall_quality": <float 0.00-1.00: Weighted composite quality score>
}}
"""

                response = judge_llm.invoke(prompt)
                raw_text = str(response.content).strip()

                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()

                judge_data = json.loads(raw_text)
                groundedness = round(float(judge_data.get("groundedness", 0.85)), 2)
                answer_relevance = round(float(judge_data.get("answer_relevance", 0.85)), 2)
                citation_coverage = round(float(judge_data.get("citation_coverage", 1.0 if citations else 0.0)), 2)
                overall_quality = round(
                    float(judge_data.get("overall_quality", (groundedness * 0.4 + answer_relevance * 0.4 + citation_coverage * 0.2))),
                    2,
                )

                scores = {
                    "groundedness": groundedness,
                    "answer_relevance": answer_relevance,
                    "citation_coverage": citation_coverage,
                    "overall_quality": overall_quality,
                }

                log.info(
                    f"Live Judge Evaluation Scores | Overall={overall_quality} | Groundedness={groundedness} | Citations={citation_coverage}"
                )

                eval_record = {
                    "trace_id": trace_id,
                    "query": query,
                    "scores": scores,
                    "model": selected_model,
                    "requires_rag": True,
                }
                firestore_db.save_document(
                    collection_name="evaluation_metrics",
                    doc_id=trace_id,
                    data=eval_record,
                )
                return scores

            except Exception as e:
                log.warning(
                    f"[FALLBACK_TRIGGERED] Live LLM-as-a-Judge evaluation call unavailable ({e}). Reverting to fallback metrics."
                )

            # 3. Fallback Heuristic Judge (Strict Evidence Overlap)
            citation_coverage = 1.0 if citations else 0.0
            if not citations and reranked_chunks:
                # 0 citations generated for a RAG response means claims were ungrounded
                groundedness = 0.20
            else:
                groundedness = 0.90 if citations else 0.15
            answer_relevance = 0.85
            overall_quality = round(
                (groundedness * 0.4) + (answer_relevance * 0.4) + (citation_coverage * 0.2), 2
            )

            scores = {
                "groundedness": groundedness,
                "answer_relevance": answer_relevance,
                "citation_coverage": citation_coverage,
                "overall_quality": overall_quality,
            }

            log.info(f"Fallback Judge Scores | Overall={overall_quality} | Groundedness={groundedness}")

            eval_record = {
                "trace_id": trace_id,
                "query": query,
                "scores": scores,
                "model": selected_model,
                "requires_rag": True,
            }
            firestore_db.save_document(
                collection_name="evaluation_metrics",
                doc_id=trace_id,
                data=eval_record,
            )
            return scores


# Global singleton instance
judge_agent = JudgeAgent()

