import json
import re
from typing import List
from backend.app.core.llm import get_llm
from backend.app.core.logging import logger, logger_timer


class QueryRewriter:
    """Query Rewriter Infrastructure Component for generating multi-angle search variations."""

    def rewrite_query(
        self, query: str, num_variations: int = 3, trace_id: str = "N/A"
    ) -> List[str]:
        """Expand user query into multi-perspective search variations using Live LLM (Gemini Flash / Groq)."""
        with logger_timer("QueryRewriter: Sub-query Expansion", trace_id=trace_id) as log:
            log.info(f"Expanding query: '{query}'")

            # 1. Try Live LLM Sub-Query Expansion
            try:
                llm = get_llm(temperature=0.2)
                prompt = (
                    f"Original Search Query: '{query}'\n\n"
                    "You are an Enterprise Search Optimization Agent. Generate UP TO 3 distinct search query variations "
                    "(only as many as necessary to cover distinct technical terms, max 3) to maximize retrieval recall in a corporate compliance database:\n"
                    "- Technical phrasing or synonyms\n"
                    "- Compliance policy acronyms & governance terms\n"
                    "- Specific rule definition terms\n\n"
                    "Return ONLY a valid JSON array of strings matching:\n"
                    '["variation 1", "variation 2"]'
                )

                response = llm.invoke(prompt)
                text = str(response.content).strip()

                if text.startswith("```json"):
                    text = text.replace("```json", "").replace("```", "").strip()
                elif text.startswith("```"):
                    text = text.replace("```", "").strip()

                variations = json.loads(text)
                if isinstance(variations, list) and len(variations) > 0:
                    # Always include original query at position 0 if missing
                    if query not in variations:
                        variations.insert(0, query)
                    
                    final_variations = variations[:num_variations]
                    log.info(f"Live LLM Query Expansion Generated {len(final_variations)} variations: {final_variations}")
                    return final_variations

            except Exception as e:
                log.warning(
                    f"[FALLBACK_TRIGGERED] Live LLM QueryRewriter call unavailable ({e}). Reverting to heuristic fallback expansion."
                )

            # 2. Heuristic Fallback
            variations = [query]
            words = [w for w in re.split(r"\W+", query) if len(w) > 3]

            if len(words) >= 2:
                variations.append(" ".join(words[:4]))
                variations.append(f"{query} compliance requirements rules policy")

            unique_variations = list(dict.fromkeys(variations))[:num_variations]
            log.info(f"Fallback Query Expansion Generated {len(unique_variations)} variations: {unique_variations}")
            return unique_variations


# Global instance
query_rewriter = QueryRewriter()

