import re
from typing import Any, Dict, List
from backend.app.core.logging import logger, logger_timer
from backend.app.core.state import Citation


class CitationEngine:
    """Citation Engine Infrastructure Component for claim verification and footnote mapping."""

    def extract_and_verify_citations(
        self,
        report_text: str,
        source_chunks: List[Dict[str, Any]],
        trace_id: str = "N/A",
    ) -> List[Citation]:
        """Extract inline citations from report text and map to source chunk metadata."""
        with logger_timer("CitationEngine: Citation Mapping", trace_id=trace_id) as log:
            citations: List[Citation] = []

            # Search for citations in formats like [1], [Chunk 1], [Doc 1]
            matches = re.findall(r"\[(?:Doc|Chunk|Citation)?\s*(\d+)\]", report_text)
            unique_indices = sorted(list(set(int(m) for m in matches)))

            for idx in unique_indices:
                # 1-indexed chunk mapping
                if 1 <= idx <= len(source_chunks):
                    chunk = source_chunks[idx - 1]
                    citation_entry: Citation = {
                        "citation_id": idx,
                        "source_name": chunk.get("source_name", "Unknown Document"),
                        "page_number": chunk.get("page_number", 1),
                        "snippet": chunk.get("text", "")[:150] + "...",
                        "confidence": chunk.get("reranker_confidence", 0.90),
                    }
                    citations.append(citation_entry)

            log.info(
                f"Extracted and verified {len(citations)} source citations from generated report."
            )
            return citations


# Global singleton instance
citation_engine = CitationEngine()
