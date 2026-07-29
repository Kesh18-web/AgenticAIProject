from typing import Any, Dict, List
from backend.app.core.logging import logger, logger_timer


class ContextBuilder:
    """Context Builder & Compression Infrastructure Component."""

    def build_context(
        self,
        chunks: List[Dict[str, Any]],
        max_characters: int = 12000,
        trace_id: str = "N/A",
    ) -> str:
        """Format reranked chunks into structured, citation-friendly context blocks with relevance confidence metrics."""
        if not chunks:
            return "No relevant background context found."

        with logger_timer("ContextBuilder: Constructing Context", trace_id=trace_id) as log:
            context_blocks = []
            total_chars = 0

            for idx, chunk in enumerate(chunks, start=1):
                doc_name = chunk.get("source_name", "Unknown Document")
                page_num = chunk.get("page_number", 1)
                text = chunk.get("text", "").strip()
                conf = chunk.get("reranker_confidence")

                conf_str = f" | Relevance Confidence: {int(conf * 100)}%" if conf is not None else ""
                header = f"--- DOCUMENT CHUNK [{idx}] | Source: {doc_name} | Page: {page_num}{conf_str} ---"
                block = f"{header}\n{text}\n"

                # Check character window limit to prevent prompt token overflow
                if total_chars + len(block) > max_characters:
                    log.warning(f"Context Window Cap reached ({max_characters} chars). Truncating remaining chunks.")
                    break

                context_blocks.append(block)
                total_chars += len(block)

            full_context = "\n".join(context_blocks)
            log.info(
                f"Constructed context window with {len(context_blocks)} chunks ({len(full_context)} characters)"
            )
            return full_context


# Global singleton instance
context_builder = ContextBuilder()
