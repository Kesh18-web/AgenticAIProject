from typing import Any, Dict
from backend.app.core.llm import get_llm
from backend.app.core.logging import logger, logger_timer
from backend.app.core.state import AnalystState


class AnalysisAgent:
    """Analysis Agent responsible for multi-step reasoning, comparison, and grounded report generation."""

    def generate_analysis(self, state: AnalystState) -> str:
        """Synthesize evidence-grounded compliance report backed by context chunks or live MCP results using selected_model."""
        query = state.get("user_query", "")
        context_text = state.get("context_text", "")
        trace_id = state.get("trace_id", "N/A")
        selected_model = state.get("selected_model", "gemini-1.5-pro")
        replan_count = state.get("reflection_count", 0)
        critique = state.get("reflection_critique", "")
        plan = state.get("plan", {})
        requires_rag = plan.get("requires_rag", True)

        with logger_timer("AnalysisAgent: Report Synthesis", trace_id=trace_id) as log:
            log.info(f"Synthesizing report using model [{selected_model}] (requires_rag={requires_rag}, replan_count={replan_count})...")

            # Build recent conversation history block from short-term working memory
            short_term_turns = state.get("short_term_turns", []) or []
            if short_term_turns:
                history_lines = []
                for turn in short_term_turns:
                    history_lines.append(f"User: {turn.get('user', '')}")
                    history_lines.append(f"Assistant: {turn.get('assistant', '')[:300]}")
                conversation_history_block = (
                    "\nRecent Conversation History (use this to recall prior context, user names, preferences, and established facts):\n"
                    + "\n".join(history_lines)
                    + "\n"
                )
            else:
                conversation_history_block = ""

            long_term_summary = state.get("long_term_summary", "")
            memory_block = f"\nLong-Term Conversation Memory Summary: {long_term_summary}\n" if long_term_summary else ""

            # Check if active context is present (from document retrieval or live MCP search execution)
            has_context = bool(context_text) and context_text != "General Knowledge Query (RAG Search Bypassed)."

            critique_instruction = ""
            if replan_count > 0 and critique:
                critique_instruction = f"\nRE-PLANNING FEEDBACK: A previous draft received low confidence due to the following critique: '{critique}'. Ensure this revised report explicitly addresses these missing evidence gaps!\n"

            try:
                llm = get_llm(model_name=selected_model, temperature=0.2)

                if has_context:
                    prompt = (
                        f"{conversation_history_block}"
                        f"User Query: {query}\n"
                        f"{memory_block}\n"
                        f"Retrieved Grounded Context Chunks & Live Evidence:\n{context_text}\n"
                        f"{critique_instruction}\n"
                        "You are an Enterprise AI Lead Analyst. Use the conversation history above (if any) to recall prior context such as the user's name, preferences, or established facts.\n"
                        "Mandatory Guidelines:\n"
                        "1. Ground your answer directly in the provided context chunks or live web evidence.\n"
                        "2. Use inline footnote citations like [Doc 1], [Doc 2] corresponding to chunk numbers if document chunks are present.\n"
                        "3. STRICT GROUNDING RULE: If the retrieved evidence does not contain the answer to the user's specific question, explicitly state what information is missing. Do NOT fabricate facts outside the evidence.\n"
                        "4. Format your response naturally: for direct questions, give a clear, direct answer without forcing unnecessary section headers. For comprehensive policy audits, use clean markdown headers."
                    )
                else:
                    prompt = (
                        f"{conversation_history_block}"
                        f"User Inquiry: {query}\n"
                        f"{memory_block}\n"
                        "You are an Enterprise AI Assistant. Use the conversation history above (if any) to recall prior context such as the user's name, preferences, or facts they shared. "
                        "Provide a clear, comprehensive, accurate, and direct response to the user inquiry."
                    )

                response = llm.invoke(prompt)
                report_text = str(response.content).strip()
                log.info(f"Live LLM Synthesized report ({len(report_text)} characters using model [{selected_model}])")
                return report_text

            except Exception as e:
                log.error(f"AnalysisAgent LLM synthesis error: {e}")
                return f"### Response\n{query}\n\n(Synthesis error occurred: {str(e)})"


# Global singleton instance
analysis_agent = AnalysisAgent()
