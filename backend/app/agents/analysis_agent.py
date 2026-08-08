from typing import Any, Dict
from backend.app.core.llm import get_llm, extract_text_content
from backend.app.core.logging import logger, logger_timer
from backend.app.core.state import AnalystState


class AnalysisAgent:
    """Analysis Agent responsible for multi-step reasoning, comparison, and grounded report generation."""

    def generate_analysis(self, state: AnalystState) -> str:
        """Synthesize response backed by selected primary_knowledge_source taxonomy using selected_model."""
        query = state.get("user_query", "")
        context_text = state.get("context_text", "")
        trace_id = state.get("trace_id", "N/A")
        selected_model = state.get("selected_model", "gemini-2.5-flash")
        replan_count = state.get("reflection_count", 0)
        critique = state.get("reflection_critique", "")
        plan = state.get("plan", {})
        source = state.get("primary_knowledge_source") or plan.get("primary_knowledge_source", "PARAMETRIC_LLM")

        with logger_timer("AnalysisAgent: Report Synthesis", trace_id=trace_id) as log:
            log.info(f"Synthesizing report using model [{selected_model}] (source={source}, replan_count={replan_count})...")

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

            critique_instruction = ""
            if replan_count > 0 and critique:
                critique_instruction = f"\nRE-PLANNING FEEDBACK: A previous draft received low confidence due to the following critique: '{critique}'. Ensure this revised report explicitly addresses these missing evidence gaps!\n"

            try:
                llm = get_llm(model_name=selected_model, temperature=0.2)

                # 1. PARAMETRIC_LLM: Direct, rich synthesis with ZERO grounding caveats
                if source == "PARAMETRIC_LLM":
                    prompt = (
                        f"{conversation_history_block}"
                        f"User Inquiry: {query}\n"
                        f"{memory_block}\n"
                        "You are an Enterprise AI Assistant. Provide a clear, comprehensive, accurate, detailed, and direct response to the user inquiry using your deep pre-trained knowledge."
                    )

                # 2. ENTERPRISE_RAG: Strict document evidence grounding
                elif source == "ENTERPRISE_RAG":
                    prompt = (
                        f"{conversation_history_block}"
                        f"User Query: {query}\n"
                        f"{memory_block}\n"
                        f"Retrieved Grounded Document Context:\n{context_text}\n"
                        f"{critique_instruction}\n"
                        "You are an Enterprise AI Lead Analyst. Use the conversation history above (if any) to recall prior context.\n"
                        "Mandatory Guidelines:\n"
                        "1. Ground your answer strictly in the provided document context chunks.\n"
                        "2. Use inline footnote citations like [Doc 1], [Doc 2] corresponding to chunk numbers.\n"
                        "3. STRICT GROUNDING RULE: If the retrieved document context does not contain the answer to the user's specific question, explicitly state what information is missing. Do NOT fabricate facts outside the evidence."
                    )

                # 3. REALTIME_WEB_MCP / GITHUB_MCP / FILESYSTEM_MCP: Web & Tool evidence synthesis
                else:
                    prompt = (
                        f"{conversation_history_block}"
                        f"User Query: {query}\n"
                        f"{memory_block}\n"
                        f"Live Retrieved Tool & Search Evidence:\n{context_text}\n"
                        f"{critique_instruction}\n"
                        "You are an Enterprise AI Lead Analyst.\n"
                        "Mandatory Guidelines:\n"
                        "1. Ground your answer in the provided live tool evidence and web content.\n"
                        "2. Use inline citations like [Web 1], [Web 2] for web evidence.\n"
                        "3. If the tool evidence is brief on a broad topic, supplement with your general pre-trained knowledge to deliver a complete, clear, and helpful response. Do NOT refuse to answer simply because a web snippet was short."
                    )

                response = llm.invoke(prompt)
                report_text = extract_text_content(response.content)
                log.info(f"Live LLM Synthesized report ({len(report_text)} characters using model [{selected_model}])")
                return report_text

            except Exception as e:
                log.error(f"AnalysisAgent LLM synthesis error: {e}")
                return f"### Response\n{query}\n\n(Synthesis error occurred: {str(e)})"


# Global singleton instance
analysis_agent = AnalysisAgent()
