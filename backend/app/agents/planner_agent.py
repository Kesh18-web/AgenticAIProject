import json
from typing import Any, Dict, List
from backend.app.core.llm import get_llm
from backend.app.core.logging import logger, logger_timer
from backend.app.core.state import AnalystState


class PlannerAgent:
    """Planner Agent responsible for intent analysis, sub-task decomposition, and dynamic retrieval strategy."""

    def plan_analysis(self, state: AnalystState) -> Dict[str, Any]:
        """Invoke Live LLM to generate execution plan, search weights, and sub-task breakdown."""
        query = state.get("user_query", "")
        trace_id = state.get("trace_id", "N/A")
        replan_count = state.get("reflection_count", 0)
        critique = state.get("reflection_critique", "")

        long_term_summary = state.get("long_term_summary", "")

        with logger_timer("PlannerAgent: Live LLM Task Decomposition", trace_id=trace_id) as log:
            log.info(f"Invoking Live LLM Planner for query: '{query}' (replan_count={replan_count})")

            # Construct system & user prompt for live LLM planning
            prompt = (
                f"User Inquiry: {query}\n"
                f"Long-Term Conversation Memory Summary: {long_term_summary if long_term_summary else 'None (New Conversation)'}\n"
                f"Re-plan Iteration: {replan_count}\n"
                f"Reflection Critique / Gap Notes: {critique if critique else 'None (Initial Pass)'}\n\n"
                "You are an Enterprise AI Lead Analyst. Deconstruct this inquiry into a structured JSON execution plan following these explicit rules:\n\n"
                "1. requires_rag:\n"
                "   - Set to true if inquiry requires internal company policies, compliance documents, SOC2 standards, or uploaded files.\n"
                "   - Set to false only for generic greetings or pure general knowledge.\n\n"
                "2. requires_mcp & mcp_tools:\n"
                "   - Set requires_mcp to true IF query requires web search, server file listing/reading, or GitHub repository inspection.\n"
                "   - Populate mcp_tools with items from ['browser_search', 'fs_list_files', 'fs_read_file', 'github_code_search', 'github_issues_search'].\n"
                "   - Extract github_repo string (e.g. 'facebook/react' or 'Kesh18-web/AgenticAIProject') IF user specified a repository name. Otherwise set github_repo to null.\n\n"
                "3. search_mode & weight tuning:\n"
                "   - 'exact_keyword': Choose when the query contains exact section numbers, rule codes, or error IDs. Heavily favor BM25 keyword search (set bm25_weight between 0.70 and 0.90, and dense_weight = 1.0 - bm25_weight).\n"
                "   - 'semantic_conceptual': Choose when the query asks for broad policy summaries or high-level concepts. Heavily favor Dense Vector search (set dense_weight between 0.70 and 0.90, and bm25_weight = 1.0 - dense_weight).\n"
                "   - 'hybrid_balanced': Choose for standard compliance inquiries requiring both exact terms and contextual meaning (set bm25_weight=0.50, dense_weight=0.50).\n"
                "   - Ensure bm25_weight + dense_weight = 1.0.\n\n"
                "4. sub_tasks & explainability_reason:\n"
                "   - Break down the inquiry into 2-4 logical sub-tasks.\n"
                "   - Generate a 1-sentence 'explainability_reason' stating clearly WHY this search mode, weights, and tools were selected (e.g., 'Favored BM25 (80%) because exact rule code SOC2 3.1 was detected.').\n\n"
                "Return ONLY a valid JSON object matching this schema:\n"
                "{\n"
                '  "requires_rag": true,\n'
                '  "requires_mcp": false,\n'
                '  "mcp_tools": [],\n'
                '  "github_repo": null,\n'
                '  "sub_tasks": ["Extract key terms", "Retrieve policy evidence", "Synthesize report"],\n'
                '  "search_mode": "hybrid_balanced",\n'
                '  "bm25_weight": 0.50,\n'
                '  "dense_weight": 0.50,\n'
                '  "top_k": 8,\n'
                '  "explainability_reason": "Balanced hybrid search selected to combine exact terms with semantic context."\n'
                "}"
            )

            try:
                llm = get_llm(temperature=0.0)
                response = llm.invoke(prompt)
                response_text = str(response.content).strip()

                # Clean markdown codeblocks if present
                if response_text.startswith("```json"):
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                elif response_text.startswith("```"):
                    response_text = response_text.replace("```", "").strip()

                plan = json.loads(response_text)
                log.info(f"Live LLM Plan Generated successfully ({len(plan.get('sub_tasks', []))} sub-tasks, mode={plan.get('search_mode')})")
                return plan

            except Exception as e:
                log.warning(f"Live LLM call failed or fallback triggered: {e}. Using deterministic plan.")
                # Graceful Heuristic Fallback
                requires_mcp = any(term in query.lower() for term in ["github", "slack", "jira", "browser"])
                sub_tasks = [
                    f"Extract core terms and policies related to: {query}",
                    "Retrieve grounded background evidence using hybrid BM25 + Qdrant dense search",
                    "Synthesize compliance report backed by verified source citations",
                ]
                if replan_count > 0:
                    sub_tasks.append(f"Targeted secondary retrieval for critique: {critique}")

                return {
                    "requires_rag": True,
                    "requires_mcp": requires_mcp,
                    "sub_tasks": sub_tasks,
                    "search_mode": "hybrid_balanced",
                    "bm25_weight": 0.5,
                    "dense_weight": 0.5,
                    "top_k": 8,
                }


# Global singleton instance
planner_agent = PlannerAgent()

