from typing import Any, Dict, List, Optional
from backend.app.core.llm import get_llm
from backend.app.core.logging import logger, logger_timer


class DualMemoryManager:
    """Dual Memory Engine: Short-Term Working Memory + LLM-backed Long-Term Memory Compactor."""

    def __init__(self, short_term_window: int = 3):
        self.short_term_window = short_term_window
        # In-memory storage per session_id: stores raw turns & active long-term summary
        self._session_store: Dict[str, Dict[str, Any]] = {}

    def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self._session_store:
            self._session_store[session_id] = {
                "turns": [],  # List of {"user": str, "assistant": str}
                "long_term_summary": "",
                "compaction_count": 0,
            }
        return self._session_store[session_id]

    def add_turn(self, session_id: str, user_message: str, assistant_reply: str) -> None:
        """Append a new interaction turn to the session memory."""
        session = self._get_or_create_session(session_id)
        session["turns"].append(
            {"user": user_message.strip(), "assistant": assistant_reply.strip()}
        )
        logger.info(
            f"[DualMemory] Added interaction turn to session '{session_id}' (Total Turns: {len(session['turns'])})"
        )

    def get_compacted_context(
        self, session_id: str, trace_id: str = "N/A"
    ) -> Dict[str, Any]:
        """Fetch memory context. Compacts older turns into an Executive Summary if turns > short_term_window."""
        session = self._get_or_create_session(session_id)
        turns = session["turns"]
        total_turns = len(turns)

        if total_turns == 0:
            return {
                "long_term_summary": "",
                "short_term_turns": [],
                "memory_compacted": False,
                "total_turns": 0,
            }

        # Case 1: Turns fit inside Short-Term Working Window
        if total_turns <= self.short_term_window:
            return {
                "long_term_summary": session.get("long_term_summary", ""),
                "short_term_turns": turns,
                "memory_compacted": False,
                "total_turns": total_turns,
            }

        # Case 2: Turns exceed window -> Execute Long-Term Memory Compactor
        with logger_timer("DualMemoryManager: Long-Term Memory Compaction", trace_id=trace_id) as log:
            older_turns = turns[:-self.short_term_window]
            recent_turns = turns[-self.short_term_window:]

            # Compact older turns into a dense executive state summary
            existing_summary = session.get("long_term_summary", "")
            turns_text = "\n".join(
                [f"User: {t['user']}\nAssistant: {t['assistant'][:200]}..." for t in older_turns]
            )

            prompt = (
                f"Existing Executive Summary: {existing_summary if existing_summary else 'None'}\n\n"
                f"Older Conversation History to Compact:\n{turns_text}\n\n"
                "You are an AI Memory Compactor. Condense the older conversation history and existing summary into a tight, dense 2-3 sentence Executive Memory Summary.\n"
                "Extract key entities, user preferences, repositories mentioned, and established compliance facts.\n"
                "Return ONLY the updated Executive Memory Summary."
            )

            try:
                llm = get_llm(temperature=0.0)
                res = llm.invoke(prompt)
                updated_summary = res.content.strip()
            except Exception as e:
                log.warning(f"Live LLM Memory Compactor fallback: {e}")
                # Fallback rule-based compactor
                updated_summary = f"Previous Session Context ({len(older_turns)} older turns compacted): User discussed compliance requirements & repository rules."

            session["long_term_summary"] = updated_summary
            session["compaction_count"] += 1

            log.info(
                f"[DualMemory] Successfully compacted {len(older_turns)} older turns into {len(updated_summary)} chars summary (Compaction #{session['compaction_count']})"
            )

            return {
                "long_term_summary": updated_summary,
                "short_term_turns": recent_turns,
                "memory_compacted": True,
                "total_turns": total_turns,
                "compacted_turns_count": len(older_turns),
            }


# Global Dual Memory Singleton Manager
dual_memory_mgr = DualMemoryManager(short_term_window=3)
