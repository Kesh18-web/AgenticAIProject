import json
import uuid
from typing import AsyncGenerator, Optional
from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.app.core.logging import logger
from backend.app.core.state import AnalystState
from backend.app.graph.analyst_graph import analyst_graph

router = APIRouter(prefix="/analyze", tags=["Analyze"])

class AnalyzeRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

async def stream_analysis_events(
    query: str, session_id: str
) -> AsyncGenerator[str, None]:
    """Async generator streaming LangGraph state machine node updates via SSE."""
    trace_id = f"trace-{str(uuid.uuid4())[:8]}"

    initial_state: AnalystState = {
        "user_query": query,
        "trace_id": trace_id,
        "session_id": session_id,
        "reflection_count": 0,
    }

    logger.info(
        f"Starting SSE Stream Analysis | session_id={session_id} | trace_id={trace_id}"
    )

    yield f"data: {json.dumps({'event': 'start', 'trace_id': trace_id, 'query': query})}\n\n"

    try:
        # Iterate over graph steps
        for event in analyst_graph.stream(initial_state):
            for node_name, node_state in event.items():
                node_log = {
                    "event": "node_complete",
                    "node": node_name,
                    "trace_id": trace_id,
                }

                if node_name == "guardrail":
                    node_log["safe"] = node_state.get("guardrail_status", {}).get(
                        "safe"
                    )
                elif node_name == "planner":
                    node_log["sub_tasks"] = node_state.get("plan", {}).get(
                        "sub_tasks", []
                    )
                elif node_name == "router":
                    node_log["selected_model"] = node_state.get("selected_model")
                elif node_name == "retrieval":
                    node_log["chunk_count"] = len(
                        node_state.get("reranked_chunks", [])
                    )
                elif node_name == "analysis":
                    node_log["report_snippet"] = node_state.get(
                        "analysis_report", ""
                    )[:100]
                elif node_name == "reflection":
                    node_log["confidence"] = node_state.get("reflection_confidence")
                    node_log["critique"] = node_state.get("reflection_critique")
                elif node_name == "judge":
                    node_log["eval_scores"] = node_state.get("judge_eval_scores")

                yield f"data: {json.dumps(node_log)}\n\n"

        # Final result event
        final_state = analyst_graph.invoke(initial_state)
        final_payload = {
            "event": "complete",
            "trace_id": trace_id,
            "report": final_state.get("analysis_report", ""),
            "citations": final_state.get("citations", []),
            "eval_scores": final_state.get("judge_eval_scores", {}),
            "guardrail_safe": final_state.get("guardrail_status", {}).get(
                "safe", True
            ),
        }
        yield f"data: {json.dumps(final_payload)}\n\n"

    except Exception as e:
        logger.error(f"Error streaming graph execution: {e}")
        yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"


@router.post("/stream")
async def analyze_stream(req: AnalyzeRequest):
    """Stream real-time agent execution events and analysis report using Server-Sent Events (SSE)."""
    session_id = req.session_id or f"session-{str(uuid.uuid4())[:8]}"
    return EventSourceResponse(stream_analysis_events(req.query, session_id))
