from typing import Literal
from langgraph.graph import END, START, StateGraph

from backend.app.core.logging import logger
from backend.app.core.state import AnalystState
from backend.app.graph.nodes import (
    analysis_node,
    guardrail_node,
    judge_node,
    planner_node,
    reflection_node,
    retrieval_node,
    router_node,
)


def route_guardrail(state: AnalystState) -> Literal["planner", "end"]:
    """Conditional Edge: Route to planner if safe, else terminate execution."""
    status = state.get("guardrail_status", {})
    if status.get("safe", True):
        return "planner"
    logger.warning("Routing graph to END due to Guardrail Security Block.")
    return "end"


def route_reflection(state: AnalystState) -> Literal["planner", "judge"]:
    """Conditional Edge: Route back to planner for re-planning if confidence is low."""
    confidence = state.get("reflection_confidence", 1.0)
    count = state.get("reflection_count", 0)

    if confidence < 0.60 and count < 2:
        logger.info(
            f"Reflection confidence low ({confidence:.2f}). Triggering RE-PLANNING loop (iteration {count})..."
        )
        return "planner"

    logger.info(
        f"Reflection passed (confidence={confidence:.2f}). Proceeding to Judge evaluation."
    )
    return "judge"


def create_analyst_graph():
    """Assemble Enterprise AI Analyst LangGraph State Machine Graph."""
    workflow = StateGraph(AnalystState)

    # 1. Add Nodes
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("router", router_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("judge", judge_node)

    # 2. Add Fixed & Conditional Edges
    workflow.add_edge(START, "guardrail")

    workflow.add_conditional_edges(
        "guardrail",
        route_guardrail,
        {
            "planner": "planner",
            "end": END,
        },
    )

    workflow.add_edge("planner", "router")
    workflow.add_edge("router", "retrieval")
    workflow.add_edge("retrieval", "analysis")
    workflow.add_edge("analysis", "reflection")

    workflow.add_conditional_edges(
        "reflection",
        route_reflection,
        {
            "planner": "planner",
            "judge": "judge",
        },
    )

    workflow.add_edge("judge", END)

    # Compile Graph
    analyst_app = workflow.compile()
    logger.info("Successfully compiled Enterprise AI Analyst StateGraph workflow!")
    return analyst_app


# Global compiled graph app instance
analyst_graph = create_analyst_graph()
