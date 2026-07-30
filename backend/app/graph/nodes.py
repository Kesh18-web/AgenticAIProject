import uuid
from typing import Any, Dict
from backend.app.agents.analysis_agent import analysis_agent
from backend.app.agents.guardrail_agent import guardrail_agent
from backend.app.agents.judge_agent import judge_agent
from backend.app.agents.planner_agent import planner_agent
from backend.app.agents.reflection_agent import reflection_agent
from backend.app.agents.router_agent import model_router_agent
from backend.app.core.logging import logger
from backend.app.core.state import AnalystState
from backend.app.db.bm25 import bm25_mgr
from backend.app.db.qdrant import qdrant_store
from backend.app.infrastructure.citation_engine import citation_engine
from backend.app.infrastructure.context_builder import context_builder
from backend.app.infrastructure.hybrid_retriever import HybridRetriever
from backend.app.infrastructure.query_rewriter import query_rewriter
from backend.app.infrastructure.reranker import cross_encoder_reranker

# Instantiate shared retriever
hybrid_retriever = HybridRetriever(qdrant_mgr=qdrant_store, bm25_mgr=bm25_mgr)


def guardrail_node(state: AnalystState) -> Dict[str, Any]:
    """LangGraph Node: Execute Guardrail Security Inspection."""
    trace_id = state.get("trace_id") or str(uuid.uuid4())[:8]
    status = guardrail_agent.check_input(state)
    return {"guardrail_status": status, "trace_id": trace_id}


def planner_node(state: AnalystState) -> Dict[str, Any]:
    """LangGraph Node: Deconstruct Query & Generate Retrieval Strategy."""
    plan = planner_agent.plan_analysis(state)
    return {"plan": plan}


def router_node(state: AnalystState) -> Dict[str, Any]:
    """LangGraph Node: Allocate Optimal LLM based on task requirement."""
    selected = model_router_agent.select_model("analysis", state)
    return {"selected_model": selected}


def retrieval_node(state: AnalystState) -> Dict[str, Any]:
    """LangGraph Node: Execute Sub-Task Research, Dynamic Weighted Hybrid RRF Retrieval, Reranking & Context Building."""
    query = state.get("user_query", "")
    trace_id = state.get("trace_id", "N/A")
    plan = state.get("plan", {})

    # Extract dynamic retrieval weights chosen by Planner Agent
    top_k = int(plan.get("top_k", 10))
    bm25_weight = float(plan.get("bm25_weight", 0.5))
    dense_weight = float(plan.get("dense_weight", 0.5))
    sub_tasks = plan.get("sub_tasks", [])
    requires_rag = plan.get("requires_rag", True)

    # Fast-path bypass for general knowledge queries that do not require document retrieval
    if not requires_rag:
        return {
            "rewritten_queries": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "context_text": "General Knowledge Query (RAG Search Bypassed).",
        }

    # 1. Build search targets: User Query + Sub-Tasks from Planner + Sub-Query Variations
    search_queries = [query]
    if sub_tasks and isinstance(sub_tasks, list):
        search_queries.extend(sub_tasks)

    # Expand original query into sub-query variations
    rewritten_variations = query_rewriter.rewrite_query(query, trace_id=trace_id)
    search_queries.extend(rewritten_variations)

    # Deduplicate search targets while preserving order
    unique_search_targets = list(dict.fromkeys(search_queries))

    # Determine search scope: Read user explicit selection directly from state ('session' default vs 'global')
    search_scope = state.get("search_scope", "session")
    session_filter = state.get("session_id") if search_scope == "session" else None

    # 2. Retrieve via Dynamic Weighted Hybrid RRF across ALL sub-task & query targets
    dummy_query_emb = [0.01 * (i + 1) for i in range(384)]
    all_retrieved_chunks = []
    seen_keys = set()

    for target_q in unique_search_targets:
        sub_hits = hybrid_retriever.retrieve_hybrid(
            collection_name="enterprise_documents",
            query=target_q,
            query_embedding=dummy_query_emb,
            top_k=top_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            session_id=session_filter,
            trace_id=trace_id,
        )
        for chunk in sub_hits:
            key = str(chunk.get("doc_id", "")) + "_" + str(chunk.get("text", "")[:50])
            if key not in seen_keys:
                seen_keys.add(key)
                all_retrieved_chunks.append(chunk)

    # 3. Rerank via Cross-Encoder (Dynamically scaled based on top_k)
    top_n = max(3, min(top_k, 7))
    reranked_chunks = cross_encoder_reranker.rerank(
        query=query, chunks=all_retrieved_chunks, top_n=top_n, trace_id=trace_id
    )

    # 4. Build Context String
    context_text = context_builder.build_context(reranked_chunks, trace_id=trace_id)

    return {
        "rewritten_queries": rewritten_variations,
        "retrieved_chunks": all_retrieved_chunks,
        "reranked_chunks": reranked_chunks,
        "context_text": context_text,
    }


def analysis_node(state: AnalystState) -> Dict[str, Any]:
    """LangGraph Node: Synthesize Evidence-Grounded Compliance Analysis & Verify Footnote Citations."""
    trace_id = state.get("trace_id", "N/A")
    reranked_chunks = state.get("reranked_chunks", [])

    raw_report = analysis_agent.generate_analysis(state)
    
    # 1. Output Guardrail Stage: Read-only Audit Inspection
    output_audit = guardrail_agent.audit_output(raw_report, trace_id=trace_id)
    
    # 2. Output Guardrail Stage: Redaction & Sanitization
    sanitized_report = guardrail_agent.sanitize_output(raw_report, trace_id=trace_id)
    
    citations = citation_engine.extract_and_verify_citations(
        report_text=sanitized_report, source_chunks=reranked_chunks, trace_id=trace_id
    )

    return {
        "analysis_report": sanitized_report,
        "citations": citations,
        "output_guardrail_audit": output_audit,
    }


def reflection_node(state: AnalystState) -> Dict[str, Any]:
    """LangGraph Node: Audit Report, Estimate Confidence & Determine Re-plan Route."""
    replan_count = state.get("reflection_count", 0)
    reflection_res = reflection_agent.evaluate_and_reflect(state)

    return {
        "reflection_confidence": reflection_res["confidence"],
        "reflection_critique": reflection_res["critique"],
        "reflection_count": replan_count + (1 if reflection_res["should_replan"] else 0),
    }


def judge_node(state: AnalystState) -> Dict[str, Any]:
    """LangGraph Node: Evaluate Groundedness Metrics & Persist to Firestore."""
    scores = judge_agent.evaluate_output(state)
    return {"judge_eval_scores": scores}
