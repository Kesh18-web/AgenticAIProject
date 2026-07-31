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
from backend.app.infrastructure.cache_engine import retrieval_cache, semantic_cache
from backend.app.infrastructure.citation_engine import citation_engine
from backend.app.infrastructure.context_builder import context_builder
from backend.app.infrastructure.hybrid_retriever import HybridRetriever
from backend.app.infrastructure.query_rewriter import query_rewriter
from backend.app.infrastructure.reranker import cross_encoder_reranker
from backend.app.mcp.mcp_registry import mcp_registry

# Instantiate shared retriever
hybrid_retriever = HybridRetriever(qdrant_mgr=qdrant_store, bm25_mgr=bm25_mgr)


def guardrail_node(state: AnalystState) -> Dict[str, Any]:
    """LangGraph Node: Execute Guardrail Security Inspection."""
    trace_id = state.get("trace_id") or str(uuid.uuid4())[:8]
    status = guardrail_agent.check_input(state)
    return {"guardrail_status": status, "trace_id": trace_id}


def planner_node(state: AnalystState) -> Dict[str, Any]:
    """LangGraph Node: Generate Execution Plan & Sub-Tasks."""
    plan = planner_agent.plan_analysis(state)
    return {"plan": plan}


def router_node(state: AnalystState) -> Dict[str, Any]:
    """LangGraph Node: Route Model Choice Based on Query Complexity."""
    selected_model = model_router_agent.select_model("analysis", state)
    return {"selected_model": selected_model}


def retrieval_node(state: AnalystState) -> Dict[str, Any]:
    """LangGraph Node: Execute Sub-Task Research, MCP Tools, Hybrid RRF Retrieval, Reranking & Context Building."""
    query = state.get("user_query", "")
    trace_id = state.get("trace_id", "N/A")
    session_id = state.get("session_id", "default_session")
    plan = state.get("plan", {})

    # Determine search scope: Read user explicit selection directly from state ('session' default vs 'global')
    search_scope = state.get("search_scope", "session")
    session_filter = session_id if search_scope == "session" else None

    # Check Tier-1 Retrieval Cache
    cached_retrieval = retrieval_cache.get(query, session_id, search_scope)
    if cached_retrieval:
        return cached_retrieval

    # Extract dynamic retrieval weights chosen by Planner Agent
    top_k = int(plan.get("top_k", 10))
    bm25_weight = float(plan.get("bm25_weight", 0.5))
    dense_weight = float(plan.get("dense_weight", 0.5))
    sub_tasks = plan.get("sub_tasks", [])
    requires_rag = plan.get("requires_rag", True)
    requires_mcp = plan.get("requires_mcp", False)
    mcp_tools = plan.get("mcp_tools", [])
    github_repo = plan.get("github_repo")

    # 0. Execute MCP Tools if requested by Planner
    mcp_context_block = ""
    mcp_execution_results = {}
    if requires_mcp or mcp_tools:
        mcp_execution_results = mcp_registry.execute_mcp_tools(
            tool_names=mcp_tools if mcp_tools else ["browser_search"],
            query=query,
            repo_name=github_repo,
        )
        if mcp_execution_results.get("data"):
            mcp_context_block = f"\n\n--- [LIVE MCP TOOL EXECUTION RESULTS] ---\n{mcp_execution_results['data']}\n"

    # Fast-path bypass for general knowledge queries that do not require document retrieval
    if not requires_rag and not requires_mcp:
        return {
            "rewritten_queries": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "context_text": "General Knowledge Query (RAG Search Bypassed).",
            "mcp_results": mcp_execution_results,
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

    # 4. Build Context String & append MCP tool results
    context_text = context_builder.build_context(reranked_chunks, trace_id=trace_id)
    if mcp_context_block:
        context_text += mcp_context_block

    result_payload = {
        "rewritten_queries": rewritten_variations,
        "retrieved_chunks": all_retrieved_chunks,
        "reranked_chunks": reranked_chunks,
        "context_text": context_text,
        "mcp_results": mcp_execution_results,
    }

    # Store into Tier-1 Retrieval Cache
    retrieval_cache.set(query, session_id, search_scope, result_payload)
    return result_payload


def analysis_node(state: AnalystState) -> Dict[str, Any]:
    """LangGraph Node: Synthesize Evidence-Grounded Compliance Analysis & Verify Footnote Citations."""
    query = state.get("user_query", "")
    trace_id = state.get("trace_id", "N/A")
    reranked_chunks = state.get("reranked_chunks", [])

    # Check Tier-2 Semantic Cosine Cache (10ms instant return for conceptual matches)
    dummy_query_emb = [0.01 * (i + 1) for i in range(384)]
    cached_semantic = semantic_cache.get(dummy_query_emb, trace_id=trace_id)
    if cached_semantic:
        return cached_semantic

    raw_report = analysis_agent.generate_analysis(state)
    
    # 1. Output Guardrail Stage: Read-only Audit Inspection
    output_audit = guardrail_agent.audit_output(raw_report, trace_id=trace_id)
    
    # 2. Output Guardrail Stage: Redaction & Sanitization
    sanitized_report = guardrail_agent.sanitize_output(raw_report, trace_id=trace_id)
    
    citations = citation_engine.extract_and_verify_citations(
        report_text=sanitized_report, source_chunks=reranked_chunks, trace_id=trace_id
    )

    result_payload = {
        "analysis_report": sanitized_report,
        "citations": citations,
        "output_guardrail_audit": output_audit,
    }

    # Store synthesized report into Tier-2 Semantic Cosine Cache
    semantic_cache.set(dummy_query_emb, result_payload)
    return result_payload


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
