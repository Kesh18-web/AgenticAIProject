from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ChunkMetadata(TypedDict, total=False):
    doc_id: str
    source_name: str
    page_number: Optional[int]
    line_start: Optional[int]
    line_end: Optional[int]
    score: float
    retrieval_method: str  # 'dense', 'bm25', or 'hybrid_rrf'


class Citation(TypedDict):
    citation_id: int
    source_name: str
    page_number: Optional[int]
    snippet: str


class AnalystState(TypedDict, total=False):
    """Core state schema passed between LangGraph cognitive agent nodes."""

    # User & Request Context
    user_query: str
    trace_id: str
    session_id: str

    # Guardrails
    guardrail_status: Dict[str, Any]  # e.g., {'safe': True, 'reason': None}

    # Dynamic Planning & Sub-Queries
    plan: Dict[str, Any]  # Sub-tasks & search strategies
    rewritten_queries: List[str]

    # Model Routing
    selected_model: str  # e.g., 'gpt-4o', 'claude-3-5-sonnet', 'gemini-1.5-flash'

    # Retrieval & Reranking
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    context_text: str

    # Analysis & Citations
    analysis_report: str
    citations: List[Citation]

    # Reflection & Re-planning Loop
    reflection_confidence: float  # Score between 0.0 and 1.0
    reflection_critique: str  # Gap analysis / missing evidence
    reflection_count: int  # Current loop iteration (max limit, e.g., 3)

    # Judge & Evaluation
    judge_eval_scores: Dict[
        str, float
    ]  # e.g., {'groundedness': 0.95, 'relevance': 0.90}

    # Live UI Execution Log Stream
    node_execution_logs: List[Dict[str, Any]]
