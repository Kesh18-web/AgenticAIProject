"""
Enterprise AI Quality Platform — Metrics Engine

Computes NDCG@k, Top-1 Retrieval Accuracy, Groundedness, Citation Coverage,
Relevance, Cache Performance, and Latency metrics from benchmark run results.
"""

import math
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# NDCG@K
# ---------------------------------------------------------------------------

def _dcg(relevances: List[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(
    retrieved_chunk_texts: List[str],
    relevant_keywords: List[str],
    k: int = 5,
) -> float:
    """
    Compute NDCG@K for a single query.

    Relevance grading (per chunk):
      - 2.0: chunk text contains >= 2 of the relevant_keywords (highly relevant)
      - 1.0: chunk text contains exactly 1 relevant_keyword (partial match)
      - 0.0: no match

    The ideal DCG is computed from the sorted perfect relevance list.
    """
    if not retrieved_chunk_texts or not relevant_keywords:
        return 0.0

    keywords_lower = [kw.lower() for kw in relevant_keywords]

    relevances: List[float] = []
    for chunk_text in retrieved_chunk_texts[:k]:
        text_lower = chunk_text.lower()
        matches = sum(1 for kw in keywords_lower if kw in text_lower)
        if matches >= 2:
            relevances.append(2.0)
        elif matches == 1:
            relevances.append(1.0)
        else:
            relevances.append(0.0)

    ideal_relevances = sorted(relevances, reverse=True)
    actual_dcg = _dcg(relevances, k)
    ideal_dcg = _dcg(ideal_relevances, k)

    if ideal_dcg == 0.0:
        return 0.0

    return round(actual_dcg / ideal_dcg, 4)


def top1_retrieval_accuracy(
    retrieved_chunk_texts: List[str],
    relevant_keywords: List[str],
) -> float:
    """
    Top-1 Retrieval Accuracy: 1.0 if the first retrieved chunk contains
    at least one relevant keyword, 0.0 otherwise.
    """
    if not retrieved_chunk_texts or not relevant_keywords:
        return 0.0

    top_chunk = retrieved_chunk_texts[0].lower()
    keywords_lower = [kw.lower() for kw in relevant_keywords]
    hit = any(kw in top_chunk for kw in keywords_lower)
    return 1.0 if hit else 0.0


# ---------------------------------------------------------------------------
# Guardrail Accuracy
# ---------------------------------------------------------------------------

def guardrail_accuracy(results: List[Dict[str, Any]]) -> float:
    """
    Fraction of guardrail test cases where the block decision was correct.
    A 'correct' guardrail result means:
      - expected_source == 'BLOCKED' AND guardrail blocked the query, OR
      - expected_source != 'BLOCKED' AND guardrail allowed the query.
    """
    guardrail_cases = [r for r in results if r.get("category") == "guardrail_security"]
    if not guardrail_cases:
        return 1.0

    correct = 0
    for r in guardrail_cases:
        expected_blocked = r.get("expected_source") == "BLOCKED"
        actually_blocked = not r.get("guardrail_passed", True)
        if expected_blocked == actually_blocked:
            correct += 1

    return round(correct / len(guardrail_cases), 4)


# ---------------------------------------------------------------------------
# PII Redaction Accuracy
# ---------------------------------------------------------------------------

def pii_redaction_accuracy(results: List[Dict[str, Any]]) -> float:
    """
    For PII test cases: 1.0 if output_report contains '[REDACTED_' tokens,
    indicating the output guardrail correctly redacted sensitive content.
    """
    pii_cases = [r for r in results if r.get("category") == "pii_redaction"]
    if not pii_cases:
        return 1.0

    correct = 0
    for r in pii_cases:
        report = r.get("analysis_report", "")
        if "[REDACTED_" in report:
            correct += 1

    return round(correct / len(pii_cases), 4)


# ---------------------------------------------------------------------------
# Planner Source Accuracy
# ---------------------------------------------------------------------------

def planner_source_accuracy(results: List[Dict[str, Any]]) -> float:
    """
    Fraction of non-blocked test cases where the Planner correctly selected
    the expected primary_knowledge_source.
    """
    non_blocked = [
        r for r in results
        if r.get("expected_source") not in ("BLOCKED",)
        and r.get("category") not in ("guardrail_security",)
    ]
    if not non_blocked:
        return 1.0

    correct = sum(
        1 for r in non_blocked
        if r.get("actual_source", "") == r.get("expected_source", "")
    )
    return round(correct / len(non_blocked), 4)


# ---------------------------------------------------------------------------
# Latency, Token, Cost Aggregates
# ---------------------------------------------------------------------------

def latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute p50, p95, mean, and max latency from a list of per-query latencies."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)
    mean_ms = round(sum(sorted_lats) / n, 2)
    p50 = round(sorted_lats[int(n * 0.50)], 2)
    p95 = round(sorted_lats[min(int(n * 0.95), n - 1)], 2)
    max_ms = round(sorted_lats[-1], 2)

    return {"mean_ms": mean_ms, "p50_ms": p50, "p95_ms": p95, "max_ms": max_ms}


def cache_hit_rate(results: List[Dict[str, Any]]) -> float:
    """Fraction of benchmark queries served from cache (either tier)."""
    if not results:
        return 0.0
    hits = sum(1 for r in results if r.get("cache_hit", False))
    return round(hits / len(results), 4)


def aggregate_judge_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Average LLM-as-a-Judge scores across all RAG query results."""
    rag_results = [
        r for r in results
        if r.get("category") == "enterprise_rag"
        and r.get("judge_scores") is not None
    ]
    if not rag_results:
        return {}

    keys = ["groundedness", "answer_relevance", "citation_coverage", "overall_quality"]
    averages: Dict[str, float] = {}
    for key in keys:
        vals = [r["judge_scores"].get(key, 0.0) for r in rag_results if r.get("judge_scores")]
        averages[key] = round(sum(vals) / len(vals), 4) if vals else 0.0

    return averages


def aggregate_ndcg(results: List[Dict[str, Any]], k: int = 5) -> float:
    """Average NDCG@K across all RAG retrieval queries."""
    rag_results = [r for r in results if r.get("category") == "enterprise_rag"]
    if not rag_results:
        return 0.0

    scores = [r.get(f"ndcg_at_{k}", 0.0) for r in rag_results]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def aggregate_top1_accuracy(results: List[Dict[str, Any]]) -> float:
    """Average Top-1 retrieval accuracy across all RAG queries."""
    rag_results = [r for r in results if r.get("category") == "enterprise_rag"]
    if not rag_results:
        return 0.0

    scores = [r.get("top1_accuracy", 0.0) for r in rag_results]
    return round(sum(scores) / len(scores), 4) if scores else 0.0
