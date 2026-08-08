"""
Enterprise AI Quality Platform — Report Generator

Generates a human-readable Markdown benchmark report from a completed
benchmark run dict produced by benchmark_runner.py.

The report is saved to:
    backend/app/quality/reports/benchmark_<run_id>.md
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


_REPORTS_DIR = Path(__file__).parent / "reports"


def _pct(val: float) -> str:
    """Format a 0–1 float as a percentage string."""
    return f"{val * 100:.1f}%"


def _ms(val: float) -> str:
    return f"{val:.0f}ms"


def generate_report(benchmark_report: Dict[str, Any]) -> Path:
    """
    Render the benchmark report to a Markdown file and return its path.

    Args:
        benchmark_report: Output dict from benchmark_runner.run_benchmark()

    Returns:
        Absolute Path to the written .md report file.
    """
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = benchmark_report.get("run_id", "unknown")
    metrics = benchmark_report.get("aggregate_metrics", {})
    cases = benchmark_report.get("per_case_results", [])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_path = _REPORTS_DIR / f"benchmark_{run_id}.md"

    lines = []

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    lines += [
        f"# Enterprise AI Analyst — Quality Benchmark Report",
        f"",
        f"**Run ID**: `{run_id}`  ",
        f"**Timestamp**: {timestamp}  ",
        f"**Total Test Cases**: {metrics.get('total_cases', 0)}  ",
        f"**RAG Test Cases**: {metrics.get('rag_cases', 0)}  ",
        f"",
        "---",
        "",
    ]

    # -----------------------------------------------------------------------
    # Aggregate Metrics Table
    # -----------------------------------------------------------------------
    lines += [
        "## Aggregate Performance Metrics",
        "",
        "### Retrieval Quality",
        "",
        "| Metric | Score |",
        "| :--- | ---: |",
        f"| NDCG@5 (Hybrid RRF + Cross-Encoder) | **{metrics.get('ndcg_at_5', 0.0):.4f}** |",
        f"| Top-1 Retrieval Accuracy | **{_pct(metrics.get('top1_retrieval_accuracy', 0.0))}** |",
        "",
        "### Generation Quality (LLM-as-a-Judge)",
        "",
        "| Metric | Score |",
        "| :--- | ---: |",
        f"| Avg Groundedness | **{_pct(metrics.get('avg_groundedness', 0.0))}** |",
        f"| Avg Answer Relevance | **{_pct(metrics.get('avg_answer_relevance', 0.0))}** |",
        f"| Avg Citation Coverage | **{_pct(metrics.get('avg_citation_coverage', 0.0))}** |",
        f"| Avg Overall Quality | **{_pct(metrics.get('avg_overall_quality', 0.0))}** |",
        "",
        "### Safety & Guardrails",
        "",
        "| Metric | Score |",
        "| :--- | ---: |",
        f"| Guardrail Block Accuracy | **{_pct(metrics.get('guardrail_accuracy', 0.0))}** |",
        f"| PII Redaction Accuracy | **{_pct(metrics.get('pii_redaction_accuracy', 0.0))}** |",
        "",
        "### Planning Accuracy",
        "",
        "| Metric | Score |",
        "| :--- | ---: |",
        f"| Planner Source Routing Accuracy | **{_pct(metrics.get('planner_source_accuracy', 0.0))}** |",
        "",
        "### Latency",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Mean Latency | **{_ms(metrics.get('latency_mean_ms', 0.0))}** |",
        f"| p50 Latency | **{_ms(metrics.get('latency_p50_ms', 0.0))}** |",
        f"| p95 Latency | **{_ms(metrics.get('latency_p95_ms', 0.0))}** |",
        f"| Max Latency | **{_ms(metrics.get('latency_max_ms', 0.0))}** |",
        "",
        "### Cache & Cost",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Cache Hit Rate | **{_pct(metrics.get('cache_hit_rate', 0.0))}** |",
        f"| Total Tokens Consumed | **{metrics.get('total_tokens_consumed', 0):,}** |",
        f"| Total Inference Cost | **${metrics.get('total_cost_usd', 0.0):.6f}** |",
        "",
        "---",
        "",
    ]

    # -----------------------------------------------------------------------
    # Per-Case Results
    # -----------------------------------------------------------------------
    lines += [
        "## Per-Case Results",
        "",
    ]

    for r in cases:
        case_id = r.get("id", "?")
        category = r.get("category", "?")
        query = r.get("query", "")
        expected_source = r.get("expected_source", "?")
        actual_source = r.get("actual_source", "N/A")
        guardrail_passed = r.get("guardrail_passed", True)
        latency_ms = r.get("latency_ms", 0.0)
        ndcg5 = r.get("ndcg_at_5", 0.0)
        top1 = r.get("top1_accuracy", 0.0)
        cache_hit = r.get("cache_hit", False)
        judge = r.get("judge_scores", {})
        citations = r.get("citation_count", 0)
        error = r.get("error")

        status_icon = "✅" if not error else "❌"
        guard_icon = "🛡️ BLOCKED" if not guardrail_passed else "✅ ALLOWED"

        lines += [
            f"### {status_icon} `{case_id}` — {category}",
            f"",
            f"**Query**: `{query}`  ",
            f"**Expected Source**: `{expected_source}` → **Actual Source**: `{actual_source}`  ",
            f"**Guardrail**: {guard_icon}  ",
            f"**Latency**: {_ms(latency_ms)} | **Cache Hit**: {'Yes' if cache_hit else 'No'}  ",
            f"**Citations Generated**: {citations}  ",
        ]

        if category == "enterprise_rag":
            lines += [
                f"**NDCG@5**: `{ndcg5:.4f}` | **Top-1 Accuracy**: `{_pct(top1)}`  ",
            ]

        if judge:
            lines += [
                f"**Judge Scores**: Groundedness=`{judge.get('groundedness', 'N/A')}` | "
                f"Relevance=`{judge.get('answer_relevance', 'N/A')}` | "
                f"Citations=`{judge.get('citation_coverage', 'N/A')}` | "
                f"Overall=`{judge.get('overall_quality', 'N/A')}`  ",
            ]

        # Show a snippet of the report (first 300 chars)
        report_snippet = r.get("analysis_report", "")
        if report_snippet:
            snippet = report_snippet[:300].replace("\n", " ").strip()
            if len(r.get("analysis_report", "")) > 300:
                snippet += "..."
            lines += [
                f"",
                f"> **Response Snippet**: {snippet}",
            ]

        if error:
            lines += [f"", f"⚠️ **Error**: `{error}`"]

        lines += ["", "---", ""]

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------
    lines += [
        "## Notes",
        "",
        "- NDCG@5 uses a graded relevance scale: 2.0 (≥2 keyword matches), 1.0 (1 match), 0.0 (no match).",
        "- Guardrail accuracy measures correct block/allow decisions against ground truth.",
        "- Judge scores are produced by the live LLM-as-a-Judge agent using `gemini-1.5-flash`.",
        "- Latency includes full end-to-end pipeline execution time (cache check → judge).",
        "- Cost is estimated from token counts using published model pricing tables.",
        "",
        f"*Generated by Enterprise AI Quality Platform — {timestamp}*",
    ]

    report_content = "\n".join(lines)
    report_path.write_text(report_content, encoding="utf-8")
    return report_path
