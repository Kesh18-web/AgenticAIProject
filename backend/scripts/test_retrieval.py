import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from backend.app.core.logging import logger
from backend.app.db.bm25 import BM25Indexer
from backend.app.db.firestore import firestore_db
from backend.app.db.qdrant import qdrant_store
from backend.app.infrastructure.citation_engine import citation_engine
from backend.app.infrastructure.context_builder import context_builder
from backend.app.infrastructure.hybrid_retriever import HybridRetriever
from backend.app.infrastructure.query_rewriter import query_rewriter
from backend.app.infrastructure.reranker import cross_encoder_reranker


def run_retrieval_verification():
    logger.info("=== Starting Phase 2 & 3 Infrastructure Verification ===")

    # 1. Sample Corporate Policy Chunks
    sample_chunks = [
        {
            "chunk_id": 1,
            "doc_id": "doc_soc2_policy",
            "source_name": "SOC2_Security_Policy_2025.pdf",
            "page_number": 4,
            "text": "All user authentication data must be encrypted using AES-256 at rest and TLS 1.3 in transit. Audit logs must be retained for at least 365 days.",
        },
        {
            "chunk_id": 2,
            "doc_id": "doc_data_retention",
            "source_name": "Data_Retention_Schedule.pdf",
            "page_number": 12,
            "text": "Customer personal identifiable information (PII) must be purged within 30 days of account deletion unless subject to legal hold requirement.",
        },
        {
            "chunk_id": 3,
            "doc_id": "doc_incident_response",
            "source_name": "Incident_Response_Plan.pdf",
            "page_number": 2,
            "text": "Severity 1 security incidents require notification to executive leadership within 1 hour and breach notification within 72 hours per GDPR guidelines.",
        },
    ]

    # 2. Test Query Rewriter
    query = "What is the policy for encryption and log retention under SOC2?"
    rewritten = query_rewriter.rewrite_query(query, num_variations=3)
    assert len(rewritten) >= 1, "Query rewriter failed to generate variations"

    # 3. Test BM25 Indexer
    bm25_mgr = BM25Indexer()
    bm25_mgr.index_chunks(sample_chunks)
    bm25_results = bm25_mgr.search_bm25("encryption AES-256 retention", top_k=2)
    logger.info(f"BM25 Top Result: {bm25_results[0]['source_name']} (score={bm25_results[0]['score']:.2f})")
    assert len(bm25_results) > 0, "BM25 keyword search returned empty results"

    # 4. Test Qdrant Vector Store
    # Generate dummy embeddings (384-dimensional) for testing
    dummy_embeddings = [
        [0.01 * (i + 1) for i in range(384)],
        [0.02 * (i + 1) for i in range(384)],
        [0.03 * (i + 1) for i in range(384)],
    ]
    qdrant_store.upsert_chunks(
        collection_name="test_collection",
        chunks=sample_chunks,
        embeddings=dummy_embeddings,
    )
    dense_results = qdrant_store.search_dense(
        collection_name="test_collection",
        query_embedding=dummy_embeddings[0],
        limit=2,
    )
    assert len(dense_results) > 0, "Qdrant dense vector search returned empty results"

    # 5. Test Hybrid Retriever (RRF)
    retriever = HybridRetriever(qdrant_mgr=qdrant_store, bm25_mgr=bm25_mgr)
    hybrid_results = retriever.retrieve_hybrid(
        collection_name="test_collection",
        query=query,
        query_embedding=dummy_embeddings[0],
        top_k=3,
    )
    assert len(hybrid_results) > 0, "Hybrid RRF retriever returned no candidates"
    logger.info(f"RRF Top Candidate: {hybrid_results[0]['source_name']} (RRF score={hybrid_results[0]['score']:.4f})")

    # 6. Test Cross-Encoder Reranker
    reranked_chunks = cross_encoder_reranker.rerank(
        query=query, chunks=hybrid_results, top_n=2
    )
    assert len(reranked_chunks) > 0, "Reranker returned empty results"

    # 7. Test Context Builder & Citation Engine
    context_text = context_builder.build_context(reranked_chunks)
    assert "DOCUMENT CHUNK" in context_text, "ContextBuilder formatting failed"

    simulated_report = "According to company policy [Doc 1], all user authentication data must be encrypted with AES-256."
    citations = citation_engine.extract_and_verify_citations(
        report_text=simulated_report, source_chunks=reranked_chunks
    )
    assert len(citations) == 1, "CitationEngine failed to extract inline citations"
    assert citations[0]["source_name"] == reranked_chunks[0]["source_name"]

    # 8. Test Firestore DB manager
    firestore_db.save_document(
        collection_name="test_logs",
        doc_id="test_run_1",
        data={"status": "passed", "query": query},
    )
    retrieved_doc = firestore_db.get_document("test_logs", "test_run_1")
    assert retrieved_doc is not None and retrieved_doc["status"] == "passed"

    logger.info("=== Phase 2 & 3 Infrastructure Verification COMPLETE: ALL TESTS PASSED! ===")
    print("\nSUCCESS: All deterministic infrastructure components (BM25, Qdrant, RRF, Reranker, ContextBuilder, CitationEngine, Firestore) verified cleanly!")


if __name__ == "__main__":
    run_retrieval_verification()
