import asyncio
import sys
from pathlib import Path
import httpx

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from backend.app.core.logging import logger
from backend.app.main import app


async def run_api_verification():
    logger.info("=== Starting Phase 5 FastAPI REST & SSE API Verification ===")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Test Health Endpoint
        res_health = await client.get("/api/v1/health")
        assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
        data_health = res_health.json()
        logger.info(f"Health Check Response: {data_health}")
        assert data_health["status"] == "healthy"

        # 2. Test Document Indexing Endpoint
        doc_payload = {
            "title": "SOC2 Compliance Overview",
            "content": "All access to production systems requires multi-factor authentication. Log files must be audited monthly.",
            "source_name": "SOC2_Overview.txt",
        }
        res_doc = await client.post("/api/v1/documents/index", json=doc_payload)
        assert res_doc.status_code == 200, f"Document indexing failed: {res_doc.text}"
        data_doc = res_doc.json()
        logger.info(f"Document Ingestion Response: {data_doc}")
        assert data_doc["chunks_indexed"] > 0

        # 3. Test SSE Streaming Analysis Endpoint
        analyze_payload = {
            "query": "What are the MFA rules for production access?",
            "session_id": "test_session_api",
        }
        logger.info("Testing SSE Event Stream endpoint...")
        async with client.stream(
            "POST", "/api/v1/analyze/stream", json=analyze_payload
        ) as response:
            assert response.status_code == 200, "SSE Streaming endpoint failed"
            received_events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    received_events.append(line)
                    logger.info(f"SSE Event Received: {line[:80]}...")
                    if len(received_events) >= 5:
                        break

            assert len(received_events) > 0, "No SSE events received from graph stream"

    logger.info("=== Phase 5 FastAPI & SSE API Verification COMPLETE: ALL TESTS PASSED! ===")
    print("\nSUCCESS: FastAPI Server endpoints (Health, Document Indexing, SSE Streaming Graph) verified cleanly!")


if __name__ == "__main__":
    asyncio.run(run_api_verification())
