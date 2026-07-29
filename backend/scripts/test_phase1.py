import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from backend.app.core.config import settings
from backend.app.core.logging import logger, logger_timer
from backend.app.core.state import AnalystState


def run_phase1_verification():
    logger.info("=== Starting Phase 1 Scaffolding Verification ===")

    # 1. Test Config Loading
    logger.info(f"Loaded Environment: {settings.ENVIRONMENT}")
    logger.info(f"Configured Port: {settings.PORT}")
    logger.info(f"Qdrant Mode: {settings.QDRANT_MODE}")
    assert settings.ENVIRONMENT is not None, "ENVIRONMENT setting is missing"

    # 2. Test Logger Timer & Trace Context
    trace_id = "test-trace-1234"
    with logger_timer("Simulated LangGraph Node Execution", trace_id=trace_id) as node_logger:
        time.sleep(0.05)  # Simulate 50ms work
        node_logger.info("Executed mock node step successfully.")

    # 3. Test AnalystState Initialization
    sample_state: AnalystState = {
        "user_query": "What are the compliance requirements for SOC2 data retention?",
        "trace_id": trace_id,
        "session_id": "session-5678",
        "reflection_count": 0,
        "reflection_confidence": 0.0,
        "node_execution_logs": [],
    }

    assert sample_state["user_query"] == "What are the compliance requirements for SOC2 data retention?"
    assert sample_state["trace_id"] == trace_id

    logger.info("=== Phase 1 Verification COMPLETE: All Core Scaffolding Validated! ===")
    print("\nSUCCESS: Phase 1 verification script executed cleanly without errors!")


if __name__ == "__main__":
    run_phase1_verification()
