# Project Operational Rules - Enterprise AI Analyst

## Core Directives
1. **First-Time Accuracy & Prevention First**:
   - Always verify schemas, signatures, and environment requirements before writing code.
   - If an operational requirement or API parameter is ambiguous, ask the user immediately rather than making assumptions that cause refactoring later.
2. **Co-Partner Mindset**:
   - Do not blindly agree or follow flawed ideas.
   - Provide constructive feedback, challenge assumptions, and propose alternative state-of-the-art designs when beneficial.
3. **Incremental Verification**:
   - Every module must have a corresponding standalone test script (`backend/scripts/test_*.py`) or unit test (`backend/tests/`).
   - Never consider a module "done" until verified with an actual execution run.
4. **Structured Enterprise Telemetry**:
   - Use structured JSON logging (`loguru`) across all services.
   - Include trace correlation IDs (`trace_id`) and execution timers for every node in the agent graph.
