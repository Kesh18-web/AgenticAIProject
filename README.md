# Enterprise AI Analyst Runtime & Hybrid RAG Engine

A production-grade, modular multi-agent runtime and Hybrid RAG Engine built with **LangGraph**, **FastAPI**, **Qdrant**, **Firebase Firestore**, and **Next.js 14**.

## Key Architecture Highlights

1. **Layered Enterprise Guardrail Engine**: Dual pre-LLM regex/LLM input inspection & post-LLM PII/secret redaction.
2. **Cognitive Agent Nodes**:
   - **Planner Agent**: Dynamic sub-task decomposition & continuous BM25/Dense weight tuning.
   - **Model Router Agent**: Real-time task complexity classification (Fast vs Reasoning models).
   - **Analysis Agent**: Multi-document grounded report synthesis.
   - **Reflection Agent**: Cyclic self-critique and dynamic re-planning loop.
   - **LLM-as-a-Judge**: Groundedness, faithfulness, and citation metrics.
3. **Hybrid Retrieval Pipeline**: BM25 Okapi + Qdrant Dense Vector search fused via **Weighted Reciprocal Rank Fusion (Weighted RRF)** and reranked using a HuggingFace Cross-Encoder Transformer.
4. **FastAPI SSE API**: Server-Sent Events real-time event streaming.
5. **Next.js 14 Dashboard**: Dark glassmorphism command center UI.

## Getting Started

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
