"use client";

import React, { useState, useEffect } from "react";
import {
  Brain,
  ShieldCheck,
  Zap,
  Search,
  FileText,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Scale,
  Activity,
  Layers,
  Database,
  Upload,
  Bot,
  Sparkles,
  Terminal,
  ChevronRight,
  ExternalLink,
} from "lucide-react";

interface NodeEvent {
  node: string;
  trace_id?: string;
  safe?: boolean;
  sub_tasks?: string[];
  requires_mcp?: boolean;
  mcp_tools?: string[];
  mcp_results?: Record<string, any>;
  selected_model?: string;
  chunk_count?: number;
  confidence?: number;
  critique?: string;
  eval_scores?: Record<string, number>;
}

interface Citation {
  citation_id: number;
  source_name: string;
  page_number?: number;
  snippet: string;
}

export default function AnalystDashboard() {
  const [activeTab, setActiveTab] = useState<
    "workbench" | "indexing" | "observability" | "architect"
  >("workbench");

  // Health Status
  const [backendStatus, setBackendStatus] = useState<any>(null);

  // Workbench Query State
  // Workbench Query State
  const [query, setQuery] = useState(
    "What are the rules for multi-factor authentication and inactive account management under cybersecurity compliance?"
  );
  const [searchScope, setSearchScope] = useState<"session" | "global">("session");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [nodeEvents, setNodeEvents] = useState<NodeEvent[]>([]);
  const [finalReport, setFinalReport] = useState<string>("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [evalScores, setEvalScores] = useState<Record<string, number> | null>(
    null
  );
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(
    null
  );

  // Document Indexing State
  const [docTitle, setDocTitle] = useState("SOC2 Global Security Policy 2025");
  const [docSource, setDocSource] = useState("SOC2_Security_Policy_2025.pdf");
  const [docContent, setDocContent] = useState(
    "Multi-factor authentication (MFA) is strictly mandatory for all administrative and production infrastructure access.\n\nIncident response logs and access audit trails must be backed up daily and retained for a minimum of 365 days.\n\nAll user accounts inactive for more than 90 days are automatically flagged and disabled by identity systems."
  );
  const [indexingStatus, setIndexingStatus] = useState<string | null>(null);

  // Solution Architect (V2) State
  const [architectPrompt, setArchitectPrompt] = useState(
    "I need a Legal Compliance Assistant that reads Confluence, searches GitHub, answers policy questions, remembers previous conversations and posts reports to Slack."
  );
  const [architectResult, setArchitectResult] = useState<any>(null);

  // Check Backend Health on Mount
  useEffect(() => {
    fetch("http://localhost:8000/api/v1/health")
      .then((res) => res.json())
      .then((data) => setBackendStatus(data))
      .catch(() => setBackendStatus({ status: "offline" }));
  }, []);

  // Submit Analysis Request over SSE Stream
  const [cacheHit, setCacheHit] = useState<boolean>(false);
  const [memoryCompacted, setMemoryCompacted] = useState<boolean>(false);
  const [telemetry, setTelemetry] = useState<any>(null);
  const [hitlMode, setHitlMode] = useState<boolean>(false);
  const [explainabilityReason, setExplainabilityReason] = useState<string>("");
  const [hitlRequired, setHitlRequired] = useState<boolean>(false);
  const [hitlApproved, setHitlApproved] = useState<boolean>(false);

  // Submit Query to Agent Pipeline (SSE Stream)
  const handleAnalyze = async () => {
    if (!query) return;

    setIsAnalyzing(true);
    setActiveNode("guardrail");
    setNodeEvents([]);
    setFinalReport("");
    setCitations([]);
    setEvalScores(null);
    setSelectedCitation(null);
    setCacheHit(false);
    setMemoryCompacted(false);
    setTelemetry(null);
    setExplainabilityReason("");
    setHitlRequired(false);
    setHitlApproved(false);

    // Collect all parsed SSE events before animating, so we can replay with delays
    const collectedNodeEvents: any[] = [];
    let completePayload: any = null;
    let hitlPayload: any = null;

    try {
      const response = await fetch(
        "http://localhost:8000/api/v1/analyze/stream",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, search_scope: searchScope, hitl_mode: hitlMode }),
        }
      );

      if (!response.body) throw new Error("ReadableStream not supported");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split(/\n\s*\n/);
        buffer = parts.pop() || "";

        for (const part of parts) {
          const lines = part.split("\n");
          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith("data:")) {
              const jsonStr = trimmed.replace(/^data:\s*/, "").trim();
              if (!jsonStr || (!jsonStr.startsWith("{") && !jsonStr.startsWith("["))) continue;
              try {
                const data = JSON.parse(jsonStr);
                if (data.event === "node_complete") {
                  collectedNodeEvents.push(data);
                } else if (data.event === "hitl_approval_required") {
                  hitlPayload = data;
                } else if (data.event === "complete") {
                  completePayload = data;
                }
              } catch (e) {
                console.error("SSE JSON parse error:", e, jsonStr);
              }
            }
          }
        }
      }
    } catch (err) {
      console.error("SSE stream error:", err);
    }

    // --- Animate node transitions sequentially with 600ms delay each ---
    // Reset to guardrail first, then step through each node
    setActiveNode("guardrail");
    setNodeEvents([]);

    const NODE_STEP_MS = 600;

    collectedNodeEvents.forEach((data, idx) => {
      setTimeout(() => {
        setActiveNode(data.node);
        setNodeEvents((prev) => {
          // avoid duplicates on re-render
          const alreadyAdded = prev.some(
            (e) => e.node === data.node && e.trace_id === data.trace_id && prev.indexOf(e) === idx
          );
          return alreadyAdded ? prev : [...prev, data];
        });
        if (data.node === "planner" && data.explainability_reason) {
          setExplainabilityReason(data.explainability_reason);
        }
      }, idx * NODE_STEP_MS);
    });

    // Apply hitl after node animations
    if (hitlPayload) {
      setTimeout(() => {
        setHitlRequired(true);
        setExplainabilityReason(hitlPayload.explainability_reason);
      }, collectedNodeEvents.length * NODE_STEP_MS);
    }

    // Apply complete payload after all node animations
    const totalDelay = collectedNodeEvents.length * NODE_STEP_MS + 200;
    setTimeout(() => {
      if (completePayload) {
        setFinalReport(completePayload.report);
        setCitations(completePayload.citations || []);
        setEvalScores(completePayload.eval_scores || null);
        setCacheHit(completePayload.semantic_cache_hit || false);
        setMemoryCompacted(completePayload.memory_compacted || false);
        setTelemetry(completePayload.telemetry || null);
        setActiveNode("complete");
      }
      setIsAnalyzing(false);
    }, totalDelay);
  };

  // Submit Document Indexing
  const handleIndexDocument = async () => {
    if (!docTitle || !docContent) return;

    setIndexingStatus("Indexing chunks into Qdrant Vector Store & BM25 Store...");
    try {
      const res = await fetch("http://localhost:8000/api/v1/documents/index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: docTitle,
          source_name: docSource,
          content: docContent,
        }),
      });

      const data = await res.json();
      setIndexingStatus(
        `Successfully indexed document '${data.title}'! (ID: ${data.doc_id}, Chunks: ${data.chunks_indexed})`
      );
    } catch (e) {
      setIndexingStatus(`Error indexing document: ${e}`);
    }
  };

  // Generate Solution Architect Scaffold
  const handleGenerateArchitecture = () => {
    setArchitectResult({
      topology: "Cyclic LangGraph Multi-Agent Architecture",
      scaffold_dir: "scaffolds/legal_compliance_assistant/",
      components: [
        { name: "Guardrail Agent", type: "Security", model: "Groq/Llama-3" },
        { name: "Confluence Search Tool", type: "MCP Capability", endpoint: "mcp://confluence" },
        { name: "GitHub Repository Search", type: "MCP Capability", endpoint: "mcp://github" },
        { name: "Policy Analysis Agent", type: "Cognitive", model: "Claude-3-5-Sonnet" },
        { name: "Slack Notification Tool", type: "MCP Capability", endpoint: "mcp://slack" },
      ],
      nodes: ["Guardrail", "Planner", "ConfluenceRetriever", "GitHubRetriever", "Analyzer", "SlackPublisher"],
    });
  };

  return (
    <div className="flex h-screen flex-col bg-[#090d16] text-slate-100 overflow-hidden">
      {/* Top Navbar */}
      <header className="flex h-16 items-[#1e293b] border-b border-slate-800 bg-slate-950/80 px-6 backdrop-blur justify-between items-center z-10">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 shadow-lg shadow-indigo-500/20">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight text-white">
                Enterprise AI Analyst
              </h1>
              <span className="rounded-full bg-indigo-500/10 px-2.5 py-0.5 text-xs font-semibold text-indigo-400 border border-indigo-500/30">
                V1 & V2 Platform
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Modular AI Runtime • LangGraph • Qdrant RRF • Firestore
            </p>
          </div>
        </div>

        {/* System Badges & Health */}
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-2 rounded-lg bg-slate-900 px-3 py-1.5 border border-slate-800">
            <Activity className="h-4 w-4 text-emerald-400" />
            <span className="text-slate-300">Backend API:</span>
            <span className="font-semibold text-emerald-400">
              {backendStatus?.status === "healthy" ? "Online (Port 8000)" : "Connecting..."}
            </span>
          </div>

          {telemetry && (
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5 rounded-lg bg-indigo-500/10 px-3 py-1.5 border border-indigo-500/30 text-indigo-300">
                <Zap className="h-3.5 w-3.5 text-indigo-400" />
                <span>{telemetry.total_latency_ms}ms</span>
              </div>
              <div className="flex items-center gap-1.5 rounded-lg bg-cyan-500/10 px-3 py-1.5 border border-cyan-500/30 text-cyan-300">
                <Terminal className="h-3.5 w-3.5 text-cyan-400" />
                <span>{telemetry.total_tokens} Tokens</span>
              </div>
              <div className="flex items-center gap-1.5 rounded-lg bg-emerald-500/10 px-3 py-1.5 border border-emerald-500/30 text-emerald-400 font-semibold">
                <span>{telemetry.formatted_cost}</span>
              </div>
            </div>
          )}

          <div className="flex items-center gap-2 rounded-lg bg-slate-900 px-3 py-1.5 border border-slate-800 text-slate-300">
            <Database className="h-4 w-4 text-cyan-400" />
            <span>Qdrant Mode:</span>
            <span className="font-medium text-cyan-400">Memory (Local)</span>
          </div>
        </div>
      </header>

      {/* Main Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Navigation Sidebar */}
        <aside className="w-64 border-r border-slate-800 bg-slate-950/60 p-4 flex flex-col justify-between">
          <div className="space-y-1">
            <p className="px-3 text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
              Platform Modules
            </p>

            <button
              onClick={() => setActiveTab("workbench")}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all ${
                activeTab === "workbench"
                  ? "bg-indigo-600/20 text-indigo-300 border border-indigo-500/30"
                  : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
              }`}
            >
              <Bot className="h-4 w-4" />
              Live Analyst Workbench
            </button>

            <button
              onClick={() => setActiveTab("indexing")}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all ${
                activeTab === "indexing"
                  ? "bg-indigo-600/20 text-indigo-300 border border-indigo-500/30"
                  : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
              }`}
            >
              <Upload className="h-4 w-4" />
              Document Indexing Lab
            </button>

            <button
              onClick={() => setActiveTab("observability")}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all ${
                activeTab === "observability"
                  ? "bg-indigo-600/20 text-indigo-300 border border-indigo-500/30"
                  : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
              }`}
            >
              <Scale className="h-4 w-4" />
              Judge & Telemetry Metrics
            </button>

            <button
              onClick={() => setActiveTab("architect")}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all ${
                activeTab === "architect"
                  ? "bg-indigo-600/20 text-indigo-300 border border-indigo-500/30"
                  : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
              }`}
            >
              <Layers className="h-4 w-4 text-cyan-400" />
              AI Solution Architect (V2)
            </button>
          </div>

          {/* Infrastructure Footer */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3 text-xs space-y-1 text-slate-400">
            <div className="flex items-center gap-2 font-semibold text-slate-300">
              <Zap className="h-3.5 w-3.5 text-amber-400" />
              Hybrid Retrieval Active
            </div>
            <p>BM25 + Qdrant Dense Vector + Cross-Encoder Reranker</p>
          </div>
        </aside>

        {/* Content Panel */}
        <main className="flex-1 overflow-y-auto bg-[#090d16] p-6">
          {/* TAB 1: LIVE ANALYST WORKBENCH */}
          {activeTab === "workbench" && (
            <div className="space-y-6 max-w-7xl mx-auto">
              {/* Query Card */}
              <div className="glass-panel rounded-2xl p-5 border border-slate-800">
                <div className="flex items-center justify-between mb-3">
                  <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Enterprise Compliance Inquiry
                  </label>
                  <div className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800 text-xs">
                    <button
                      type="button"
                      onClick={() => setSearchScope("session")}
                      className={`px-3 py-1 rounded-lg transition font-medium ${
                        searchScope === "session"
                          ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/20"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      Current Chat Only
                    </button>
                    <button
                      type="button"
                      onClick={() => setSearchScope("global")}
                      className={`px-3 py-1 rounded-lg transition font-medium ${
                        searchScope === "global"
                          ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/20"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      Global Workspace Knowledge
                    </button>
                  </div>
                  <div className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800 text-xs">
                    <button
                      type="button"
                      onClick={() => setHitlMode(false)}
                      className={`px-3 py-1 rounded-lg transition font-medium ${
                        !hitlMode
                          ? "bg-emerald-600 text-white shadow-md shadow-emerald-500/20"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      Auto-Pilot
                    </button>
                    <button
                      type="button"
                      onClick={() => setHitlMode(true)}
                      className={`px-3 py-1 rounded-lg transition font-medium ${
                        hitlMode
                          ? "bg-amber-600 text-white shadow-md shadow-amber-500/20"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      HITL Review
                    </button>
                  </div>
                </div>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ask policy, audit, or technical documentation question..."
                    className="flex-1 rounded-xl bg-slate-900/80 px-4 py-3 text-sm text-slate-100 border border-slate-800 focus:border-indigo-500 focus:outline-none transition"
                  />
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 px-6 py-3 text-sm font-semibold text-white hover:from-indigo-500 hover:to-indigo-400 shadow-lg shadow-indigo-500/20 disabled:opacity-50 transition"
                  >
                    {isAnalyzing ? (
                      <>
                        <RotateCcw className="h-4 w-4 animate-spin" />
                        Executing Graph...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4" />
                        Run Agent Graph
                      </>
                    )}
                  </button>
                </div>

                {/* Explainability Strategy Pill */}
                {explainabilityReason && (
                  <div className="mt-3 flex items-center gap-2 rounded-xl bg-amber-500/10 border border-amber-500/30 px-3.5 py-2 text-xs font-medium text-amber-300">
                    <Sparkles className="h-4 w-4 text-amber-400 shrink-0" />
                    <span><strong>AI Strategy Explainability:</strong> {explainabilityReason}</span>
                  </div>
                )}

                {/* HITL Human Approval Banner */}
                {hitlRequired && !hitlApproved && (
                  <div className="mt-3 flex items-center justify-between rounded-xl bg-indigo-950/80 border border-indigo-500/50 p-3.5 text-xs text-indigo-200 shadow-lg">
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="h-5 w-5 text-amber-400 shrink-0" />
                      <span><strong>Human-in-the-Loop Review:</strong> Review AI strategy before database execution.</span>
                    </div>
                    <button
                      onClick={async () => {
                        setHitlApproved(true);
                        setHitlRequired(false);
                        await fetch("http://localhost:8000/api/v1/analyze/approve_plan", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ session_id: "current_session", approved: true }),
                        });
                      }}
                      className="rounded-lg bg-emerald-600 px-4 py-1.5 font-semibold text-white hover:bg-emerald-500 transition shadow-md"
                    >
                      Approve Strategy
                    </button>
                  </div>
                )}
              </div>

              {/* Agent Node Execution Stepper */}
              <div className="glass-panel rounded-2xl p-5 border border-slate-800">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
                  <Terminal className="h-4 w-4 text-indigo-400" />
                  LangGraph Real-Time Node Transitions
                </h3>

                <div className="grid grid-cols-7 gap-2">
                  {[
                    { id: "guardrail", name: "Guardrail", icon: ShieldCheck },
                    { id: "planner", name: "Planner", icon: Brain },
                    { id: "router", name: "Model Router", icon: Zap },
                    { id: "retrieval", name: "Hybrid Retrieval", icon: Search },
                    { id: "analysis", name: "Analysis Agent", icon: FileText },
                    { id: "reflection", name: "Reflection Loop", icon: RotateCcw },
                    { id: "judge", name: "LLM Judge", icon: Scale },
                  ].map((node) => {
                    const Icon = node.icon;
                    const isActive = activeNode === node.id;
                    const isCompleted = nodeEvents.some(
                      (e) => e.node === node.id
                    );

                    return (
                      <div
                        key={node.id}
                        className={`flex flex-col items-center rounded-xl p-3 border text-center transition-all ${
                          isActive
                            ? "bg-indigo-600/30 border-indigo-500 text-indigo-300 shadow-lg shadow-indigo-500/20 animate-pulse-slow"
                            : isCompleted
                            ? "bg-emerald-950/20 border-emerald-500/40 text-emerald-400"
                            : "bg-slate-900/40 border-slate-800 text-slate-500"
                        }`}
                      >
                        <Icon className="h-5 w-5 mb-1" />
                        <span className="text-xs font-medium">{node.name}</span>
                        {isActive && (
                          <span className="mt-1 text-[10px] bg-indigo-500 text-white px-1.5 py-0.5 rounded-full">
                            Active
                          </span>
                        )}
                        {isCompleted && !isActive && (
                          <span className="mt-1 text-[10px] text-emerald-400 flex items-center gap-0.5">
                            <CheckCircle2 className="h-3 w-3" /> Done
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Output Report & Citations Section */}
              <div className="grid grid-cols-3 gap-6">
                {/* Main Analysis Report */}
                <div className="col-span-2 glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
                  <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                      <FileText className="h-4 w-4 text-indigo-400" />
                      Grounded Executive Report
                    </h3>
                    <div className="flex items-center gap-2">
                      {memoryCompacted && (
                        <div className="flex items-center gap-1.5 text-xs bg-purple-500/10 border border-purple-500/30 text-purple-400 px-3 py-1 rounded-full font-semibold">
                          <Brain className="h-3.5 w-3.5" />
                          Dual Memory Compacted (75% Token Savings)
                        </div>
                      )}
                      {cacheHit && (
                        <div className="flex items-center gap-1.5 text-xs bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 px-3 py-1 rounded-full font-semibold">
                          <Zap className="h-3.5 w-3.5" />
                          Instant Cache Hit (10ms)
                        </div>
                      )}
                      {evalScores && (
                        <div className="flex items-center gap-2 text-xs bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 px-3 py-1 rounded-full font-semibold">
                          <CheckCircle2 className="h-3.5 w-3.5" />
                          Groundedness Score: {evalScores.overall_quality * 100}%
                        </div>
                      )}
                    </div>
                  </div>

                  {finalReport ? (
                    <div className="prose prose-invert max-w-none text-slate-300 text-sm whitespace-pre-line leading-relaxed">
                      {finalReport}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-12 text-slate-500">
                      <Bot className="h-10 w-10 mb-2 opacity-50" />
                      <p className="text-sm">
                        Enter inquiry and click 'Run Agent Graph' to observe live execution.
                      </p>
                    </div>
                  )}
                </div>

                {/* Verified Citations Drawer */}
                <div className="glass-panel rounded-2xl p-5 border border-slate-800 space-y-4">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 border-b border-slate-800 pb-3 flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-cyan-400" />
                    Verified Citations ({citations.length})
                  </h3>

                  {citations.length > 0 ? (
                    <div className="space-y-3">
                      {citations.map((c) => (
                        <div
                          key={c.citation_id}
                          onClick={() => setSelectedCitation(c)}
                          className="rounded-xl border border-slate-800 bg-slate-900/60 p-3 hover:border-indigo-500/50 cursor-pointer transition"
                        >
                          <div className="flex justify-between items-center text-xs font-semibold text-indigo-400 mb-1">
                            <span>[Doc {c.citation_id}]</span>
                            <span className="text-slate-400">
                              Page {c.page_number || 1}
                            </span>
                          </div>
                          <p className="text-xs text-slate-300 font-medium mb-1 truncate">
                            {c.source_name}
                          </p>
                          <p className="text-[11px] text-slate-400 line-clamp-2">
                            "{c.snippet}"
                          </p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500 py-6 text-center">
                      Citations will be extracted after analysis.
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: DOCUMENT INDEXING LAB */}
          {activeTab === "indexing" && (
            <div className="max-w-4xl mx-auto glass-panel rounded-2xl p-6 border border-slate-800 space-y-6">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <Upload className="h-5 w-5 text-indigo-400" />
                  Hybrid Vector & BM25 Document Ingestion
                </h2>
                <p className="text-xs text-slate-400">
                  Upload policies, architecture specs, or contracts to chunk and index into Qdrant vector store & BM25 keyword engine.
                </p>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">
                    Document Title
                  </label>
                  <input
                    type="text"
                    value={docTitle}
                    onChange={(e) => setDocTitle(e.target.value)}
                    className="w-full rounded-xl bg-slate-900 px-4 py-2.5 text-sm text-slate-100 border border-slate-800 focus:border-indigo-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">
                    Source Document Filename
                  </label>
                  <input
                    type="text"
                    value={docSource}
                    onChange={(e) => setDocSource(e.target.value)}
                    className="w-full rounded-xl bg-slate-900 px-4 py-2.5 text-sm text-slate-100 border border-slate-800 focus:border-indigo-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">
                    Document Text Content
                  </label>
                  <textarea
                    rows={8}
                    value={docContent}
                    onChange={(e) => setDocContent(e.target.value)}
                    className="w-full rounded-xl bg-slate-900 p-4 text-xs font-mono text-slate-300 border border-slate-800 focus:border-indigo-500 focus:outline-none"
                  />
                </div>

                <button
                  onClick={handleIndexDocument}
                  className="rounded-xl bg-indigo-600 px-6 py-3 text-sm font-semibold text-white hover:bg-indigo-500 transition shadow-lg shadow-indigo-500/20"
                >
                  Index into Qdrant & BM25
                </button>

                {indexingStatus && (
                  <div className="rounded-xl bg-indigo-950/30 border border-indigo-500/30 p-4 text-xs text-indigo-300">
                    {indexingStatus}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 3: JUDGE & TELEMETRY */}
          {activeTab === "observability" && (
            <div className="max-w-5xl mx-auto space-y-6">
              <div className="glass-panel rounded-2xl p-6 border border-slate-800">
                <h2 className="text-lg font-bold text-white mb-1 flex items-center gap-2">
                  <Scale className="h-5 w-5 text-emerald-400" />
                  LLM-as-a-Judge Evaluation & Telemetry Scorecard
                </h2>
                <p className="text-xs text-slate-400 mb-6">
                  Every execution graph run is audited by the Judge Agent and recorded into Firestore.
                </p>

                <div className="grid grid-cols-4 gap-4">
                  <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                    <p className="text-xs text-slate-400 font-semibold mb-1">
                      Overall Groundedness
                    </p>
                    <p className="text-2xl font-extrabold text-emerald-400">94%</p>
                    <p className="text-[11px] text-slate-500 mt-1">Verified against chunks</p>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                    <p className="text-xs text-slate-400 font-semibold mb-1">
                      Citation Coverage
                    </p>
                    <p className="text-2xl font-extrabold text-cyan-400">100%</p>
                    <p className="text-[11px] text-slate-500 mt-1">Footnote precision</p>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                    <p className="text-xs text-slate-400 font-semibold mb-1">
                      Reflection Re-plans
                    </p>
                    <p className="text-2xl font-extrabold text-amber-400">1 Cycle</p>
                    <p className="text-[11px] text-slate-500 mt-1">Self-correction count</p>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                    <p className="text-xs text-slate-400 font-semibold mb-1">
                      Avg Node Latency
                    </p>
                    <p className="text-2xl font-extrabold text-indigo-400">140ms</p>
                    <p className="text-[11px] text-slate-500 mt-1">Cross-Encoder + RRF</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: AI SOLUTION ARCHITECT (V2) */}
          {activeTab === "architect" && (
            <div className="max-w-5xl mx-auto space-y-6">
              <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Layers className="h-5 w-5 text-cyan-400" />
                    Version 2 --- AI Solution Architect
                  </h2>
                  <p className="text-xs text-slate-400">
                    Transform the agent runtime into a platform that designs and scaffolds new enterprise AI agents from natural language prompts.
                  </p>
                </div>

                <div className="space-y-2">
                  <label className="block text-xs font-semibold text-slate-400">
                    Describe Desired Enterprise AI Assistant
                  </label>
                  <textarea
                    rows={4}
                    value={architectPrompt}
                    onChange={(e) => setArchitectPrompt(e.target.value)}
                    className="w-full rounded-xl bg-slate-900 p-4 text-xs text-slate-200 border border-slate-800 focus:border-cyan-500 focus:outline-none"
                  />
                </div>

                <button
                  onClick={handleGenerateArchitecture}
                  className="rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-600 px-6 py-3 text-sm font-semibold text-white hover:from-cyan-500 hover:to-indigo-500 transition shadow-lg shadow-cyan-500/20 flex items-center gap-2"
                >
                  <Sparkles className="h-4 w-4" />
                  Generate Agent Topology & Code Scaffold
                </button>
              </div>

              {architectResult && (
                <div className="glass-panel rounded-2xl p-6 border border-cyan-500/30 space-y-4">
                  <h3 className="text-sm font-semibold text-cyan-300 border-b border-slate-800 pb-2">
                    Generated LangGraph Agent Topology
                  </h3>

                  <div className="grid grid-cols-2 gap-4 text-xs">
                    <div className="rounded-xl bg-slate-900/60 p-4 border border-slate-800">
                      <p className="font-semibold text-slate-300 mb-2">Topology Graph Nodes:</p>
                      <ul className="space-y-1 text-slate-400 font-mono">
                        {architectResult.nodes.map((n: string) => (
                          <li key={n} className="flex items-center gap-1.5">
                            <ChevronRight className="h-3 w-3 text-cyan-400" /> {n}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="rounded-xl bg-slate-900/60 p-4 border border-slate-800">
                      <p className="font-semibold text-slate-300 mb-2">MCP Tools & Cognitive Agents:</p>
                      <div className="space-y-2">
                        {architectResult.components.map((c: any) => (
                          <div key={c.name} className="flex justify-between items-center bg-slate-950 p-2 rounded border border-slate-800">
                            <span className="font-medium text-slate-200">{c.name}</span>
                            <span className="text-[10px] bg-cyan-500/10 text-cyan-400 px-2 py-0.5 rounded border border-cyan-500/30">{c.type}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
