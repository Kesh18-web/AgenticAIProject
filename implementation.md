# Enterprise AI Analyst

## Final Engineering Blueprint (V1 & V2)

> **Vision**
>
> Build a flagship Enterprise AI application that demonstrates **AI
> Systems Engineering** rather than "another RAG chatbot".
>
> The product is **Enterprise AI Analyst**.
>
> Internally it is powered by a **modular AI runtime** built around
> reasoning agents, infrastructure components and MCP-powered external
> capabilities.

------------------------------------------------------------------------

# Core Design Philosophy

There are **three kinds of components**.

## 🧠 Cognitive Agents (They Think)

These components use LLM reasoning and make decisions.

1.  Planner Agent
2.  Analysis Agent
3.  Reflection Agent
4.  Judge Agent
5.  Model Router Agent
6.  Guardrail Agent

------------------------------------------------------------------------

## 🔧 Infrastructure Components (They Execute)

These do not reason.

They execute deterministic logic.

-   Query Rewriter
-   Hybrid Retrieval
-   Dense Retrieval
-   BM25
-   Reciprocal Rank Fusion (RRF)
-   Cross Encoder Reranker (MANDATORY)
-   Metadata Filtering
-   Parent/Child Retrieval
-   Context Builder
-   Context Compression
-   Citation Engine
-   Semantic Memory
-   Redis Cache
-   Qdrant
-   Firestore
-   LangSmith

------------------------------------------------------------------------

## 🌍 External Capabilities

Accessed ONLY through MCP.

Examples

-   GitHub
-   Filesystem
-   Browser
-   SQL
-   Slack
-   Jira
-   Confluence (V2)

------------------------------------------------------------------------

# Version 1 --- Enterprise AI Analyst

## Business Problem

Enterprise employees spend hours reading policies, audit reports,
architecture documents, contracts and technical documentation.

Enterprise AI Analyst behaves like an experienced consultant that plans
the analysis, gathers evidence, reasons over it, critiques itself and
produces grounded, citation-backed reports.

------------------------------------------------------------------------

# End-to-End Workflow

User

↓

Frontend (Next.js)

↓

FastAPI

↓

LangGraph Runtime

↓

Planner Agent

↓

Model Router Agent

↓

Analysis Agent

↓

Uses Infrastructure Components

-   Query Rewriter
-   Hybrid Retrieval
-   BM25
-   Dense Retrieval
-   RRF
-   Cross Encoder Reranker
-   Metadata Filtering
-   Context Builder
-   Semantic Memory
-   MCP Tools (if required)

↓

Reflection Agent

↓

If confidence is low

↓

Planner Agent (Re-plan)

↓

Additional Retrieval

↓

Analysis Again

↓

Judge Agent

↓

Guardrail Agent

↓

Final Answer

↓

LangSmith Trace + Metrics

------------------------------------------------------------------------

# Cognitive Agents

## Planner Agent

Responsibilities

-   Intent understanding
-   Task decomposition
-   Dynamic graph generation
-   Decide whether RAG is needed
-   Decide whether MCP tools are required
-   Decide retrieval strategy
-   Decide evaluation strategy
-   Trigger replanning

------------------------------------------------------------------------

## Model Router Agent

Reasons about

-   Cost
-   Latency
-   Model capability
-   Fallback strategy

Examples

Simple summarisation → Gemini Flash

Fast response → Groq

Complex reasoning → GPT-4.1 / Claude Sonnet

Coding → GPT-4.1

------------------------------------------------------------------------

## Analysis Agent

Responsible for

-   Multi-step reasoning
-   Comparison
-   Compliance analysis
-   Report generation
-   Root-cause analysis
-   Evidence-grounded responses only

------------------------------------------------------------------------

## Reflection Agent

Responsible for

-   Detect hallucinations
-   Detect unsupported claims
-   Detect missing requirements
-   Estimate confidence
-   Decide whether more retrieval is required
-   Request Planner to re-plan

This is the primary source of agentic behaviour.

------------------------------------------------------------------------

## Judge Agent

LLM-as-a-Judge

Evaluates

-   Faithfulness
-   Groundedness
-   Completeness
-   Answer relevance
-   Citation coverage

Stores evaluation metrics.

------------------------------------------------------------------------

## Guardrail Agent

Responsible for

-   Prompt injection detection
-   PII detection
-   Output validation
-   Safety checks
-   Policy enforcement

Blocks unsafe responses.

------------------------------------------------------------------------

# Mandatory Infrastructure

-   LangGraph orchestration
-   Query rewriting
-   Hybrid Retrieval
-   Dense Retrieval
-   BM25
-   Reciprocal Rank Fusion
-   Cross Encoder Reranking
-   Metadata filtering
-   Context compression
-   Parent-child retrieval hooks
-   Citation engine
-   Semantic session memory
-   Redis cache
-   LangSmith tracing

------------------------------------------------------------------------

# MCP

MCP is NEVER used for communication between internal components.

It is ONLY used for external capabilities.

Planner decides when external tools are required.

------------------------------------------------------------------------

# Observability

Track

-   Planner decisions
-   Execution graph
-   Retrieval scores
-   Reranker scores
-   Tool calls
-   Reflection count
-   Evaluation scores
-   Latency
-   Token usage
-   Cost
-   Cache hit rate

------------------------------------------------------------------------

# Version 2 --- AI Solution Architect

## Goal

Transform the runtime into a platform that can DESIGN and SCAFFOLD
entirely new enterprise AI agents from natural language.

This is NOT another chatbot.

It is an AI Architect.

------------------------------------------------------------------------

## Workflow

Developer describes an application.

Example

"I need a Legal Compliance Assistant that reads Confluence, searches
GitHub, answers policy questions, remembers previous conversations and
posts reports to Slack."

↓

Planner Agent analyses requirements.

↓

Planner generates

-   Functional requirements
-   LangGraph topology
-   Agent architecture
-   Retrieval strategy
-   Memory strategy
-   Evaluation strategy
-   Guardrail strategy
-   Model routing policy
-   MCP integrations
-   Deployment architecture

↓

Runtime assembles the agent using reusable modules.

↓

Generate

-   LangGraph graph
-   Project scaffold
-   Config files
-   Prompt templates
-   Tool configuration
-   Deployment configuration

↓

Developer reviews

↓

Deploy

------------------------------------------------------------------------

# Why Version 2 is Different

Version 1

-   Solves ONE enterprise problem exceptionally well.
-   Fixed product.
-   Runtime is internal.
-   Demonstrates production AI engineering.

Version 2

-   Runtime becomes the product.
-   Builds new enterprise AI applications.
-   Planner becomes an AI Solution Architect.
-   Runtime becomes reusable across HR, Legal, Finance, Compliance,
    DevOps and Customer Support agents.

------------------------------------------------------------------------

# Tech Stack

Frontend - Next.js - React - TailwindCSS

Backend - FastAPI - LangGraph - LangChain

Models - GPT-4.1 - Claude Sonnet - Gemini Flash - Groq

Vector Store - Qdrant

Memory - Redis + Vector Memory

Database - Firebase Firestore

Tracing - LangSmith

Deployment - Docker

------------------------------------------------------------------------

# Resume Objective

The interviewer should naturally ask:

-   Why separate Agents from Infrastructure?
-   Why LangGraph?
-   Why Reflection instead of a linear workflow?
-   Why Hybrid Retrieval?
-   Why RRF?
-   Why Cross Encoder Reranking?
-   Why MCP?
-   Why model routing?
-   Why LLM-as-a-Judge?
-   Why Guardrails?
-   Why semantic memory?
-   Why dynamic planning?

If the interview shifts toward these questions instead of "How did you
build a chatbot?", the project has achieved its goal.
