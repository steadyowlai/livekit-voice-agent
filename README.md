# livekit-voice-agent

A real-time voice AI assistant pipeline integrating **LiveKit Agents**, **LangGraph**, **LangChain**, and **FastMCP**.

---

## Executive Summary (For Engineering Managers)

This project demonstrates how to build a production-grade conversational banking agent by composing four specialized AI frameworks together:

1. **LiveKit Agents (Voice Orchestration):** Manages real-time WebRTC audio streaming, Voice Activity Detection (VAD), Speech-to-Text (STT), Text-to-Speech (TTS), and conversational turn-taking with ultra-low latency.
2. **LiveKit TaskGroups (Multi-Turn Intake State Machine):** Replaces messy single-prompt data collection with structured, sequential intake stages (`LoanRequestTask` → `FinancialProfileTask`) to reliably collect user financials over voice without confusion.
3. **LangGraph (Deterministic Credit Underwriting):** Eliminates LLM math hallucinations by delegating underwriting calculations, ratio analysis, risk tiering, and approval decisions to a pure, deterministic 3-node computational state graph.
4. **LangChain (Tool Reusability & Adapters):** Bridges existing enterprise LangChain tools (market interest rate and Fed policy web search) directly into LiveKit using a generic adapter that runs synchronous network I/O in worker threads without stalling real-time audio.
5. **FastMCP Server (Decoupled Enterprise Knowledge):** Exposes bank product catalogs, customer account data, and compliance guidelines over the Model Context Protocol (SSE transport), maintaining a persistent session connection across all agent tasks.

---

## System Architecture

```
                                ┌────────────────────────┐
                                │   Client Audio Stream  │
                                └───────────┬────────────┘
                                            │ WebRTC
                                            ▼
                                ┌────────────────────────┐
                                │      LiveKit Agent     │
                                │  • STT: AssemblyAI     │
                                │  • VAD: Silero         │
                                │  • LLM: OpenAI GPT-4.1 │
                                │  • TTS: Cartesia       │
                                └───────────┬────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        │                                   │                                   │
        ▼                                   ▼                                   ▼
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│   TaskGroup & Graph   │       │   LangChain Adapter   │       │   FastMCP Server      │
├───────────────────────┤       ├───────────────────────┤       ├───────────────────────┤
│ • TaskGroup Intake    │       │ • adapt_langchain_tool│       │ • Transport: SSE      │
│   - LoanRequestTask   │       │ • DuckDuckGo search   │       │ • Port: 8000          │
│   - FinancialProfile  │       │ • Non-blocking async  │       │ • Endpoints/Tools:    │
│ • LangGraph StateGraph│       │   via thread pool     │       │   - list_products     │
│   - calculate_ratios  │       └───────────────────────┘       │   - fetch_policy      │
│   - eval_credit_risk  │                                       │   - get_profile       │
│   - underwrite_decide │                                       │   - read_guidelines   │
│ • Native EMI Tool     │                                       └───────────────────────┘
└───────────────────────┘
```

---

## Core Subsystems Explained

### 1. LiveKit TaskGroups: Structured Voice Intake
**Why it is needed:**
Collecting 5 different financial numbers in a single open-ended voice conversation often causes LLMs to miss fields, ask redundant questions, or hallucinate. LiveKit's `TaskGroup` pattern turns the application intake into a clean, sequential state machine.

**How it works:**
1. When the caller asks to apply for a loan, the main agent invokes `evaluate_loan_underwriting`.
2. Control temporarily transitions to **Stage 1 (`LoanRequestTask`)**, which proactively asks for the loan amount and property value. Once collected, `record_loan_request` completes Stage 1.
3. The `TaskGroup` immediately advances to **Stage 2 (`FinancialProfileTask`)**, which asks for monthly income, debt payments, and credit score, calling `record_financial_profile`.
4. When both tasks finish, the aggregated data is handed off to LangGraph, and the final underwriting verdict is returned to the main agent to announce once.

```
[Main Assistant] ──calls──> evaluate_loan_underwriting()
                                 │
                                 ▼
                          ┌──────────────┐
                          │  TaskGroup   │
                          └──────┬───────┘
                                 │
       ┌─────────────────────────┴─────────────────────────┐
       ▼                                                   ▼
 Stage 1: LoanRequestTask                   Stage 2: FinancialProfileTask
 - Prompts: loan amount & property value    - Prompts: income, debt, credit score
 - Completes & cleanly yields               - Completes & cleanly yields
       │                                                   │
       └─────────────────────────┬─────────────────────────┘
                                 ▼
                    LangGraph Underwriting Graph
```

---

### 2. LangGraph StateGraph: Deterministic Underwriting Engine
**Why it is needed:**
Underwriting requires strict adherence to regulatory rules, Debt-to-Income (DTI) thresholds, Loan-to-Value (LTV) limits, and accurate 30-year fixed loan amortization formulas. Leaving this to an LLM risks mathematical errors and compliance violations.

**How it works:**
1. We build and compile a pure computational `StateGraph` in `langgraph/stategraph.py`.
2. We import this graph into the LiveKit agent tools (`livekit_agent/tools/loan_task.py`) and invoke it via `underwriting_graph.ainvoke(initial_state)` upon intake completion.
3. The graph executes 3 sequential compute nodes:
   - **`calculate_ratios_node`**: Computes `DTI = (monthly_debt / monthly_income) * 100` and `LTV = (loan_amount / property_value) * 100`.
   - **`evaluate_credit_risk_node`**: Categorizes the credit score into risk tiers:
     - Tier 1 (Prime): Credit Score >= 740, Base Rate = 6.25%
     - Tier 2 (Standard): Credit Score >= 680, Base Rate = 6.75%
     - Tier 3 (Near-Prime): Credit Score >= 620, Base Rate = 7.50%
     - Tier 4 (Subprime): Credit Score < 620, Base Rate = 9.25%
   - **`underwrite_decision_node`**: Evaluates approval rules (`DTI <= 45%`, `Credit Score >= 620`), adds PMI requirements and rate adjustment (+0.25%) if `LTV > 80%` or `DTI > 38%`, computes exact monthly payments via 30-year fixed amortization formula, and generates a structured verdict.

```python
# State Schema
class UnderwritingState(TypedDict):
    loan_amount: float
    property_value: float
    monthly_income: float
    monthly_debt: float
    credit_score: int
    dti_ratio: float
    ltv_ratio: float
    credit_tier: str
    base_interest_rate: float
    approval_status: str
    final_interest_rate: float | None
    estimated_monthly_payment: float | None
    underwriting_notes: str
    summary: str
```

---

### 3. LangChain Tool Adapter: Reusable Enterprise Tools
**Why it is needed:**
Enterprises often have established tool repositories written in LangChain (such as web scrapers, vector search, or database queries). Rather than rewriting these tools specifically for LiveKit, we build a generic adapter to import them directly into the voice agent.

**How it works:**
1. In `langchain/tools.py`, we define standard synchronous LangChain `@tool` functions (`search_market_rates`, `search_fed_policy`).
2. In `livekit_agent/tools/adapted_tools.py`, we implement `adapt_langchain_tool(lc_tool)`:
   - `@functools.wraps(lc_tool.func)` copies the function signature, argument types, and docstrings so LiveKit automatically generates the OpenAI/Gemini JSON tool schema for the LLM.
   - `asyncio.to_thread` runs the synchronous web search on a background worker thread so live HTTP calls never freeze the real-time audio pipeline or Speech-to-Text processing.
3. Existing LangChain tools are adapted in single clean statements:
   ```python
   adapted_search_market_rates = adapt_langchain_tool(search_market_rates)
   adapted_search_fed_policy = adapt_langchain_tool(search_fed_policy)
   ```

---

### 4. FastMCP Server: Decoupled Knowledge & Policy Layer
**Why it is needed:**
Banking product offerings, interest rate policies, and customer records should live in an independent service rather than being hardcoded into the voice bot. Anthropic's Model Context Protocol (MCP) provides a standardized protocol for exposing these tools.

**How it works:**
1. A FastMCP server (`mcp/mcp_server.py`) runs as a standalone service on `http://127.0.0.1:8000/sse` using Server-Sent Events (SSE).
2. In `livekit_agent/main.py`, the MCP server is registered at the **session level** using `tools=[MCPToolset(id="bank_mcp", mcp_server=MCPServerHTTP(...))]`.
3. Because the toolset is session-scoped, the connection stays open permanently throughout the call, surviving sub-agent handoffs and task transitions.
4. The server exposes:
   - `list_available_loan_products`: Retail & commercial loan catalogs with interest rate ranges and borrowing limits.
   - `fetch_bank_policy`: Policy limits, minimum credit scores, and down payment requirements.
   - `get_customer_profile`: Account verification and pre-approval status.
   - `read_lending_guidelines`: Regulatory compliance and closing document checklists.

---

### 5. Native EMI Calculator Tool: Instant Ad-Hoc Math
**Why it is needed:**
When customers ask quick hypothetical questions during a call (e.g. *"What would my payment be for $250k at 6.5% over 20 years?"*), the agent needs a fast, direct calculation tool that runs locally without invoking external services.

**How it works:**
1. Implemented in `livekit_agent/tools/emi_calculator.py` as a native `@llm.function_tool`.
2. Takes `principal`, `interest_rate`, and `tenure_years`, validates inputs, applies the standard amortization formula, and returns monthly payment, total interest, and total cost in milliseconds.

---

## Directory Structure

```
├── livekit_agent.py              # CLI entrypoint
├── livekit_agent/
│   ├── main.py                   # Worker setup, pipeline initialization, session MCP config
│   ├── assistant.py              # Assistant agent definition and tool binding
│   └── tools/
│       ├── __init__.py           # Unified tool exports
│       ├── loan_task.py          # TaskGroup intake flow and LangGraph evaluation
│       ├── emi_calculator.py     # Native loan amortization calculator
│       ├── adapted_tools.py      # Standard LangChain-to-LiveKit tool adapter
│       └── underwriting.py       # LangGraph invocation wrapper
├── langchain/
│   └── tools.py                  # LangChain search tools (market rates, Fed policy)
├── langgraph/
│   └── stategraph.py             # 3-node Underwriting StateGraph
├── mcp/
│   ├── mcp_server.py             # FastMCP SSE server
│   ├── db.json                   # Mock banking database
│   └── bank_guidelines.txt       # Policy documentation
└── pyproject.toml                # Project configuration and dependencies
```

---

## Setup & Installation

### Requirements
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- LiveKit Cloud credentials (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`)

### Installation
```bash
git clone <repo-url>
cd livekit-voice-agent
uv sync
```

### Environment Configuration
```bash
cp .env.example .env
```
Populate `.env` with required API keys:
```env
LIVEKIT_URL=wss://...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

---

## Running the Application

### 1. Start the MCP Server
In your first terminal, start the FastMCP server:
```bash
uv run mcp/mcp_server.py
```
*Server runs on `http://127.0.0.1:8000/sse`.*

### 2. Start the LiveKit Voice Agent Worker
In a second terminal, launch the agent worker:

**Console Mode (Local audio testing via microphone & speaker):**
```bash
uv run livekit_agent.py console
```

**Dev Mode (Connects to LiveKit Cloud Room via WebRTC):**
```bash
uv run livekit_agent.py dev
```

---

## Verification & Test Scenarios

| Capability | Sample User Voice Prompt | Underlying Subsystem & Execution Flow | Expected Outcome |
|---|---|---|---|
| **Loan Underwriting** | *"I'd like to apply for a loan."* | `evaluate_loan_underwriting` → `TaskGroup` (Stages 1 & 2) → LangGraph | Sequentially collects loan amount, property value, income, debt, and credit score; executes LangGraph; speaks approval status, rate, and monthly payment. |
| **Market Rates** | *"What are current 30-year mortgage rates?"* | `adapted_search_market_rates` (LangChain Adapter) | Runs LangChain DuckDuckGo search in background worker thread; returns trimmed rate summary without audio stutter. |
| **Product Inquiries** | *"What commercial loan options do you offer?"* | `list_available_loan_products` (FastMCP SSE) | Queries FastMCP server over SSE; returns commercial real estate and equipment loan terms. |
| **Loan Calculations** | *"Calculate monthly payment for $200k at 6% over 30 years."* | `calculate_loan_emi` (Native Tool) | Computes exact amortization formula and returns monthly payment + total interest. |
