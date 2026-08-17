# livekit-voice-agent

A real-time voice AI assistant pipeline integrating **LiveKit Agents**, **LangGraph**, **LangChain**, and **FastMCP**.

---

## Summary

This project demonstrates how to build a production-grade conversational banking agent by composing four specialized AI frameworks together:

1. **LiveKit Agents (Voice Orchestration):** Manages real-time WebRTC audio streaming, Voice Activity Detection (VAD), Speech-to-Text (STT), Text-to-Speech (TTS), and conversational turn-taking with ultra-low latency.
2. **LiveKit TaskGroups (Multi-Turn Intake State Machine):** Replaces messy single-prompt data collection with structured, sequential intake stages (`LoanRequestTask` → `FinancialProfileTask`) to reliably collect user financials over voice without confusion.
3. **LangGraph 1: Background Underwriting Tool:** Eliminates LLM math hallucinations by delegating underwriting calculations, ratio analysis, risk tiering, and approval decisions to a pure, deterministic computational state graph running as a background tool.
4. **LangGraph 2: LLMAdapter Voice Recommender:** Completely takes over the voice interaction to run a dynamic, multi-turn decision tree (Loan Recommender) using `langchain.LLMAdapter` and dual-mode (`messages` and `values`) stream state monitoring.
5. **LangChain (Tool Reusability & Adapters):** Bridges existing enterprise LangChain tools (market interest rate and Fed policy web search) directly into LiveKit using a generic adapter that runs synchronous network I/O in worker threads without stalling real-time audio.
6. **FastMCP Server (Decoupled Enterprise Knowledge):** Exposes bank product catalogs, customer account data, and compliance guidelines over the Model Context Protocol (SSE transport), maintaining a persistent session connection across all agent tasks.

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
                                │  • STT: Deepgram Nova3 │
                                │  • VAD: Silero         │
                                │  • LLM: OpenAI GPT-4.1 │
                                │  • TTS: Cartesia       │
                                └───────────┬────────────┘
                                            │
        ┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
        │                         │                         │                         │
        ▼                         ▼                         ▼                         ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│  Underwriting Graph   │ │   Recommender Graph   │ │   LangChain Adapter   │ │    FastMCP Server     │
├───────────────────────┤ ├───────────────────────┤ ├───────────────────────┤ ├───────────────────────┤
│ • Background Tool     │ │ • Voice Sub-Agent     │ │ • adapt_langchain_tool│ │ • Transport: SSE      │
│ • TaskGroup Intake    │ │ • langchain.LLMAdapter│ │ • DuckDuckGo search   │ │ • Port: 8000          │
│ • StateGraph          │ │ • Custom llm_node     │ │ • Non-blocking async  │ │ • Endpoints/Tools:    │
│   - calculate_ratios  │ │ • StateGraph          │ │   via thread pool     │ │   - list_products     │
│   - eval_credit_risk  │ │   - classify_input    │ └───────────────────────┘ │   - fetch_policy      │
│   - underwrite_decide │ │   - ask_purpose       │                           │   - get_profile       │
└───────────────────────┘ │   - ask_down_payment  │                           │   - read_guidelines   │
                          │   - recommend_loan    │                           └───────────────────────┘
                          └───────────────────────┘
```

---

## Implementation Patterns

### Pattern 1: Using Native Tools
When a simple, stateless utility function is needed (like math), a native `@llm.function_tool` is the fastest and most direct approach.

**How to trigger it:**
Ask the agent: *"What would my monthly payment be for a $250,000 loan at 6.5% interest over 30 years?"*

**How it works:**
1. Implemented in `livekit_agent/tools/emi_calculator.py` as a native `@llm.function_tool`.
2. Takes `principal`, `interest_rate`, and `tenure_years`, validates inputs, applies the standard amortization formula, and returns monthly payment, total interest, and total cost in milliseconds.

---

### Pattern 2: Using Adapted LangChain Tools
Enterprises often have established tool repositories written in LangChain. Rather than rewriting these tools specifically for LiveKit, we build a generic adapter to import them directly into the voice agent.

**How to trigger it:**
Ask the agent: *"What are the current market interest rates?"* or *"What did the Fed say about interest rates recently?"*

**How it works:**
1. In `langchain/tools.py`, we define standard synchronous LangChain `@tool` functions (`search_market_rates`, `search_fed_policy`).
2. In `livekit_agent/tools/adapted_tools.py`, we implement `adapt_langchain_tool(lc_tool)`:
   - `@functools.wraps(lc_tool.func)` copies the function signature, argument types, and docstrings so LiveKit automatically generates the JSON tool schema for the LLM.
   - `asyncio.to_thread` runs the synchronous web search on a background worker thread so live HTTP calls never freeze the real-time audio pipeline or Speech-to-Text processing.
3. Existing LangChain tools are adapted in single clean statements:
   ```python
   adapted_search_market_rates = adapt_langchain_tool(search_market_rates)
   adapted_search_fed_policy = adapt_langchain_tool(search_fed_policy)
   ```

---

### Pattern 3: Using MCP (Model Context Protocol)
Banking product offerings, interest rate policies, and customer records should live in an independent service rather than being hardcoded into the voice bot. Anthropic's Model Context Protocol (MCP) provides a standardized protocol for exposing these tools.

**How to trigger it:**
Ask the agent: *"What commercial loan products do you offer?"* or *"Can you read the lending guidelines?"*

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

### Pattern 4: Using LangGraph as a Tool with TaskGroup
Collecting complex financial data in a single open-ended voice conversation often causes LLMs to miss fields or hallucinate math. We solve this by using LiveKit's `TaskGroup` to enforce a sequential voice intake interview, and then passing that data into a deterministic LangGraph tool to do the heavy underwriting math.

**How to trigger it:**
Tell the agent: *"I want to apply for a loan."* 
*(After it gives the result, test the revision flow by saying: "Wait, my credit score is actually 750".)*

**How it works:**
1. **The Intake Interview (`TaskGroup`):** When the caller asks to apply for a loan, the agent invokes `evaluate_loan_underwriting`. Control temporarily transitions to **Stage 1 (`LoanRequestTask`)**, which proactively asks for the loan amount and property value. The `TaskGroup` then advances to **Stage 2 (`FinancialProfileTask`)**, which asks for monthly income, debt payments, and credit score.
2. **The Underwriting Engine (`LangGraph`):** When both tasks finish, the aggregated data is handed off to a pure computational `StateGraph` (`langgraph/stategraph.py`). It calculates `DTI` and `LTV`, categorizes credit risk into tiers, evaluates approval rules, and generates a structured verdict. The final verdict is returned to the main agent to announce once.
3. **Session Caching & Revision:** The final state is cached in memory (`SessionStateCache`). If the user later corrects a single value (e.g., *"Wait, my credit score is 750"*), the LLM calls `revise_loan_underwriting(credit_score=750)`. The tool uses the cached `userdata` as a baseline, patches only the provided value, and re-runs the LangGraph engine instantly—skipping the multi-stage intake interview entirely.

```text
[Main Assistant]
       │
       ├─[Initial Flow] ─────────┐
       │                         ▼
       │             evaluate_loan_underwriting()
       │                         │
       │                  ┌──────▼──────┐
       │                  │  TaskGroup  │
       │                  └──────┬──────┘
       │       ┌─────────────────┴─────────────────┐
       │       ▼                                   ▼
       │    Stage 1: LoanRequestTask      Stage 2: FinancialProfileTask
       │    - Prompts: loan/property      - Prompts: income/debt/credit
       │    - Completes                     - Completes 
       │       │                                   │
       │       └─────────────────┬─────────────────┘
       │                         │ (Data saved to context.userdata.loan.last_underwriting_state)
       │                         │
       └─[Revision Flow] ────────│────────┐
                                 │        ▼
                                 │    revise_loan_underwriting()
                                 │        │ (Bypasses TaskGroup loan interview, patches context.userdata.loan.last_underwriting_state)
                                 │        │
                                 ▼        ▼
                      LangGraph Underwriting Graph
```

---

### Pattern 5: Using LangGraph as an LLM with LLMAdapter and TaskGroup
We use LangGraph to completely drive complex, multi-stage voice conversations (rather than just running it as a background tool). We are doing this to enforce strict decision-tree flows—such as a loan type recommender—that ask branching questions and route the user down different dialogue paths based on their real-time answers.

**How to trigger it:**
Tell the agent: *"Help me choose a loan type."* The agent will hand off the microphone to the LangGraph recommender until you finish the flow.

**How it works:**
1. We use the official `langchain.LLMAdapter` from `livekit-plugins-langchain` to wrap our LangGraph as a LiveKit-compatible LLM. The adapter converts LiveKit's persistent `ChatContext` history into standard LangChain message types on every turn.
2. The loan recommender graph is built as a custom `StateGraph` that evaluates the entire conversation history statelessly. A `classify_input` node runs first to extract `loan_purpose` and `down_payment_percent`, allowing the graph to dynamically heal from interruptions.
3. Our `LoanRecommenderTask` sub-agent overrides `llm_node` to stream directly from the graph using `astream(stream_mode=["messages", "values"])`. It filters for `AIMessageChunk` instances (preventing raw JSON tool calls from being spoken) while simultaneously listening to the `values` stream to detect when the graph reaches a termination state (`wants_to_exit`).
4. When the user says *"Help me choose a loan type"*, the main OpenAI agent calls `recommend_home_loan()`, which yields control using a `TaskGroup` to safely hand off the microphone to the LangGraph-powered agent until the session is complete.

```text
[Main Assistant (OpenAI)]
        │
        │ User: "What loan should I get?"
        │ → calls recommend_home_loan()
        │ → TaskGroup pauses Main Assistant & starts LoanRecommenderTask
        │
        ▼
[LoanRecommenderTask (langchain.LLMAdapter + llm_node override)]
        │
        ├── "Is this for a purchase or renovation?"
        │       │
        │       ├── "purchase" → "What % down payment?"
        │       │                    │
        │       │               ├── < 20% → Recommend FHA Loan
        │       │               └── >= 20% → Recommend Conventional
        │       │
        │       └── "renovation" → Recommend HELOC
        │
        │ User: "I have no more questions." → wants_to_exit: True
        │ → self.complete(None)
        │
        ▼
[Control returns to Main Assistant]
```

---

## Directory Structure

```
├── .env.example                  # Template for environment variables
├── livekit_agent.py              # CLI entrypoint
├── livekit_agent/
│   ├── main.py                   # Worker setup and pipeline initialization
│   ├── assistant.py              # Assistant agent definition and tool binding
│   ├── session_state.py          # Per-caller session state dataclass (userdata)
│   └── tools/
│       ├── __init__.py           # Unified tool exports
│       ├── loan_task.py          # TaskGroup intake flow and LangGraph evaluation
│       ├── loan_recommender_task.py  # LangGraph LLM adapter agent handoff
│       ├── emi_calculator.py     # Native loan amortization calculator
│       └── adapted_tools.py      # Standard LangChain-to-LiveKit tool adapter
├── langchain/
│   └── tools.py                  # LangChain search tools (market rates, Fed policy)
├── langgraph/
│   ├── stategraph.py             # 3-node Underwriting StateGraph
│   └── loan_recommender_graph.py # Conversational loan type recommender graph
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
| **Underwriting Revision** | *"Actually, my income is $25k, not $20k."* | `revise_loan_underwriting` → LangGraph | Patches the cached state with the new income and instantly re-runs the underwriting graph without restarting the interview. |
| **Market Rates** | *"What are current 30-year mortgage rates?"* | `adapted_search_market_rates` (LangChain Adapter) | Runs LangChain DuckDuckGo search in background worker thread; returns trimmed rate summary without audio stutter. |
| **Product Inquiries** | *"What commercial loan options do you offer?"* | `list_available_loan_products` (FastMCP SSE) | Queries FastMCP server over SSE; returns commercial real estate and equipment loan terms. |
| **Loan Calculations** | *"Calculate monthly payment for $200k at 6% over 30 years."* | `calculate_loan_emi` (Native Tool) | Computes exact amortization formula and returns monthly payment + total interest. |
| **Loan Type Recommender** | *"What type of loan should I get?"* | `recommend_home_loan` → `session.update_agent(LoanRecommenderAgent)` → LangGraph LLM Adapter | Hands off to a LangGraph-powered agent that asks purpose and down payment, then recommends FHA, Conventional, or HELOC. |
