# First Apex National Bank — LiveKit Voice AI Agent

An enterprise-grade, real-time conversational voice AI assistant for banking and loan services, powered by **LiveKit Agents** and integrated with a **FastMCP Server** over Server-Sent Events (SSE).

---

## Architecture Overview

```
                      ┌───────────────────────────────────────┐
                      │          Caller Voice Audio           │
                      └──────────────────┬────────────────────┘
                                         │ WebRTC
                                         ▼
                      ┌───────────────────────────────────────┐
                      │         LiveKit Voice Pipeline        │
                      │  • STT: AssemblyAI / Deepgram         │
                      │  • VAD: Silero + Semantic Turn Model  │
                      │  • LLM: OpenAI GPT-4.1 / Gemini       │
                      │  • TTS: Cartesia Sonic / Inworld      │
                      └──────────────────┬────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │    Local Function Tool    │                   │   FastMCP Server (SSE)    │
   │  • calculate_loan_emi     │                   │   http://127.0.0.1:8000   │
   │    (Calculates exact      │                   ├───────────────────────────┤
   │     monthly repayment,    │                   │ • list_available_loan_... │
   │     interest & total cost)│                   │ • fetch_bank_policy       │
   └───────────────────────────┘                   │ • get_customer_profile    │
                                                   │ • read_lending_guidelines │
                                                   └───────────────────────────┘
```

---

## Features

- **Decoupled Knowledge & Policy Engine (MCP):** Dynamic retrieval of loan products, interest rate policies, customer account profiles, and compliance disclosures via Model Context Protocol.
- **Real-Time Mathematical Accuracy:** Native LiveKit function tool computes amortization schedules, monthly payments (EMI), and total loan costs on the fly.
- **Enterprise Multi-Provider Fallbacks:**
  - **LLM:** OpenAI `gpt-4.1-mini` with automatic failover to Google `gemini-2.5-flash`.
  - **STT:** AssemblyAI `universal-streaming` with failover to Deepgram `nova-3`.
  - **TTS:** Cartesia `sonic-3` with failover to Inworld TTS.
- **Advanced Voice Interaction:**
  - Silero VAD + Multilingual semantic turn detection for natural interruptions.
  - Background voice cancellation (BVC) and preemptive audio generation.
  - Comprehensive turn metrics (TTFT, TTFB, EOU delay, token counts, and durations).

---

## Project Structure

```
├── livekit_agent.py          # Main LiveKit voice pipeline and assistant logic
├── mcp/
│   ├── mcp_server.py         # FastMCP Server running on SSE transport
│   ├── db.json               # Mock banking database (rates, products, accounts)
│   └── bank_guidelines.txt   # Lending guidelines & compliance disclosures
├── pyproject.toml            # Project dependencies managed by uv
├── .env.example              # Environment variables template
└── README.md                 # Project documentation
```

---

## Prerequisites

- Python 3.12+ (or 3.13)
- [uv](https://github.com/astral-sh/uv) package manager
- API Keys:
  - [LiveKit Cloud](https://cloud.livekit.io/) (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`)
  - OpenAI (`OPENAI_API_KEY`)
  - Google Gemini (`GOOGLE_API_KEY`)
  - AssemblyAI (`ASSEMBLYAI_API_KEY`)
  - Deepgram (`DEEPGRAM_API_KEY`)
  - Cartesia (`CARTESIA_API_KEY`)

---

## Setup & Installation

1. **Clone the repository and navigate into the folder:**
   ```bash
   git clone <repo-url>
   cd livekit-voice-agent
   ```

2. **Install dependencies using `uv`:**
   ```bash
   uv sync
   ```

3. **Configure Environment Variables:**
   Create a `.env` file from the template:
   ```bash
   cp .env.example .env
   ```
   Fill in your API keys in `.env`.

---

## Running the Application

### Step 1: Start the MCP Server
In your first terminal, start the FastMCP policy server over SSE:
```bash
uv run mcp/mcp_server.py
```
*The server will start listening on `http://127.0.0.1:8000/sse`.*

### Step 2: Start the LiveKit Voice Agent
In a second terminal, launch the voice agent in local console mode:
```bash
uv run livekit_agent.py console
```

---

## Sample Voice Prompts to Try

- **Product Catalog Inquiries:**
  > *"What retail loan products do you offer?"*
  > *(Queries MCP tool: `list_available_loan_products`)*

- **Policy Lookups:**
  > *"What is the standard interest rate and down payment required for a retail personal loan?"*
  > *(Queries MCP tool: `fetch_bank_policy`)*

- **Loan Calculations:**
  > *"If I borrow $250,000 for a home loan at 6.5% interest over 20 years, what will my monthly payment be?"*
  > *(Executes native tool: `calculate_loan_emi`)*

- **Customer Profile & Compliance:**
  > *"Can you check account ACC-101 and tell me what documents I need to bring for my loan closing?"*
  > *(Queries MCP tools: `get_customer_profile` and `read_lending_guidelines`)*
