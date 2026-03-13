# PitchPal v2 — AI-Powered Startup Pitch Evaluator

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Gemini](https://img.shields.io/badge/Gemini-AI-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade AI agent that evaluates startup pitches using a **custom ReAct (Reasoning + Acting) loop** — no LangChain, no frameworks. Built from scratch with real-time WebSocket streaming, multi-provider LLM support, and live web research via Tavily.

> **v2 is a complete rebuild.** v1 used LangChain + Streamlit + OpenAI. v2 replaces everything with a custom agent, FastAPI backend, and Next.js frontend with glassmorphism UI.

---

## What Makes This Different

| Feature | Generic AI Chatbots | PitchPal v2 |
|---|---|---|
| **Agent Architecture** | LangChain wrappers | Custom ReAct loop built from scratch |
| **Research** | No web access | Live Tavily search with domain filtering |
| **Scoring** | Inconsistent text output | Structured JSON with 5-7 scored dimensions |
| **Caching** | None | Semantic (embedding-based) + evaluation (hash-based) |
| **Streaming** | Wait for full response | Real-time WebSocket step-by-step streaming |
| **Deck Analysis** | Text only | PDF/PPTX upload with Gemini Vision OCR |
| **Reliability** | Raw LLM output | 5-strategy JSON repair + retry with fallback |

---

## Architecture

```
                                +-----------------------+
                                |   Next.js Frontend    |
                                |   (Glassmorphism UI)  |
                                +----------+------------+
                                           |
                                    WebSocket / REST
                                           |
                                +----------v------------+
                                |   FastAPI Backend     |
                                |                       |
                                |  +-- Security Layer --+-----> Input Sanitization
                                |  |   Rate Limiter     |-----> IP-based (3/5 per 24h)
                                |  |   Prompt Injection  |-----> 16+ pattern detection
                                |  +--------------------+
                                |                       |
                                |  +-- Cache Layer -----+-----> Evaluation Cache (SHA-256)
                                |  |                    |-----> Semantic Cache (embeddings)
                                |  +--------------------+
                                |                       |
                                |  +-- ReAct Agent -----+-----> 12-step budget
                                |  |   (Custom Loop)    |-----> Role-specific prompts
                                |  |                    |-----> 5-strategy JSON repair
                                |  +--------+-----------+
                                |           |           |
                                +-----------+-----------+
                                            |
                          +-----------------+------------------+
                          |                 |                  |
                  +-------v------+  +-------v------+  +-------v-------+
                  | Gemini / Groq|  | Tavily Search|  | Gemini Vision |
                  | LLM Provider |  | (4 tools)    |  | (PDF OCR)     |
                  +--------------+  +--------------+  +---------------+
```

---

## Features (29 Total)

### Custom ReAct Agent (No Frameworks)
- **Hand-built ReAct loop** with Thought → Action → Observation cycling
- **12-step budget** to prevent infinite loops and control costs
- **Role-specific evaluation**: Startup founder (5 dimensions) vs Investor (7 dimensions)
- **5-strategy JSON repair pipeline**: regex extraction, bracket fixing, truncation repair, trailing comma fix, markdown fence stripping
- **Retry with exponential backoff** + provider fallback on failure

### Multi-Provider LLM Abstraction
- **4 providers supported**: Gemini, Groq, OpenAI, Anthropic
- **Single config switch** via `LLM_PROVIDER` env var
- **Automatic fallback** if primary provider fails

### Live Web Research (Tavily Integration)
- **4 specialized search tools** with domain filtering:
  - `search_market_data` — market size, growth rates, industry reports
  - `search_competitor_info` — competitor analysis, market positioning
  - `search_industry_trends` — emerging trends, technology shifts
  - `search_financial_benchmarks` — funding rounds, revenue benchmarks
- **Semantic caching** of search results to avoid duplicate API calls

### Intelligent Caching System
- **Evaluation cache**: SHA-256 hash of (pitch + role) → deterministic scores for identical pitches
- **Semantic cache**: Gemini `text-embedding-001` embeddings (768-dim) with cosine similarity at 0.72 threshold
- **Similarity detection**: Warns when a pitch is similar to a previously evaluated one (0.87 threshold)
- **24-hour TTL** with disk persistence (survives server restarts)

### PDF/PPTX Deck Upload
- **PyMuPDF** for text extraction + slide rendering
- **Gemini Vision OCR** fallback for image-based PDFs (no text layer)
- **python-pptx** for PowerPoint file support
- **Deck quality analysis**: design, narrative, data visualization scores (0-10)
- **Automatic startup name detection** from slide content
- **20MB file size limit** with early validation

### Security & Input Protection
- **HTML/script stripping** — prevents XSS via pitch text
- **Prompt injection detection** — 16+ regex patterns catch jailbreak attempts
- **Unicode NFKD normalization** — prevents bypass with look-alike characters (Greek omicron, zero-width joiners)
- **Length enforcement** — min 50, max 5,000 characters
- **CORS hardening** — explicit methods and headers, no wildcards
- **Sanitized error messages** — no internal details leaked to clients

### Rate Limiting
- **IP-based** with role-specific limits: 3 evaluations/24h (startup), 5/24h (investor)
- **Cache hits bypass rate limits** — zero API cost, no reason to limit
- **Thread-safe** with `threading.Lock`
- **Auto-reset** on window expiry

### Real-Time WebSocket Streaming
- **Step-by-step agent streaming** — see each Thought, Action, and Observation live
- **Rate limit status** included in start message
- **Similar pitch warnings** streamed before evaluation begins
- **Graceful error handling** with WebSocket close

### Shareable Evaluation Links
- **Cryptographic share IDs** using `secrets.token_urlsafe(24)`
- **7-day TTL** with automatic expiry
- **View counter** tracking
- **FIFO eviction** at 1,000 entries

### Investor Mode
- **Access code authentication** with 6-hour session tokens
- **7-dimension analysis**: Market Opportunity, Revenue & Unit Economics, Scalability, Competitive Moat, Team & Execution, Risk Assessment, Exit Potential
- **vs. Startup mode** (5 dimensions): Problem Clarity, Market Opportunity, Business Model, Competitive Advantage, Team Strength

### Frontend (Next.js 16 + React 19)
- **Glassmorphism UI** with backdrop blur, translucent surfaces, floating orbs
- **Dark/Light theme** with OS preference detection and localStorage persistence
- **Framer Motion animations** — staggered fade-ins, card hover effects, modal transitions
- **Recharts data visualization** — radar charts, bar charts for dimension scores
- **PDF export** via html2canvas-pro + jsPDF
- **Evaluation history** stored in localStorage
- **Pitch comparison** — side-by-side analysis of multiple evaluations
- **Responsive design** — mobile, tablet, desktop breakpoints

### Observability
- **Structured NDJSON logging** with event types, timestamps, and context
- **In-memory metrics**: latency percentiles (p50/p95/p99), cache hit rate, error rate
- **Live `/metrics` and `/stats` endpoints** for monitoring

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Next.js 16, React 19, TypeScript 5 | App Router, SSR, type safety |
| **Styling** | Tailwind CSS v4, Framer Motion | Glassmorphism UI, animations |
| **Charts** | Recharts | Radar + bar chart visualizations |
| **Export** | html2canvas-pro, jsPDF | PDF report generation |
| **Backend** | FastAPI, Uvicorn | REST API + WebSocket server |
| **LLM** | Gemini, Groq, OpenAI, Anthropic | Multi-provider abstraction |
| **Search** | Tavily | Live web research with domain filtering |
| **PDF** | PyMuPDF, python-pptx | Deck parsing + slide rendering |
| **Vision** | Gemini Vision | OCR for image-based PDFs |
| **Embeddings** | Gemini text-embedding-001 | Semantic cache (768-dim, cosine similarity) |
| **Testing** | pytest, httpx | 87 tests across 7 test files |
| **Deployment** | Docker, Vercel, Render | Containerized + cloud deploy |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Health check (alias) |
| `GET` | `/sample-pitches` | 3 sample startup pitches |
| `GET` | `/stats` | Server stats, cache stats, rate limiter stats |
| `GET` | `/metrics` | Live latency percentiles, error rate, cache hit rate |
| `GET` | `/rate-limit/status` | Current rate limit for requesting IP |
| `POST` | `/evaluate` | Synchronous pitch evaluation (REST) |
| `POST` | `/verify-code` | Investor access code → 6-hour session token |
| `POST` | `/upload-deck` | PDF/PPTX upload → text extraction + deck quality |
| `POST` | `/share` | Create shareable evaluation link (7-day TTL) |
| `GET` | `/eval/{share_id}` | Retrieve shared evaluation |
| `DELETE` | `/cache/clear` | Clear all caches |
| `DELETE` | `/cache/entry` | Delete specific cache entry |
| `WebSocket` | `/ws/evaluate` | Real-time ReAct agent streaming |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- [Gemini API Key](https://ai.google.dev/) (free tier available)
- [Tavily API Key](https://tavily.com/) (free tier: 1,000 searches/month)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env`:
```env
GEMINI_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key
INVESTOR_ACCESS_CODE=your_secret_code
LLM_PROVIDER=gemini
FRONTEND_URL=http://localhost:3000
ENV=development
```

```bash
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Run Tests

```bash
cd backend
python -m pytest tests/ -v
```

```
87 passed in 5.4s
```

---

## Docker Deployment

```bash
docker compose up --build
```

This starts both services:
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`

---

## Cloud Deployment (Free)

| Service | Platform | Cost |
|---|---|---|
| Frontend | Vercel | Free |
| Backend | Render | Free |
| Keep-alive | UptimeRobot | Free |

See deployment steps in the [Deployment Plan](#deployment-plan) section below.

### Deployment Plan

1. Push code to GitHub
2. **Render**: Create Web Service → root dir `backend` → build `pip install -r requirements.txt` → start `uvicorn app.main:app --host 0.0.0.0 --port 8000` → add env vars
3. **Vercel**: Import repo → root dir `frontend` → add `NEXT_PUBLIC_API_URL` env var pointing to Render URL
4. **Render**: Update `FRONTEND_URL` env var to Vercel URL (for CORS)
5. **UptimeRobot**: Monitor `https://your-app.onrender.com/health` every 14 min (prevents cold starts)

---

## Project Structure

```
PitchPal-v2/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app + REST + WebSocket endpoints
│   │   ├── config.py               # Environment configuration
│   │   ├── security.py             # Input sanitization + injection detection
│   │   ├── logger.py               # Structured NDJSON logging
│   │   ├── metrics.py              # In-memory performance metrics
│   │   ├── models/
│   │   │   └── schemas.py          # Pydantic models (evaluation, deck, agent steps)
│   │   └── agent/
│   │       ├── react_agent.py      # Custom ReAct loop (no frameworks)
│   │       ├── llm.py              # Multi-provider LLM abstraction
│   │       ├── tools.py            # 4 Tavily search tools with domain filtering
│   │       ├── deck_analyzer.py    # PDF/PPTX parsing + Gemini Vision OCR
│   │       ├── evaluation_cache.py # SHA-256 hash-based evaluation cache
│   │       ├── semantic_cache.py   # Embedding-based similarity cache
│   │       ├── share_store.py      # Shareable evaluation links (7-day TTL)
│   │       └── rate_limiter.py     # IP-based rate limiting
│   ├── tests/
│   │   ├── test_api.py             # 17 API integration tests
│   │   ├── test_security.py        # 10 input sanitization tests
│   │   ├── test_rate_limiter.py    # 7 rate limiter tests
│   │   ├── test_evaluation_cache.py# 8 cache tests
│   │   ├── test_share_store.py     # 6 share store tests
│   │   └── test_agent_quality.py   # 39 agent quality benchmarks
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx            # Landing page (role selection)
│   │   │   ├── evaluate/page.tsx   # Main evaluation interface
│   │   │   ├── eval/[id]/page.tsx  # Shared evaluation view
│   │   │   ├── history/page.tsx    # Evaluation history
│   │   │   ├── compare/page.tsx    # Side-by-side pitch comparison
│   │   │   └── why/page.tsx        # Feature comparison page
│   │   ├── components/
│   │   │   ├── AgentStream.tsx     # Real-time agent step streaming
│   │   │   ├── EvaluationResults.tsx# Structured scoring display
│   │   │   ├── PitchForm.tsx       # Pitch input form
│   │   │   ├── DeckUpload.tsx      # PDF/PPTX upload component
│   │   │   ├── RadarChart.tsx      # Dimension radar chart
│   │   │   ├── ScoreBarChart.tsx   # Score bar visualization
│   │   │   ├── ShareButton.tsx     # Share + PDF export
│   │   │   ├── Header.tsx          # Navigation + theme toggle
│   │   │   └── ThemeProvider.tsx   # Dark/light theme context
│   │   └── lib/
│   │       ├── api.ts              # API client + WebSocket
│   │       ├── auth.ts             # Role + token management
│   │       ├── storage.ts          # LocalStorage for history
│   │       └── pdfExport.ts        # PDF export utilities
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Testing

87 tests across 7 files, covering:

| Test File | Count | What It Tests |
|---|---|---|
| `test_api.py` | 17 | Health, rate limits, auth, cache, shares, deck upload, stats |
| `test_security.py` | 10 | HTML stripping, injection detection, Unicode normalization |
| `test_rate_limiter.py` | 7 | Limits, window reset, IP isolation, thread safety |
| `test_evaluation_cache.py` | 8 | Set/get, TTL, role separation, key normalization |
| `test_share_store.py` | 6 | Create/get, view counter, expiry, FIFO eviction |
| `test_agent_quality.py` | 39 | Schema validation, JSON repair, score consistency, benchmarks |

---

## v1 vs v2 Comparison

| Aspect | v1 | v2 |
|---|---|---|
| **Agent** | LangChain `create_react_agent` | Custom ReAct loop from scratch |
| **LLM** | OpenAI GPT-4 only | Gemini, Groq, OpenAI, Anthropic |
| **Frontend** | Streamlit | Next.js 16 + React 19 + Tailwind |
| **Backend** | Streamlit server | FastAPI + WebSocket |
| **Search** | LangChain tools (no real web) | Tavily live search (4 specialized tools) |
| **Caching** | None | Semantic (embeddings) + evaluation (SHA-256) |
| **Streaming** | Streamlit spinner | Real-time WebSocket step streaming |
| **File Upload** | None | PDF/PPTX with Gemini Vision OCR |
| **Security** | None | Input sanitization, rate limiting, injection detection |
| **Testing** | None | 87 tests (pytest) |
| **Deployment** | Streamlit Cloud | Docker + Vercel + Render |
| **Design** | Default Streamlit | Glassmorphism with dark/light themes |

---

## Author

**Jeet Patel**
- GitHub: [Jeet-51](https://github.com/Jeet-51)

---

## License

MIT License - see [LICENSE](LICENSE) file for details.
