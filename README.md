# SHL Conversational Assessment Recommender

A production-ready FastAPI service that acts as a conversational agent helping hiring managers find the right SHL assessments. Powered by Google Gemini Flash and hybrid BM25+semantic search.

## Features

- 🔍 **Hybrid Search**: BM25 keyword matching + Sentence Transformer semantic embeddings
- 🤖 **Conversational Agent**: Multi-turn dialogue with context tracking and intent detection
- 🛡️ **Scope Enforcement**: Hardcoded refusal triggers for off-topic queries and prompt injection attempts
- ✅ **URL Validation**: Every recommended URL is validated against `catalog.json` before returning
- ⚡ **Fast Startup**: Catalog and embeddings loaded at boot, not per-request
- 🚀 **Render Ready**: Deploy on Render free tier with `render.yaml`

## Project Structure

```
SHL-Assignment/
├── catalog.json          # Scraped SHL catalog (source of truth)
├── main.py               # FastAPI app
├── agent.py              # Conversation logic + LLM calls
├── retriever.py          # Hybrid BM25 + semantic search
├── prompts.py            # System prompts + refusal triggers
├── models.py             # Pydantic request/response schemas
├── scraper.py            # One-time catalog scraper
├── test_agent.py         # Pytest test suite (7 scenarios)
├── requirements.txt      # Pinned dependencies
├── render.yaml           # Render deployment config
├── .env.example          # Environment variable template
├── approach.md           # Design decisions & architecture
└── README.md
```

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I'm hiring a software engineer..."},
    {"role": "assistant", "content": "What level of seniority..."},
    {"role": "user", "content": "Senior level, 5+ years experience"}
  ]
}
```

**Response:**
```json
{
  "reply": "Based on your requirements, here are the most suitable SHL assessments...",
  "recommendations": [
    {
      "name": "Verify Numerical Reasoning",
      "url": "https://www.shl.com/products/product-catalog/view/verify-numerical-reasoning/",
      "test_type": "A"
    }
  ],
  "end_of_conversation": false
}
```

**Rules:**
- `recommendations` is `[]` when gathering context or refusing
- `recommendations` contains 1–10 items when committing to a shortlist
- `end_of_conversation` is `true` only when the agent considers the task complete

## Setup & Installation

### Prerequisites
- Python 3.11+
- Google Gemini API key (free tier at [Google AI Studio](https://makersuite.google.com/app/apikey))

### 1. Clone and Install Dependencies

```bash
git clone <your-repo-url>
cd SHL-Assignment
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY="your_key_here"

# Linux/Mac
export GEMINI_API_KEY="your_key_here"
```

### 3. Generate the Catalog (if not present)

```bash
python scraper.py
```

This scrapes ~383 Individual Test Solutions from the SHL catalog and saves `catalog.json`. Takes ~5-10 minutes due to polite rate limiting.

### 4. Run Locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or:
```bash
python main.py
```

The server will:
1. Load `catalog.json` (fails loudly if missing)
2. Build BM25 index
3. Download and load `all-MiniLM-L6-v2` sentence transformer (~90MB)
4. Pre-compute all catalog embeddings
5. Start serving on port 8000

**First startup takes ~60-90 seconds.** Subsequent requests are fast.

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need an assessment for a senior software engineer"}]}'
```

### 6. Run Tests

```bash
pytest test_agent.py -v
```

## Deployment on Render

### Prerequisites
1. Push this project to a GitHub repository
2. Get a free Gemini API key

### Steps

1. **Sign up** at [render.com](https://render.com)
2. **New Web Service** → Connect your GitHub repo
3. Render will auto-detect `render.yaml` and configure the service
4. **Add Environment Variable**: Go to Environment → Add `GEMINI_API_KEY` with your key
5. **Deploy**

> **Note**: The evaluator allows **2 minutes** for cold start. The first `/health` call may take up to 90 seconds while the embedding model loads.

### Manual Render Configuration (if not using render.yaml)

| Setting | Value |
|---------|-------|
| Runtime | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python main.py` |
| Health Check Path | `/health` |

## Agent Decision Flow

```
User message received
        │
        ▼
Scope check (hardcoded regex)
   ├── Off-topic? → Canned refusal (no LLM)
   ├── Prompt injection? → Canned refusal (no LLM)
        │
        ▼
Intent detection
   ├── compare → retrieve named assessments → LLM
   ├── refine → re-retrieve with updated constraints → LLM
   ├── recommend → check context sufficiency
   │       ├── Sufficient (role mentioned)? → retrieve → LLM
   │       └── Insufficient? → clarify (1 question)
        │
        ▼
Turn cap enforcement
   └── Turn ≥ 6 → force recommendation regardless of context
   └── Turn ≥ 8 → set end_of_conversation = true

        ▼
URL validation → drop any non-catalog URLs

        ▼
Return ChatResponse
```

## Test Scenarios Covered

| Scenario | Expected Behavior |
|----------|-------------------|
| Vague first message | Clarify, empty recommendations |
| Job description pasted | 1–10 recommendations, valid URLs |
| Refine mid-conversation | Updated shortlist |
| Compare two assessments | Grounded answer, no hallucination |
| Off-topic message (salary) | Refusal, empty recommendations |
| Prompt injection attempt | Refusal, empty recommendations |
| 8-turn conversation | Recommendation by turn 6 |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |
| `PORT` | Optional | Server port (default: 8000) |
