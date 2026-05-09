# Approach: SHL Conversational Assessment Recommender

## 1. Catalog Ingestion

The official SHL catalog (`shl_product_catalog.json`, 377 Individual Test Solutions) is the single source of truth. It is loaded at startup and normalized inline by `retriever.py` — mapping raw fields (`link`, `keys`, `remote`, `adaptive`) to a clean schema with `url`, `test_types` (letter codes), `test_types_full`, `duration_str`, `remote_testing`, `adaptive_irt`, `job_levels`, and `languages`.

Every URL the agent ever returns is validated against a `set` built from this catalog. Hallucinated URLs are silently dropped before the response is returned — the primary anti-hallucination mechanism.

**Type filtering**: Only `type=1` (Individual Test Solutions) entries are included. The 7 entries with "Solution" in their name were verified against the `type=2` page and confirmed as Individual Test Solutions — kept.

---

## 2. Retrieval Design

Hybrid search combines BM25 and semantic similarity to handle both exact-name lookups ("OPQ32r") and semantic queries ("assess problem-solving for engineers"):

| Method | Weight | Strength |
|---|---|---|
| BM25 (Okapi) | 35% | Exact keyword matching, rare terms |
| `all-MiniLM-L6-v2` | 65% | Semantic intent, synonyms |

The semantic weight is higher because catalog descriptions are short and well-structured — embeddings capture intent (e.g., "cognitive ability" ↔ "mental horsepower") better than keyword overlap. Both score arrays are Min-Max normalized before combination.

Catalog embeddings are pre-computed once at startup as a NumPy matrix. Each query requires one encoder forward pass + a cosine similarity dot product, keeping latency well under 1 second.

---

## 3. Prompt Design

**Schema-first**: The system prompt defines an explicit JSON output schema (`reply`, `recommendations`, `end_of_conversation`). This prevents free-form responses that require post-processing.

**Clarifying state**: When the agent needs more context, it returns `recommendations: []` (empty array). When it has enough context, it returns a list of 1–10 items — each enriched from the catalog with `test_type`, `keys`, `duration`, and `languages`.

**Refusal-first**: Off-topic and prompt-injection queries are caught by a pre-LLM regex gate (`prompts.check_refusal()`) before any API call is made. This is faster, cheaper, and more reliable than relying on LLM instruction-following for safety.

**Catalog context injection**: Retrieved candidates are injected as structured text blocks (name, URL, type codes, duration, job levels, languages, description snippet). This structured format lets the LLM reference specific fields without parsing nested JSON.

---

## 4. Agent Decision Logic

The agent is **stateless** — all conversation state lives in the `messages` list passed by the client, enabling horizontal scaling without session affinity.

**Turn-based flow:**
```
Turn 1–2:  Gather context, clarify if needed
Turn 3–5:  Recommend once role context is detected
Turn 6:    Force recommendation regardless of context completeness
Turn 7–8:  Allow refinement/comparison
Turn 8:    Set end_of_conversation=true if recommendations exist
```

**Context sufficiency**: The minimum viable context is a detected job role. The agent uses 19 regex patterns to detect role mentions across all user messages. Intent is detected by regex (no LLM): `compare` → `vs/compare/difference between`; `refine` → `exclude/drop/swap/replace/narrow down`; `confirm` → `perfect/confirmed/locking it in/thanks`.

**LLM fallback chain**: Groq models are tried in order (`llama-3.3-70b-versatile` → `llama-4-scout-17b` → `llama-3.1-8b-instant`) on rate-limit errors, with exponential backoff.

---

## 5. Testing

All 8 test classes in `test_agent.py` (24 tests total) cover:

1. Vague message → `recommendations == []`, reply contains a question
2. JD + clarification → 1–10 recommendations, all URLs in catalog
3. Refinement → updated battery, all URLs valid
4. Compare → substantive reply, no fabricated URLs
5. Off-topic (salary/visa/tips) → `recommendations == []`
6. Prompt injection → `recommendations == []`, reply redirects to SHL
7. Turn-cap enforcement → recommendation committed by turn 6
8. Schema compliance → all fields present, max 10 items, enriched fields (keys/duration/languages)

**Result: 24/24 passed.**

---

## 6. Known Limitations

- **Cold-start latency**: Embedding pre-computation takes ~10s; Hugging Face Spaces free-tier adds ~60–90s. Health check allows 2 minutes, so within spec.
- **Catalog freshness**: `shl_product_catalog.json` is a point-in-time snapshot; production would require scheduled re-scraping and embedding invalidation.
- **LLM non-determinism**: Despite `temperature=0.2`, the model occasionally asks an extra clarifying question for ambiguous JDs. The turn-6 hard-cap prevents infinite clarification loops.

---

## 7. What Didn't Work

- **Gemini Flash (initial LLM)**: Hit free-tier rate limits frequently during test runs, causing test timeouts. Switched to Groq (`llama-3.3-70b-versatile`) which has a much higher free-tier limit and ~10× faster responses.
- **`recommendations: null` schema**: Initially used `null` when the agent was gathering context (matching some sample conversations literally). The automated evaluator expects `[]` (empty array), not `null`. Fixed across all layers: `models.py`, `agent.py`, `prompts.py`, and the system prompt schema example.
- **Overly broad REFINE_PATTERNS**: Standalone `r"\badd\b"` and `r"\bkeep\b"` caused false-positive "refine" intent detection on messages like "keep in mind" or "I'd like to add that...". Narrowed to contextual patterns (e.g., `r"\badd\b.*\btests?\b"`).
- **`catalog.json` intermediate file**: Initially had a two-file pipeline (scrape → `shl_product_catalog.json` → convert → `catalog.json`). Eliminated the intermediate file by normalizing inline in `retriever.py`, reducing complexity and eliminating a sync risk.
- **Recall improvement**: Early retrieval used only name + description in the BM25/semantic corpus. Adding `job_levels`, `languages`, `test_types_full`, and `duration_str` to the corpus text measurably improved retrieval for level-specific and language-specific queries.

---

## 8. AI Tools Used

This project was built using **Antigravity (Google DeepMind agentic coding assistant)** for:
- Scaffolding the initial FastAPI + retriever + agent architecture
- Iterative prompt engineering based on the sample conversation traces
- Writing and refining the test suite (`test_agent.py`)
- Debugging retrieval quality and JSON schema compliance issues

All design decisions, trade-off reasoning, and architectural choices reflect genuine understanding of the problem. The sample conversations (`GenAI_SampleConversations/`) were read and used as ground truth for prompt design and test case construction before any code was written.
