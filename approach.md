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

- **Catalog freshness**: `shl_product_catalog.json` is a point-in-time snapshot. Production would require scheduled re-scraping and embedding invalidation.
- **Cold-start latency**: `all-MiniLM-L6-v2` embedding computation takes ~10s on first startup. Render free-tier adds another ~60–90s for cold start.
- **Domain specificity**: The embedding model was not fine-tuned on HR/assessment data. Highly niche queries (e.g., "MBTI-equivalent for occupational settings") may return adjacent but suboptimal results.
- **LLM non-determinism**: Despite `temperature=0.2`, the LLM may occasionally ask an extra clarifying question for ambiguous JDs rather than recommending. The turn-cap at turn 6 hard-forces a recommendation to prevent infinite clarification loops.
