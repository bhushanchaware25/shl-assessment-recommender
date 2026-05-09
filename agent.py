"""
agent.py — Conversation logic for the SHL Assessment Recommender.

Uses Groq inference API (llama-3.3-70b-versatile) for fast, reliable completions.

Decision flow:
1. Scope check (no LLM) → refusal if off-topic/injection
2. Intent detection
3. Context sufficiency check
4. Clarify OR Recommend via LLM
5. URL validation against catalog (anti-hallucination gate)
6. Turn cap enforcement

Aligned with GenAI_SampleConversations:
- recommendations=[] (empty list) when gathering context
- Rich Recommendation objects with keys, duration, languages
"""

import json
import logging
import os
import re
import time
from typing import Optional

from groq import Groq, RateLimitError, APIStatusError

import prompts
import retriever
from models import ChatResponse, Recommendation

logger = logging.getLogger(__name__)

# ── Groq configuration ────────────────────────────────────────────────────────
# Fallback chain: tried in order on rate limit / quota errors
GROQ_MODELS = [
    "llama-3.3-70b-versatile",          # primary — most capable
    "meta-llama/llama-4-scout-17b-16e-instruct",  # faster alternative
    "llama-3.1-8b-instant",             # lightest fallback
]
MAX_OUTPUT_TOKENS = 1200
TEMPERATURE = 0.2

# ── Agent constants ───────────────────────────────────────────────────────────
MAX_TURNS = 8
FORCE_RECOMMEND_BY = 6

# ── Role / context keyword patterns ──────────────────────────────────────────
ROLE_PATTERNS = [
    r"\b(engineer|developer|programmer|coder)\b",
    r"\b(manager|director|lead|head\s+of)\b",
    r"\b(analyst|researcher|scientist|data scientist)\b",
    r"\b(sales|account\s*manager|representative|rep)\b",
    r"\b(customer\s*service|support\s*agent|call\s*cent(er|re))\b",
    r"\b(executive|c[- ]?suite|ceo|cto|cfo|coo)\b",
    r"\b(graduate|intern|entry.?level|junior|senior|mid.?level)\b",
    r"\b(role|position|job|vacancy|opening|hire|hiring)\b",
    r"\b(nurse|doctor|physician|medical)\b",
    r"\b(teacher|professor|educator)\b",
    r"\b(accountant|finance|financial|bookkeep)\b",
    r"\b(marketing|designer|creative)\b",
    r"\b(operations|logistics|supply\s*chain)\b",
    r"\b(hr|human\s*resources|recruiter|talent)\b",
    r"\b(administrator|clerk|assistant)\b",
    r"\b(technician|operator|mechanic)\b",
    r"\b(trainee|scheme|programme|program)\b",
    r"\b(contact\s*cent(er|re)|inbound|outbound|agent)\b",
    r"\bcxo\b",
    r"\bic\b",
]

REFINE_PATTERNS = [
    r"\badd\b.*\btests?\b",
    r"\binclude\b",
    r"\bexclude\b",
    r"\binstead\b",
    r"\bmore\s+(specific|focused|targeted)\b",
    r"\bfilter\b",
    r"\brefine\b",
    r"\bnarrow\s+down\b",
    r"\bonly\s+(show|give|include)\b",
    r"\bwithout\b.*(test|assessment|personality|cognitive|aptitude)",
    r"\bdrop\b",
    r"\bremove\b",
    r"\bswap\b",
    r"\breplace\b",
]

COMPARE_PATTERNS = [
    r"\bcompare\b",
    r"\bdifference\s+between\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bwhich\s+(is\s+)?better\b",
    r"\bwhat.s\s+the\s+difference\b",
]

CONFIRM_PATTERNS = [
    r"\bperfect\b",
    r"\bthat.s\s+what\s+we\s+need\b",
    r"\bthat\s+works\b",
    r"\bconfirmed\b",
    r"\blocking\s+it\s+in\b",
    r"\bfinal\s+(list|battery|shortlist)\b",
    r"\bthank(s| you)\b",
    r"\bgood\s+(two.stage|design|call|choice)\b",
    r"\bkeep\s+(verify|opq|it)\b",
]

_ROLE_RE    = [re.compile(p, re.IGNORECASE) for p in ROLE_PATTERNS]
_REFINE_RE  = [re.compile(p, re.IGNORECASE) for p in REFINE_PATTERNS]
_COMPARE_RE = [re.compile(p, re.IGNORECASE) for p in COMPARE_PATTERNS]
_CONFIRM_RE = [re.compile(p, re.IGNORECASE) for p in CONFIRM_PATTERNS]


# ── Groq client ───────────────────────────────────────────────────────────────

def _init_groq() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set. "
            "Get a free key at https://console.groq.com/keys"
        )
    return Groq(api_key=api_key)


# ── Intent helpers ────────────────────────────────────────────────────────────

def _has_role_context(messages: list[dict]) -> bool:
    for msg in messages:
        if msg.get("role") == "user":
            if any(p.search(msg.get("content", "")) for p in _ROLE_RE):
                return True
    return False


def _detect_intent(user_message: str) -> str:
    if any(p.search(user_message) for p in _COMPARE_RE):
        return "compare"
    if any(p.search(user_message) for p in _CONFIRM_RE):
        return "confirm"
    if any(p.search(user_message) for p in _REFINE_RE):
        return "refine"
    if any(p.search(user_message) for p in _ROLE_RE):
        return "recommend"
    return "clarify"


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json_from_response(text: str) -> Optional[dict]:
    """Strip markdown fences and parse JSON from LLM output."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try the first {...} block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ── Recommendation validation ─────────────────────────────────────────────────

def _format_languages(langs: list) -> str:
    if not langs:
        return "—"
    if len(langs) <= 4:
        return ", ".join(langs)
    return ", ".join(langs[:4]) + f" (+{len(langs)-4} more)"


def _validate_recommendations(raw_recs) -> list[Recommendation]:
    """Validate URLs against catalog; enrich with authoritative catalog fields."""
    if not raw_recs:
        return []

    valid = []
    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue
        name = rec.get("name", "").strip()
        url  = rec.get("url", "").strip()
        if not name or not url:
            continue

        # Anti-hallucination gate
        if not retriever.is_valid_catalog_url(url):
            logger.warning(f"Dropping non-catalog URL: {url!r}")
            continue

        # Look up authoritative catalog entry
        entry = next(
            (e for e in retriever.get_all_assessments()
             if e["url"].rstrip("/") == url.rstrip("/")),
            None,
        )

        if entry:
            codes = ",".join(entry.get("test_types", []))
            keys  = " & ".join(entry.get("test_types_full", [])) or rec.get("keys", "")
            dur   = (entry.get("duration_str")
                     or ("—" if entry.get("duration") is None
                         else f"{entry['duration']} minutes"))
            langs = _format_languages(entry.get("languages", []))
        else:
            codes = rec.get("test_type", "")
            keys  = rec.get("keys", "")
            dur   = rec.get("duration", "—")
            langs = rec.get("languages", "—")

        valid.append(Recommendation(
            name=name, url=url,
            test_type=codes, keys=keys,
            duration=dur, languages=langs,
        ))

    return valid[:10]


# ── LLM call with model fallback ──────────────────────────────────────────────

def _call_llm(prompt: str, client: Groq) -> Optional[dict]:
    """
    Call Groq with a model fallback chain.
    - On RateLimitError: wait briefly, then try next model.
    - On non-JSON response: retry once on same model, then move on.
    """
    for model_name in GROQ_MODELS:
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
                text = (response.choices[0].message.content or "").strip()
                if not text:
                    logger.warning(f"{model_name}: empty response (attempt {attempt+1})")
                    break  # try next model

                parsed = _extract_json_from_response(text)
                if parsed is not None:
                    logger.debug(f"Success with {model_name}")
                    return parsed

                logger.warning(
                    f"{model_name}: non-JSON (attempt {attempt+1}): {text[:120]}"
                )
                # retry same model once for non-JSON
                continue

            except RateLimitError:
                wait = 8 * (attempt + 1)  # 8s, 16s
                logger.warning(
                    f"{model_name} rate limited (attempt {attempt+1}), "
                    f"waiting {wait}s then trying next model..."
                )
                time.sleep(wait)
                if attempt == 1:
                    logger.warning(f"{model_name}: giving up, switching model.")
                continue

            except APIStatusError as e:
                logger.error(f"{model_name} API error (attempt {attempt+1}): {e}")
                break  # non-recoverable for this model

            except Exception as e:
                logger.error(f"{model_name} unexpected error (attempt {attempt+1}): {e}")
                break

    logger.error("All Groq models exhausted — no response.")
    return None


# ── Fallback response ─────────────────────────────────────────────────────────

def _safe_default(msg: str = "") -> ChatResponse:
    return ChatResponse(
        reply=(
            msg or
            "I encountered a technical issue. Please try again, or describe "
            "the role you're hiring for and I'll find the best SHL assessments."
        ),
        recommendations=[],
        end_of_conversation=False,
    )


# ── Search query builder ──────────────────────────────────────────────────────

def _build_search_query(messages: list[dict], latest: str) -> str:
    parts = [latest]
    for msg in messages[-6:]:
        if msg.get("role") == "user":
            c = msg.get("content", "")
            if c != latest and len(c) > 10:
                parts.append(c)
    return " ".join(parts)


# ── Agent ─────────────────────────────────────────────────────────────────────

class SHLAgent:
    """Stateless agent — all conversation state lives in the messages list."""

    def __init__(self) -> None:
        self._client: Optional[Groq] = None

    def _get_client(self) -> Groq:
        if self._client is None:
            self._client = _init_groq()
        return self._client

    def process_turn(self, messages: list[dict]) -> ChatResponse:
        if not messages:
            return _safe_default("Please describe the role you're hiring for.")

        latest_msg = messages[-1]
        if latest_msg.get("role") != "user":
            return _safe_default("I'm waiting for your message.")

        user_text = latest_msg.get("content", "").strip()
        if not user_text:
            return _safe_default("Please describe the role you're hiring for.")

        # ── 1. Scope check (no LLM) ───────────────────────────────────────────
        should_refuse, reason = prompts.check_refusal(user_text)
        if should_refuse:
            reply = (
                prompts.INJECTION_REFUSAL if reason == "injection"
                else prompts.OFF_TOPIC_REFUSAL
            )
            return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

        # ── 2. Intent & turn ──────────────────────────────────────────────────
        turn_num       = len(messages)
        force_recommend = turn_num >= FORCE_RECOMMEND_BY
        intent         = _detect_intent(user_text)
        prior_messages = messages[:-1]

        # ── 3. Context sufficiency ────────────────────────────────────────────
        has_role          = _has_role_context(messages)
        context_sufficient = (
            has_role or force_recommend
            or intent in ("refine", "compare", "confirm")
        )

        # ── 4. Retrieval ──────────────────────────────────────────────────────
        candidates: list = []
        if context_sufficient:
            try:
                query      = _build_search_query(prior_messages, user_text)
                results    = retriever.hybrid_search(query, top_k=12)
                candidates = [e for e, _ in results]
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")

        # ── 5. Build prompt ───────────────────────────────────────────────────
        if not context_sufficient and intent == "clarify":
            prompt = prompts.build_clarify_prompt(user_text, history=prior_messages)
        else:
            prompt = prompts.build_prompt(
                user_message=user_text,
                candidates=candidates,
                history=prior_messages,
            )

        # ── 6. LLM call ───────────────────────────────────────────────────────
        try:
            client = self._get_client()
        except RuntimeError as e:
            return _safe_default(str(e))

        parsed = _call_llm(prompt, client)
        if parsed is None:
            return _safe_default()

        # ── 7. Extract, validate, return ──────────────────────────────────────
        reply    = parsed.get("reply", "Here are my recommendations.")
        raw_recs = parsed.get("recommendations")  # None or list from LLM

        # Force null when still clarifying or comparing without confirmation
        if not context_sufficient and intent == "clarify":
            raw_recs = None
        if intent == "compare" and not raw_recs:
            raw_recs = None

        # Validate — drops hallucinated URLs
        validated: list[Recommendation] = []
        if raw_recs is not None:
            validated = _validate_recommendations(raw_recs)

        end_conv = bool(parsed.get("end_of_conversation", False))
        if not validated:
            end_conv = False
        if turn_num >= MAX_TURNS and validated:
            end_conv = True

        return ChatResponse(
            reply=reply,
            recommendations=validated,
            end_of_conversation=end_conv,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
_agent = SHLAgent()


def get_agent() -> SHLAgent:
    return _agent
