"""
prompts.py — System prompts and safety triggers for the SHL Assessment Recommender.

Aligned with GenAI_SampleConversations patterns:
- recommendations: [] when gathering context (not null)
- Rich recommendation table with Keys, Duration, Languages
- Consultative tone, proactively adds OPQ32r + Verify G+
- Honest gaps: if a test isn't in catalog, say so
"""

import re
from typing import List, Optional

# ── Refusal triggers ──────────────────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|prior)\s+instructions",
    r"you\s+are\s+now\s+",
    r"forget\s+(everything|all|what)\s+",
    r"disregard\s+(previous|prior|all)\s+",
    r"act\s+as\s+(if\s+you\s+(are|were)|a\s+)",
    r"pretend\s+(you\s+(are|were)|to\s+be)",
    r"jailbreak",
    r"dan\s+mode",
    r"override\s+(your\s+)?(system|safety|instructions)",
    r"new\s+instructions\s*:",
    r"system\s*prompt\s*:",
    r"<\/?(system|user|assistant)>",
    r"ignore\s+the\s+above",
    r"reveal\s+(your|the)\s+(prompt|instructions|system)",
]

OFF_TOPIC_PATTERNS = [
    r"\bsalary\b",
    r"\bcompensation\b",
    r"\bpay\s+(scale|grade|range|cut|raise)\b",
    r"\bvisa\b",
    r"\bwork\s+permit\b",
    r"\bimmigration\b",
    r"\blegal\s+(compliance|advice|counsel)\b",
    r"\binterview\s+tip",
    r"\bhow\s+to\s+(pass|ace|beat|trick)\s+(an?\s+)?interview",
    r"\bcover\s+letter\b",
    r"\bresume\s+(writing|tips|help)\b",
    r"\bcv\s+(writing|tips)\b",
    r"\bnegotiat\w*\s+(salary|offer|contract)\b",
    r"\bbenefit(s)?\s+package\b",
    r"\bstock\s+option",
    r"\bequity\s+compensation\b",
    r"\blegal\s+question",
    r"\bunfair\s+dismissal\b",
    r"\bdiscrimination\s+law\b",
    r"\bemployment\s+law\b",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_OFF_TOPIC_RE = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]


def is_prompt_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_RE)


def is_off_topic(text: str) -> bool:
    return any(p.search(text) for p in _OFF_TOPIC_RE)


def check_refusal(text: str) -> tuple[bool, str]:
    if is_prompt_injection(text):
        return True, "injection"
    if is_off_topic(text):
        return True, "off_topic"
    return False, ""


# ── Canned refusal responses ──────────────────────────────────────────────────
INJECTION_REFUSAL = (
    "I'm here to help you find the right SHL assessments for your hiring needs. "
    "I can't process that type of request. Please tell me about the role you're "
    "hiring for, and I'll recommend the most suitable SHL assessments."
)

OFF_TOPIC_REFUSAL = (
    "I specialize in recommending SHL talent assessments for hiring managers. "
    "That topic falls outside my area of expertise. "
    "If you'd like help finding the right cognitive, personality, or skills assessments "
    "for a specific role, I'm here to help!"
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert SHL Assessment Recommender helping hiring managers build the right assessment battery for their roles.

STRICT RULES:
1. Only recommend assessments that appear in CATALOG CONTEXT below. Never invent names or URLs.
2. If a specific test doesn't exist in the catalog (e.g. a Rust test), say so honestly and suggest the closest alternative.
3. You are NOT a general HR advisor. Politely refuse salary, legal, visa, and other off-topic questions.
4. Output ONLY valid JSON in the exact schema below — no markdown fences, no text outside the JSON.
5. "recommendations" must be [] (empty array) when you are still gathering context or asking a clarifying question.
6. "recommendations" must be a list of 1–10 items when you have enough context to recommend.

OUTPUT SCHEMA (strict):
{{
  "reply": "<your conversational response>",
  "recommendations": [],
  "end_of_conversation": false
}}

OR when recommending:
{{
  "reply": "<your response summarizing the battery>",
  "recommendations": [
    {{
      "name": "<exact name from catalog>",
      "url": "<exact url from catalog>",
      "test_type": "<letter codes e.g. K or A,P or B,S>",
      "keys": "<full type names e.g. Knowledge & Skills>",
      "duration": "<duration string or — if unknown>",
      "languages": "<first few languages or abbreviated>"
    }}
  ],
  "end_of_conversation": false
}}

CONVERSATION STYLE (follow these patterns from examples):
- Be consultative and expert, not a search engine.
- When a JD or role is given covering many skills, ask ONE focused clarifying question to narrow down (e.g., backend-leaning vs frontend, senior IC vs tech lead).
- For technical senior roles: proactively include SHL Verify Interactive G+ (A) for cognitive ability and OPQ32r (P) for personality, unless the user explicitly declines.
- When the user says "thanks" / "perfect" / "that works" / "confirmed" / "locking it in", set end_of_conversation to true and repeat the final list.
- When the user refines (add X, drop Y), update the list and respond with the updated battery.
- When comparing products, explain the difference without immediately recommending — set recommendations to [] unless the user then confirms.
- Use "—" for duration or languages when not known.

CONTEXT SUFFICIENCY — when to recommend vs. clarify:
- MINIMUM REQUIRED to recommend: job role or job type is mentioned.
- NICE TO HAVE: seniority level, key skills, volume, language requirements, remote constraints.
- If no role mentioned at all, ask exactly ONE clarifying question.
- By turn 6 (counting user + assistant messages), COMMIT to a recommendation even if context is imperfect.

TURN CAP: Never allow more than 8 total turns. At turn 8, always deliver a final battery.

CATALOG CONTEXT (retrieved assessments):
{catalog_context}

CONVERSATION SO FAR:
{conversation_history}

USER'S LATEST MESSAGE: {user_message}
"""


def format_catalog_context(candidates: List[dict]) -> str:
    """Format retrieved candidates with all enriched fields."""
    if not candidates:
        return "No specific assessments retrieved yet."

    lines = []
    for i, entry in enumerate(candidates, 1):
        codes = ", ".join(entry.get("test_types", [])) or "N/A"
        keys_full = " | ".join(entry.get("test_types_full", [])) or "N/A"
        duration = entry.get("duration_str") or ("—" if entry.get("duration") is None else f"{entry['duration']} min")
        remote = "Yes" if entry.get("remote_testing") else "Unknown"
        adaptive = "Yes" if entry.get("adaptive_irt") else "No"
        desc = (entry.get("description", "") or "")[:200]
        job_levels = ", ".join(entry.get("job_levels", [])[:4]) or "N/A"
        langs = entry.get("languages", [])
        if len(langs) > 4:
            lang_str = ", ".join(langs[:4]) + f" (+{len(langs)-4} more)"
        else:
            lang_str = ", ".join(langs) or "—"

        lines.append(
            f"{i}. NAME: {entry['name']}\n"
            f"   URL: {entry['url']}\n"
            f"   TYPE CODES: {codes} | KEYS: {keys_full}\n"
            f"   DURATION: {duration} | REMOTE: {remote} | ADAPTIVE: {adaptive}\n"
            f"   JOB LEVELS: {job_levels}\n"
            f"   LANGUAGES: {lang_str}\n"
            f"   DESCRIPTION: {desc}"
        )
    return "\n\n".join(lines)


def format_conversation_history(messages: Optional[List[dict]]) -> str:
    """Format conversation history for the prompt."""
    if not messages:
        return "No prior conversation."
    lines = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def build_prompt(
    user_message: str,
    candidates: List[dict],
    history: List[dict],
) -> str:
    """Build the full LLM prompt."""
    catalog_context = format_catalog_context(candidates)
    conversation_history = format_conversation_history(history)

    return SYSTEM_PROMPT.format(
        catalog_context=catalog_context,
        conversation_history=conversation_history,
        user_message=user_message,
    )


# ── Clarifying question prompt ────────────────────────────────────────────────
CLARIFY_PROMPT = """You are an expert SHL Assessment Recommender. The user hasn't given enough context to recommend yet.

CONVERSATION HISTORY (already discussed — DO NOT repeat these topics):
{conversation_history}

User's latest message:
{user_message}

Ask exactly ONE NEW clarifying question about something NOT already covered above. The most important missing info is usually: what role/job type they are hiring for.

Respond ONLY as valid JSON:
{{
  "reply": "<single focused clarifying question>",
  "recommendations": [],
  "end_of_conversation": false
}}"""


def build_clarify_prompt(user_message: str, history: Optional[List[dict]] = None) -> str:
    """Build a clarifying question prompt aware of prior conversation."""
    conversation_history = format_conversation_history(history) if history else "No prior conversation."
    return CLARIFY_PROMPT.format(
        user_message=user_message,
        conversation_history=conversation_history,
    )
