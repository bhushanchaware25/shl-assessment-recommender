"""
eval_recall.py — Offline evaluation of the SHL Assessment Recommender.

Measures:
1. Recall@K  — fraction of ground-truth assessments appearing in the agent's
               final shortlist (averaged across all conversation traces).
2. Groundedness — fraction of recommended URLs that come from the catalog.
3. Schema compliance — every response has reply/recommendations/end_of_conversation.
4. Turn compliance  — conversation ends within MAX_TURNS.
5. Clarify-first    — vague turn-1 queries do NOT get recommendations immediately.

Ground truth is extracted from GenAI_SampleConversations/C*.md files.
Each .md's FINAL recommendation table is treated as the reference shortlist.

Usage:
    python eval_recall.py                 # run all traces, K=10
    python eval_recall.py --k 5           # Recall@5
    python eval_recall.py --trace C1      # single trace
"""

import argparse
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=10, help="K for Recall@K (default 10)")
parser.add_argument("--trace", type=str, default=None,
                    help="Run a single trace by name e.g. C1")
args = parser.parse_args()

K = args.k
TRACES_DIR = "GenAI_SampleConversations"

# ---------------------------------------------------------------------------
# Bootstrap retriever (needed to validate URLs)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
print("Initializing retriever...", flush=True)
import retriever
retriever.initialize()
catalog_urls = {e["url"].rstrip("/") for e in retriever.get_all_assessments()}

from agent import SHLAgent

# ---------------------------------------------------------------------------
# Ground-truth extraction
# ---------------------------------------------------------------------------

def extract_ground_truth(md_text: str) -> list[str]:
    """
    Extract URLs from the LAST recommendation table in the markdown file.
    Returns a list of canonical (rstrip('/')) URLs.
    """
    # Find all markdown table rows that contain a URL
    url_pattern = re.compile(r'https://www\.shl\.com/[^\s|>]+', re.IGNORECASE)
    # Split into turns and take the last block that has URLs
    tables = url_pattern.findall(md_text)
    # Deduplicate while preserving order
    seen, urls = set(), []
    for u in tables:
        u = u.rstrip("/").rstrip(")")  # strip trailing ) from markdown links
        if u not in seen:
            seen.add(u)
            urls.append(u)
    # The final table is the committed shortlist — we want the LAST set of URLs
    # (they repeat across turns; last occurrence = final committed list)
    # Re-extract from the last occurrence of a table
    blocks = md_text.split("### Turn")
    final_urls = []
    for block in reversed(blocks):
        found = url_pattern.findall(block)
        if found:
            seen2, final_urls = set(), []
            for u in found:
                u = u.rstrip("/").rstrip(")")
                if u not in seen2:
                    seen2.add(u)
                    final_urls.append(u)
            break
    return final_urls


def extract_conversation_turns(md_text: str) -> list[dict]:
    """
    Parse the markdown conversation into a list of user messages only.
    Returns: [{"role": "user", "content": "..."}, ...]
    """
    messages = []
    # Match **User** blocks with quoted content
    user_blocks = re.findall(
        r'\*\*User\*\*\s*\n+\s*>\s*(.+?)(?=\n\n|\*\*Agent\*\*|\Z)',
        md_text, re.DOTALL
    )
    for block in user_blocks:
        content = block.strip()
        # Strip leading > from multi-line quoted blocks
        content = re.sub(r'\n+>\s*', ' ', content)
        messages.append({"role": "user", "content": content})
    return messages


# ---------------------------------------------------------------------------
# Recall@K computation
# ---------------------------------------------------------------------------

def recall_at_k(recommended: list[str], ground_truth: list[str], k: int) -> float:
    if not ground_truth:
        return 1.0  # no ground truth = trivially correct
    top_k = set(r.rstrip("/") for r in recommended[:k])
    hits = sum(1 for g in ground_truth if g.rstrip("/") in top_k)
    return hits / len(ground_truth)


# ---------------------------------------------------------------------------
# Single-trace evaluation
# ---------------------------------------------------------------------------

def evaluate_trace(trace_name: str, md_text: str, agent: SHLAgent) -> dict:
    ground_truth_urls = extract_ground_truth(md_text)
    user_turns = extract_conversation_turns(md_text)

    if not user_turns:
        return {"trace": trace_name, "error": "No user turns found"}

    history = []
    final_recommendations = []
    schema_errors = []
    turn_count = 0
    groundedness_violations = 0
    clarify_first_ok = True

    for i, user_msg in enumerate(user_turns):
        history.append(user_msg)
        turn_count += 1
        t0 = time.time()

        try:
            response = agent.process_turn(history)
        except Exception as e:
            return {"trace": trace_name, "error": f"Agent error on turn {i+1}: {e}"}

        latency = time.time() - t0

        # Schema compliance
        if not response.reply:
            schema_errors.append(f"Turn {i+1}: empty reply")
        if not isinstance(response.recommendations, list):
            schema_errors.append(f"Turn {i+1}: recommendations not a list")
        if not isinstance(response.end_of_conversation, bool):
            schema_errors.append(f"Turn {i+1}: end_of_conversation not bool")

        recs = response.recommendations or []

        # Clarify-first check: turn 1 with vague query should have empty recs
        if i == 0 and len(user_turns[0]["content"].split()) < 8 and len(recs) > 0:
            clarify_first_ok = False

        # Groundedness check
        for rec in recs:
            if rec.url.rstrip("/") not in catalog_urls:
                groundedness_violations += 1

        # Track the final non-empty shortlist
        if recs:
            final_recommendations = [rec.url for rec in recs]

        # Add assistant turn to history
        history.append({
            "role": "assistant",
            "content": response.reply
        })

        print(f"  Turn {i+1}: {len(recs)} recs, end={response.end_of_conversation}, "
              f"latency={latency:.1f}s")

        if response.end_of_conversation:
            break

    recall = recall_at_k(final_recommendations, ground_truth_urls, K)

    return {
        "trace": trace_name,
        "turns": turn_count,
        "turn_compliant": turn_count <= 8,
        "schema_errors": schema_errors,
        "groundedness_violations": groundedness_violations,
        "clarify_first_ok": clarify_first_ok,
        "ground_truth_urls": ground_truth_urls,
        "final_recommended_urls": final_recommendations,
        f"recall@{K}": recall,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    trace_files = sorted(
        f for f in os.listdir(TRACES_DIR) if f.endswith(".md")
    )
    if args.trace:
        trace_files = [f for f in trace_files if f.startswith(args.trace)]
        if not trace_files:
            print(f"No trace file found for: {args.trace}")
            sys.exit(1)

    agent = SHLAgent()
    results = []

    for fname in trace_files:
        trace_name = fname.replace(".md", "")
        path = os.path.join(TRACES_DIR, fname)
        with open(path, encoding="utf-8") as f:
            md_text = f.read()

        print(f"\n{'='*60}")
        print(f"Evaluating trace: {trace_name}")
        print(f"{'='*60}")

        result = evaluate_trace(trace_name, md_text, agent)
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        recall_key = f"recall@{K}"
        print(f"  Ground truth ({len(result['ground_truth_urls'])} items): "
              f"{[u.split('view/')[-1].rstrip('/') for u in result['ground_truth_urls']]}")
        print(f"  Agent recs   ({len(result['final_recommended_urls'])} items): "
              f"{[u.split('view/')[-1].rstrip('/') for u in result['final_recommended_urls']]}")
        print(f"  Recall@{K}:          {result[recall_key]:.3f}")
        print(f"  Turns used:          {result['turns']} (compliant: {result['turn_compliant']})")
        print(f"  Groundedness errors: {result['groundedness_violations']}")
        print(f"  Schema errors:       {result['schema_errors'] or 'none'}")
        print(f"  Clarify-first:       {result['clarify_first_ok']}")

    # Aggregate summary
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\nNo valid results.")
        return

    recall_key = f"recall@{K}"
    mean_recall = sum(r[recall_key] for r in valid) / len(valid)
    total_schema_errors = sum(len(r["schema_errors"]) for r in valid)
    total_groundedness = sum(r["groundedness_violations"] for r in valid)
    all_turn_compliant = all(r["turn_compliant"] for r in valid)
    clarify_rate = sum(r["clarify_first_ok"] for r in valid) / len(valid)

    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({len(valid)} traces)")
    print(f"{'='*60}")
    print(f"  Mean Recall@{K}:        {mean_recall:.3f}")
    print(f"  Total schema errors:  {total_schema_errors}")
    print(f"  Total groundedness:   {total_groundedness} hallucinated URLs")
    print(f"  Turn compliance:      {'PASS' if all_turn_compliant else 'FAIL'}")
    print(f"  Clarify-first rate:   {clarify_rate:.0%}")

    print(f"\n  Per-trace Recall@{K}:")
    for r in valid:
        bar = "#" * int(r[recall_key] * 20)
        print(f"    {r['trace']:5s} {bar:<20s} {r[recall_key]:.3f}")


if __name__ == "__main__":
    main()
