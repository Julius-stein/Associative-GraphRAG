"""FG-RAG-compatible LLM-as-a-judge utilities."""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from .logging_utils import log


JUDGE_SYSTEM_PROMPT = """---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
"""


FG_RAG_EVALUATE_PROMPT = """---Goal---

You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?

- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?

- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (ether Answer 1 or Answer 2) and explain why. Then, select an overal winner based on these three categories.

Here is the question:
{input_query}

Here are the two answers:

**Answer 1:**
{first_answer}

**Answer 2:**
{second_answer}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }} ,
    "Diversity": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }} ,
    "Empowerment": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }} ,
    "Overall Winner": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Summarize why ths answer is the overall winner based on the three criteria]" }}
}}
"""


def _extract_json(text):
    """Recover JSON even if the model wraps it in prose or code fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def build_judge_prompt(query, answer_a, answer_b):
    """Build the exact FG-RAG evaluation prompt template."""
    return FG_RAG_EVALUATE_PROMPT.format(
        input_query=query,
        first_answer=answer_a,
        second_answer=answer_b,
    )


def _normalize_winner(raw_value):
    value = (raw_value or "").strip().lower()
    if "tie" in value:
        return "tie"
    if "answer 1" in value or value == "1":
        return "a"
    if "answer 2" in value or value == "2":
        return "b"
    if "answer a" in value or value == "a":
        return "a"
    if "answer b" in value or value == "b":
        return "b"
    return "tie"


def _map_swapped_winner(winner):
    if winner == "a":
        return "b"
    if winner == "b":
        return "a"
    return "tie"


CRITERIA_KEYS = [
    "Comprehensiveness",
    "Diversity",
    "Empowerment",
    "Overall Winner",
]


def _map_letter_winner(winner):
    if winner == "a":
        return "candidate"
    if winner == "b":
        return "baseline"
    return "tie"


def _extract_dimension_votes(verdict_ab, verdict_ba_raw):
    """Map both evaluation orders back into candidate/baseline vote counts."""
    mapped = {}
    for key in CRITERIA_KEYS:
        winner_ab = _map_letter_winner(_normalize_winner(verdict_ab.get(key, {}).get("Winner", "")))
        winner_ba = _map_letter_winner(_map_swapped_winner(_normalize_winner(verdict_ba_raw.get(key, {}).get("Winner", ""))))
        votes = Counter([winner_ab, winner_ba])
        mapped[key] = {
            "candidate": votes["candidate"],
            "baseline": votes["baseline"],
            "tie": votes["tie"],
        }
    return mapped


def _fallback_verdict():
    """Fail-safe verdict so one malformed model output does not kill the whole run."""
    return {
        key: {
            "Winner": "Tie",
            "Explanation": "Evaluator output could not be parsed reliably; treated as a tie for robustness.",
        }
        for key in CRITERIA_KEYS
    }


def _generate_verdict(prompt, llm_client, max_attempts=3):
    """Retry parsing a judge response a few times before falling back to a tie."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = llm_client.generate(prompt, system_prompt=JUDGE_SYSTEM_PROMPT, temperature=0.0, max_tokens=900)
            return _extract_json(raw)
        except Exception as exc:
            last_error = exc
            log(f"[judge] parse failure attempt={attempt}/{max_attempts}: {exc}")
    log(f"[judge] falling back to tie verdict after repeated parse failures: {last_error}")
    return _fallback_verdict()


def judge_pair(query, candidate_answer, baseline_answer, llm_client):
    """Judge one answer pair in both orders to reduce position bias."""
    prompt_ab = build_judge_prompt(query, candidate_answer, baseline_answer)
    verdict_ab = _generate_verdict(prompt_ab, llm_client)
    prompt_ba = build_judge_prompt(query, baseline_answer, candidate_answer)
    verdict_ba_raw = _generate_verdict(prompt_ba, llm_client)

    overall_ab = _normalize_winner(verdict_ab.get("Overall Winner", {}).get("Winner", "Tie"))
    overall_ba = _map_swapped_winner(_normalize_winner(verdict_ba_raw.get("Overall Winner", {}).get("Winner", "Tie")))
    votes = Counter([overall_ab, overall_ba])
    if votes["a"] > votes["b"] and votes["a"] > votes["tie"]:
        final = "candidate"
    elif votes["b"] > votes["a"] and votes["b"] > votes["tie"]:
        final = "baseline"
    else:
        final = "tie"

    dimension_votes = _extract_dimension_votes(verdict_ab, verdict_ba_raw)

    return {
        "order_ab": verdict_ab,
        "order_ba": verdict_ba_raw,
        "mapped_overall_votes": {
            "candidate": votes["a"],
            "baseline": votes["b"],
            "tie": votes["tie"],
        },
        "dimension_votes": dimension_votes,
        "final_winner": final,
    }


def run_winrate_judgement(questions, candidate_answers, baseline_answers, llm_client, output_path=None, max_workers=None):
    """Run pairwise judging and aggregate both overall and per-dimension stats."""
    if len(candidate_answers) != len(baseline_answers):
        raise ValueError("Candidate and baseline answer counts do not match")
    max_workers = max_workers or llm_client.max_concurrency
    log(f"[judge] start total={len(candidate_answers)} model={llm_client.model} workers={max_workers}")
    verdicts = [None] * len(candidate_answers)
    summary_counter = Counter()
    dimension_counter = {
        key: Counter({"candidate": 0, "baseline": 0, "tie": 0}) for key in CRITERIA_KEYS
    }
    pairs = list(zip(questions, candidate_answers, baseline_answers))

    def _one(item):
        idx, (query, cand, base) = item
        verdict = judge_pair(query, cand["model_answer"], base["model_answer"], llm_client)
        return idx, {
            "index": idx,
            "query": query,
            "candidate_answer": cand["model_answer"],
            "baseline_answer": base["model_answer"],
            **verdict,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_one, item): item[0] for item in enumerate(pairs, start=1)}
        completed = 0
        for future in as_completed(future_map):
            idx, verdict_payload = future.result()
            verdicts[idx - 1] = verdict_payload
            summary_counter[verdict_payload["final_winner"]] += 1
            for key, votes in verdict_payload["dimension_votes"].items():
                for label, count in votes.items():
                    dimension_counter[key][label] += count
            completed += 1
            log(
                f"[judge {completed}/{len(candidate_answers)}] q{idx:03d} "
                f"result={verdict_payload['final_winner']} "
                f"votes={verdict_payload['mapped_overall_votes']}"
            )
    total = len(verdicts)
    dimension_summary = {}
    for key, counter in dimension_counter.items():
        total_votes = counter["candidate"] + counter["baseline"] + counter["tie"]
        dimension_summary[key] = {
            "candidate": counter["candidate"],
            "baseline": counter["baseline"],
            "tie": counter["tie"],
            "candidate_probability": round(counter["candidate"] / max(total_votes, 1), 4),
            "baseline_probability": round(counter["baseline"] / max(total_votes, 1), 4),
            "tie_probability": round(counter["tie"] / max(total_votes, 1), 4),
            "total_votes": total_votes,
        }
    payload = {
        "summary": {
            "total": total,
            "candidate_wins": summary_counter["candidate"],
            "baseline_wins": summary_counter["baseline"],
            "ties": summary_counter["tie"],
            "candidate_win_rate": round(summary_counter["candidate"] / max(total, 1), 4),
            "baseline_win_rate": round(summary_counter["baseline"] / max(total, 1), 4),
        },
        "criteria_summary": dimension_summary,
        "verdicts": verdicts,
    }
    if output_path is not None:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[judge] wrote {output_path}")
    return payload
