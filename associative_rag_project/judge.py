"""FG-RAG-compatible LLM-as-a-judge utilities."""

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .logging_utils import log


JUDGE_SYSTEM_PROMPT = """---Role---
You are an expert tasked with evaluating two answers to the same question.
You must also identify what kind of organization the query requires and what kind of organization each answer actually uses.
"""


FG_RAG_EVALUATE_PROMPT = """---Goal---

You will evaluate two answers to the same question based on:

- **Comprehensiveness**
- **Diversity**
- **Empowerment**
- **Focus Match**
- **Evidence Anchoring**
- **Scope Discipline**
- **Scenario Fidelity**

First classify the query's required organization contract as exactly one of:
- `section-grounded`
- `mechanism-grounded`
- `comparison-grounded`
- `theme-grounded`

Then classify each answer's dominant organization type using the same four labels.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?

- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?

- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

- **Focus Match**: How well does the answer's organization match what the query actually requires?
  Examples:
  - if the query needs section-grounded organization, a thematic overview is a mismatch
  - if the query needs mechanism-grounded organization, generic advice is a mismatch
  - if the query needs comparison-grounded organization, a broad summary without clear contrasts is a mismatch

- **Evidence Anchoring**: How well is the answer's chosen organization stably anchored in corpus-specific evidence rather than only sounding plausible?

- **Scope Discipline**: Which answer better stays within the scope of the query itself?
  Reward answers that stay within the asked entities, time range, section range, comparison axes, and task boundary.
  Penalize answers that expand into clearly unasked territory just to sound broader.

- **Scenario Fidelity**: Which answer better avoids inventing extra situations, motives, workflows, user settings, or narrative setups not required by the query?
  Reward answers that stay at the level the query supports.
  Penalize answers that "set the scene" or add specific contexts without clear need.

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why.
Then select an Overall Winner using a contract-aware standard:

- For `theme-grounded` queries, prioritize Comprehensiveness, Diversity, and Focus Match, but still penalize scope drift and invented scenarios.
- For `mechanism-grounded` queries, prioritize Focus Match, Empowerment, Evidence Anchoring, Scope Discipline, and Scenario Fidelity.
- For `section-grounded` queries, prioritize Focus Match, Evidence Anchoring, Scope Discipline, and Scenario Fidelity, while staying anchored to the relevant sections or local evidence bands.
- For `comparison-grounded` queries, prioritize Focus Match, Comprehensiveness, Empowerment, and Scope Discipline, while keeping the comparison axes explicit and avoiding invented framing.

Here is the question:
{input_query}

Here are the two answers:

**Answer 1:**
{first_answer}

**Answer 2:**
{second_answer}

Evaluate both answers using the criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "Query Organization Need": {{ "Label": "[section-grounded | mechanism-grounded | comparison-grounded | theme-grounded]", "Explanation": "[Why this query needs that organization]" }},
    "Answer 1 Organization": {{ "Label": "[section-grounded | mechanism-grounded | comparison-grounded | theme-grounded]", "Explanation": "[What organization Answer 1 actually uses]" }},
    "Answer 2 Organization": {{ "Label": "[section-grounded | mechanism-grounded | comparison-grounded | theme-grounded]", "Explanation": "[What organization Answer 2 actually uses]" }},
    "Comprehensiveness": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }} ,
    "Diversity": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }} ,
    "Empowerment": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }} ,
    "Focus Match": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Which answer better matches the query's required organization and why]" }} ,
    "Evidence Anchoring": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Which answer is better anchored in evidence for its chosen organization and why]" }} ,
    "Scope Discipline": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Which answer better stays within the query scope and avoids irrelevant expansion]" }} ,
    "Scenario Fidelity": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Which answer better avoids inventing extra situations or setups]" }} ,
    "Overall Winner": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Summarize why this answer is the overall winner under the contract-aware standard]" }}
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
    "Focus Match",
    "Evidence Anchoring",
    "Scope Discipline",
    "Scenario Fidelity",
    "Overall Winner",
]

WINRATE_TABLE_METRICS = [
    "Overall Winner",
    "Comprehensiveness",
    "Diversity",
    "Empowerment",
    "Focus Match",
    "Evidence Anchoring",
    "Scope Discipline",
    "Scenario Fidelity",
]

NON_OVERALL_CRITERIA_KEYS = [key for key in CRITERIA_KEYS if key != "Overall Winner"]

ORGANIZATION_LABELS = {
    "section-grounded",
    "mechanism-grounded",
    "comparison-grounded",
    "theme-grounded",
}

CONTRACT_PRIMARY_METRICS = {
    "theme-grounded": [
        "Comprehensiveness",
        "Diversity",
        "Focus Match",
        "Scope Discipline",
        "Scenario Fidelity",
    ],
    "mechanism-grounded": [
        "Focus Match",
        "Empowerment",
        "Evidence Anchoring",
        "Scope Discipline",
        "Scenario Fidelity",
    ],
    "section-grounded": [
        "Focus Match",
        "Evidence Anchoring",
        "Scope Discipline",
        "Scenario Fidelity",
    ],
    "comparison-grounded": [
        "Focus Match",
        "Comprehensiveness",
        "Empowerment",
        "Scope Discipline",
        "Scenario Fidelity",
    ],
}


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


def _normalize_contract_label(raw_value):
    value = (raw_value or "").strip().lower()
    if value in ORGANIZATION_LABELS:
        return value
    if "section" in value:
        return "section-grounded"
    if "mechan" in value or "cause" in value or "process" in value:
        return "mechanism-grounded"
    if "compar" in value or "contrast" in value:
        return "comparison-grounded"
    return "theme-grounded"


def _extract_organization_analysis(verdict_ab, verdict_ba_raw):
    query_contract_ab = _normalize_contract_label(verdict_ab.get("Query Organization Need", {}).get("Label", ""))
    query_contract_ba = _normalize_contract_label(verdict_ba_raw.get("Query Organization Need", {}).get("Label", ""))
    query_contract = query_contract_ab if query_contract_ab == query_contract_ba else query_contract_ab
    candidate_labels = [
        _normalize_contract_label(verdict_ab.get("Answer 1 Organization", {}).get("Label", "")),
        _normalize_contract_label(verdict_ba_raw.get("Answer 2 Organization", {}).get("Label", "")),
    ]
    baseline_labels = [
        _normalize_contract_label(verdict_ab.get("Answer 2 Organization", {}).get("Label", "")),
        _normalize_contract_label(verdict_ba_raw.get("Answer 1 Organization", {}).get("Label", "")),
    ]
    candidate_contract = Counter(candidate_labels).most_common(1)[0][0]
    baseline_contract = Counter(baseline_labels).most_common(1)[0][0]
    return {
        "query_contract": query_contract,
        "candidate_answer_contract": candidate_contract,
        "baseline_answer_contract": baseline_contract,
        "candidate_matches_query_contract": candidate_contract == query_contract,
        "baseline_matches_query_contract": baseline_contract == query_contract,
    }


def _winner_from_counter(counter):
    if counter["candidate"] > counter["baseline"] and counter["candidate"] > counter["tie"]:
        return "candidate"
    if counter["baseline"] > counter["candidate"] and counter["baseline"] > counter["tie"]:
        return "baseline"
    return "tie"


def _aggregate_metric_votes(metric_names, dimension_votes):
    counter = Counter({"candidate": 0, "baseline": 0, "tie": 0})
    for metric_name in metric_names:
        votes = dimension_votes.get(metric_name, {})
        counter["candidate"] += votes.get("candidate", 0)
        counter["baseline"] += votes.get("baseline", 0)
        counter["tie"] += votes.get("tie", 0)
    return counter


def _resolve_contract_conditioned_decision(query_contract, dimension_votes, llm_overall_votes):
    primary_metrics = CONTRACT_PRIMARY_METRICS.get(query_contract, CONTRACT_PRIMARY_METRICS["theme-grounded"])
    primary_counter = _aggregate_metric_votes(primary_metrics, dimension_votes)
    primary_winner = _winner_from_counter(primary_counter)
    return {
        "winner": primary_winner,
        "decided_by": "primary_metrics_only",
        "primary_metrics": primary_metrics,
        "primary_vote_totals": dict(primary_counter),
        "llm_overall_vote_totals": dict(llm_overall_votes),
    }


def _metric_pass_summary(metric_names, dimension_summary):
    summary = {}
    for metric_name in metric_names:
        metric = dimension_summary.get(metric_name, {})
        candidate_probability = metric.get("candidate_probability", 0.0)
        summary[metric_name] = {
            "candidate_probability": candidate_probability,
            "passes_threshold": candidate_probability > 0.5,
        }
    return summary


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
        llm_overall_winner = "candidate"
    elif votes["b"] > votes["a"] and votes["b"] > votes["tie"]:
        llm_overall_winner = "baseline"
    else:
        llm_overall_winner = "tie"

    dimension_votes = _extract_dimension_votes(verdict_ab, verdict_ba_raw)
    organization_analysis = _extract_organization_analysis(verdict_ab, verdict_ba_raw)
    contract_conditioned = _resolve_contract_conditioned_decision(
        organization_analysis["query_contract"],
        dimension_votes,
        {
            "candidate": votes["a"],
            "baseline": votes["b"],
            "tie": votes["tie"],
        },
    )

    return {
        "order_ab": verdict_ab,
        "order_ba": verdict_ba_raw,
        "organization_analysis": organization_analysis,
        "mapped_overall_votes": {
            "candidate": votes["a"],
            "baseline": votes["b"],
            "tie": votes["tie"],
        },
        "llm_overall_winner": llm_overall_winner,
        "dimension_votes": dimension_votes,
        "contract_conditioned_decision": contract_conditioned,
        "final_winner": contract_conditioned["winner"],
    }


def run_winrate_judgement(questions, candidate_answers, baseline_answers, llm_client, output_path=None, max_workers=None):
    """Run pairwise judging and aggregate both overall and per-dimension stats."""
    if len(candidate_answers) != len(baseline_answers):
        raise ValueError("Candidate and baseline answer counts do not match")
    max_workers = max_workers or llm_client.max_concurrency
    log(f"[judge] start total={len(candidate_answers)} model={llm_client.model} workers={max_workers}")
    verdicts = [None] * len(candidate_answers)
    summary_counter = Counter()
    llm_summary_counter = Counter()
    dimension_counter = {
        key: Counter({"candidate": 0, "baseline": 0, "tie": 0}) for key in CRITERIA_KEYS
    }
    dimension_counter_by_contract = {}
    contract_conditioned_counter_by_contract = defaultdict(Counter)
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
            llm_summary_counter[verdict_payload["llm_overall_winner"]] += 1
            for key, votes in verdict_payload["dimension_votes"].items():
                for label, count in votes.items():
                    dimension_counter[key][label] += count
            query_contract = verdict_payload["organization_analysis"]["query_contract"]
            if query_contract not in dimension_counter_by_contract:
                dimension_counter_by_contract[query_contract] = {
                    key: Counter({"candidate": 0, "baseline": 0, "tie": 0}) for key in CRITERIA_KEYS
                }
            for key, votes in verdict_payload["dimension_votes"].items():
                for label, count in votes.items():
                    dimension_counter_by_contract[query_contract][key][label] += count
            contract_conditioned_counter_by_contract[query_contract][verdict_payload["final_winner"]] += 1
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
    dimension_summary_by_contract = {}
    for query_contract, counters in dimension_counter_by_contract.items():
        dimension_summary_by_contract[query_contract] = {}
        for key, counter in counters.items():
            total_votes = counter["candidate"] + counter["baseline"] + counter["tie"]
            dimension_summary_by_contract[query_contract][key] = {
                "candidate": counter["candidate"],
                "baseline": counter["baseline"],
                "tie": counter["tie"],
                "candidate_probability": round(counter["candidate"] / max(total_votes, 1), 4),
                "baseline_probability": round(counter["baseline"] / max(total_votes, 1), 4),
                "tie_probability": round(counter["tie"] / max(total_votes, 1), 4),
                "total_votes": total_votes,
            }
    organization_summary = {
        "query_contracts": Counter(),
        "candidate_answer_contracts": Counter(),
        "baseline_answer_contracts": Counter(),
        "candidate_contract_matches": 0,
        "baseline_contract_matches": 0,
    }
    for verdict in verdicts:
        analysis = verdict["organization_analysis"]
        organization_summary["query_contracts"][analysis["query_contract"]] += 1
        organization_summary["candidate_answer_contracts"][analysis["candidate_answer_contract"]] += 1
        organization_summary["baseline_answer_contracts"][analysis["baseline_answer_contract"]] += 1
        organization_summary["candidate_contract_matches"] += int(analysis["candidate_matches_query_contract"])
        organization_summary["baseline_contract_matches"] += int(analysis["baseline_matches_query_contract"])
    payload = {
        "summary": {
            "total": total,
            "candidate_wins": summary_counter["candidate"],
            "baseline_wins": summary_counter["baseline"],
            "ties": summary_counter["tie"],
            "candidate_win_rate": round(summary_counter["candidate"] / max(total, 1), 4),
            "baseline_win_rate": round(summary_counter["baseline"] / max(total, 1), 4),
        },
        "llm_overall_summary": {
            "total": total,
            "candidate_wins": llm_summary_counter["candidate"],
            "baseline_wins": llm_summary_counter["baseline"],
            "ties": llm_summary_counter["tie"],
            "candidate_win_rate": round(llm_summary_counter["candidate"] / max(total, 1), 4),
            "baseline_win_rate": round(llm_summary_counter["baseline"] / max(total, 1), 4),
        },
        "criteria_summary": dimension_summary,
        "criteria_summary_by_contract": dimension_summary_by_contract,
        "contract_conditioned_summary_by_contract": {
            query_contract: {
                "total": counts["candidate"] + counts["baseline"] + counts["tie"],
                "candidate_wins": counts["candidate"],
                "baseline_wins": counts["baseline"],
                "ties": counts["tie"],
                "candidate_win_rate": round(
                    counts["candidate"] / max(counts["candidate"] + counts["baseline"] + counts["tie"], 1), 4
                ),
                "baseline_win_rate": round(
                    counts["baseline"] / max(counts["candidate"] + counts["baseline"] + counts["tie"], 1), 4
                ),
                "primary_metrics": CONTRACT_PRIMARY_METRICS.get(query_contract, CONTRACT_PRIMARY_METRICS["theme-grounded"]),
                "primary_metric_summary": _metric_pass_summary(
                    CONTRACT_PRIMARY_METRICS.get(query_contract, CONTRACT_PRIMARY_METRICS["theme-grounded"]),
                    dimension_summary_by_contract.get(query_contract, {}),
                ),
                "all_primary_metrics_above_50": all(
                    dimension_summary_by_contract.get(query_contract, {})
                    .get(metric_name, {})
                    .get("candidate_probability", 0.0)
                    > 0.5
                    for metric_name in CONTRACT_PRIMARY_METRICS.get(
                        query_contract, CONTRACT_PRIMARY_METRICS["theme-grounded"]
                    )
                ),
                "overall_win_above_50": round(
                    counts["candidate"] / max(counts["candidate"] + counts["baseline"] + counts["tie"], 1), 4
                )
                > 0.5,
            }
            for query_contract, counts in contract_conditioned_counter_by_contract.items()
        },
        "organization_summary": {
            "query_contracts": dict(organization_summary["query_contracts"]),
            "candidate_answer_contracts": dict(organization_summary["candidate_answer_contracts"]),
            "baseline_answer_contracts": dict(organization_summary["baseline_answer_contracts"]),
            "candidate_contract_match_rate": round(organization_summary["candidate_contract_matches"] / max(total, 1), 4),
            "baseline_contract_match_rate": round(organization_summary["baseline_contract_matches"] / max(total, 1), 4),
        },
        "verdicts": verdicts,
    }
    if output_path is not None:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[judge] wrote {output_path}")
    return payload


def build_winrate_table_rows(payload):
    """Normalize overall + per-criterion win rates into row objects for table rendering."""
    rows = []
    overall = payload.get("summary", {})
    rows.append(
        {
            "metric": "Overall Winner",
            "candidate_win_rate": overall.get("candidate_win_rate", 0.0),
            "baseline_win_rate": overall.get("baseline_win_rate", 0.0),
            "tie_rate": round(overall.get("ties", 0) / max(overall.get("total", 1), 1), 4),
            "total_votes": overall.get("total", 0),
        }
    )
    criteria_summary = payload.get("criteria_summary", {})
    for metric_name in WINRATE_TABLE_METRICS[1:]:
        metric = criteria_summary.get(metric_name, {})
        rows.append(
            {
                "metric": metric_name,
                "candidate_win_rate": metric.get("candidate_probability", 0.0),
                "baseline_win_rate": metric.get("baseline_probability", 0.0),
                "tie_rate": metric.get("tie_probability", 0.0),
                "total_votes": metric.get("total_votes", 0),
            }
        )
    return rows


def render_winrate_markdown_table(payload, candidate_label, baseline_label, title=None):
    """Render a compact Markdown win-rate table for one baseline comparison."""
    rows = build_winrate_table_rows(payload)
    lines = []
    if title:
        lines.append(f"## {title}")
        lines.append("")
    lines.append(f"| Metric | {candidate_label} | {baseline_label} | Tie | Total |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['candidate_win_rate']:.4f} | "
            f"{row['baseline_win_rate']:.4f} | {row['tie_rate']:.4f} | {row['total_votes']} |"
        )
    return "\n".join(lines)
