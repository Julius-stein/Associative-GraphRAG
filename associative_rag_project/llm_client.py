"""Answer-generation client and prompt builder."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

from openai import OpenAI

from .config import load_llm_config
from .logging_utils import log


GENERATION_SYSTEM_PROMPT = """You are an expert at evidence-grounded question answering.
Answer the user's query directly using the provided evidence package.
Match the shape of the answer to the query: explain mechanisms when asked how, compare when asked to compare, and name concrete examples when the query points to people, periods, sections, or items.
Synthesize across sources when it helps, but do not inflate the answer into a broad thematic essay unless the query truly calls for that.
State uncertainty briefly when evidence is thin or mixed.
Do not mention that the input came from a graph or a retrieval system unless the user asks."""


def _contract_template_hints(query_contract: str) -> str:
    if query_contract == "section-grounded":
        return """Default answer shape for this query:
- Prefer following the relevant sections, periods, parts, or local evidence bands.
- Keep the answer anchored to those units instead of drifting into a free-form overview."""
    if query_contract == "mechanism-grounded":
        return """Default answer shape for this query:
- Prefer explaining what drove the change, how it operated, and what it affected.
- Keep causal links explicit instead of replacing them with broad topical coverage."""
    if query_contract == "comparison-grounded":
        return """Default answer shape for this query:
- Prefer an explicit side-by-side comparison or clear comparison dimensions.
- Surface both commonalities and differences when the evidence supports them."""
    return """Default answer shape for this query:
- Organize the answer into multiple query-relevant aspects/themes.
- Use representative examples to keep each aspect concrete and query-focused.
- Prefer broad but still evidence-grounded coverage over a narrow deep dive when the query asks about overall influence or emergence."""


def _theme_qfs_output_template(query_contract: str) -> str:
    if query_contract != "theme-grounded":
        return ""
    return """Theme QFS Response Template (use these exact section headers):
P1. Titles
- Provide 4-8 concise aspect titles that directly answer the query.

P2. Answer Outline
- Give a 2-4 sentence high-level synthesis across aspects.

P3. Queries, Summaries, and Evidence
- For each aspect, use one tuple with this structure:
  ⟨Question⟩ aspect-specific sub-question
  ⟨Summary⟩ 2-4 sentence focused summary
  ⟨Evidence⟩ 2-4 concrete evidence points (named items, events, periods, institutions, works, or relations)

P4. Document Sections
- Map each aspect to the most relevant source sections/chunks/groups in the evidence package.
- Keep this concise and concrete (avoid generic statements like “many sources mention this”).

P5. Refinement
- For each tuple, refine the summary and evidence by removing weak/duplicated points and keeping only query-relevant support.
- Briefly note uncertainty where evidence is thin or mixed.
"""


def _is_broad_theme_query(query: str, query_contract: str) -> bool:
    if query_contract != "theme-grounded":
        return False
    query_lower = " ".join(query.lower().split())
    broad_cues = (
        "what are the primary reasons",
        "what are the reasons",
        "what are the benefits",
        "what are the potential benefits",
        "what strategies",
        "what essential aspects",
        "what aspects",
        "what factors",
        "what themes",
        "what examples",
        "which external resources",
        "how did",
        "influence the emergence",
        "influence the development",
        "across",
        "various",
    )
    return any(cue in query_lower for cue in broad_cues)


def build_generation_prompt(query, prompt_context, query_contract="theme-grounded"):
    """Build a QFS-oriented prompt from the evidence package."""
    query_lower = query.lower()
    extra_constraints = []
    if any(term in query_lower for term in ["character", "characters", "passage", "passages", "narrative", "narratives"]):
        extra_constraints.append("- This query is about passage content. Prioritize people, events, themes, and historical or social forces over dataset or method descriptions.")
    if any(term in query_lower for term in ["historical", "history", "socio-political", "conflict", "conflicts"]):
        extra_constraints.append("- Emphasize comparative historical interpretation, not metadata about how the evidence was collected or modeled.")
    if _is_broad_theme_query(query, query_contract):
        extra_constraints.extend(
            [
                "- This is a broad theme query: prioritize breadth of supported aspects, not just one tight storyline.",
                "- Cover at least 5 distinct, query-relevant aspects if evidence supports them.",
                "- For each aspect, include at least one concrete named example (movement/artist/event/policy/institution/time period).",
                "- Include cross-period or cross-context variety when evidence supports it.",
                "- Use the coverage checklist proactively to avoid missing major supported angles.",
            ]
        )
    extra_block = "\n".join(extra_constraints)
    contract_hint_block = _contract_template_hints(query_contract)
    theme_qfs_template_block = _theme_qfs_output_template(query_contract)
    return f"""Answer the query using the evidence package below.

Requirements:
- Write a substantive query-focused summary.
- Answer the question in the form it asks for.
- Start from the most direct answer, then add supporting synthesis only where it helps.
- Synthesize across sources instead of listing isolated facts, but keep concrete examples and named items when they are central to the query.
- Ignore isolated low-support context items unless they clearly strengthen the answer.
- Focus on the substantive content of passages, characters, events, and themes.
- Unless the query explicitly asks about datasets, models, benchmarks, or methods, do not center the answer on metadata, annotation schemes, or NLP systems.
- If the evidence is partial or mixed, say so briefly.
- Do not fabricate details outside the evidence.
- Use the facet groups and linked sources as evidence aids, not as a script for the final prose.
- Let strong anchor evidence determine the main points; do not let a broad facet summary dominate if its support is weak for this query.
- You do not need to use every facet group. Select the evidence that best answers the question.
- Merge overlapping evidence when it supports the same point, but do not merge away distinctions the query cares about.
- Read the facet summaries, linked-source previews, and sources together; avoid summarizing the Root Chunks table by itself.
- Use Focused Entities and Focused Relations only when they help clarify the answer.

Contract-Specific Hints:
{contract_hint_block}
{extra_block}
{theme_qfs_template_block}

Query:
{query}

Evidence Package:
{prompt_context}
"""


class OpenAICompatibleClient:
    """Thin wrapper around an OpenAI-compatible chat endpoint."""

    def __init__(self, llm_config=None):
        self.config = llm_config or load_llm_config()
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"],
            timeout=self.config.get("timeout", 120),
            max_retries=self.config.get("max_retries", 3),
        )
        self.model = self.config["model"]
        self.max_concurrency = self.config.get("max_concurrency", 4)
        self.generation_max_attempts = self.config.get("generation_max_attempts", 6)
        self.retry_backoff_seconds = self.config.get("retry_backoff_seconds", 2.0)

    def _is_retryable_error(self, exc):
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True
        name = exc.__class__.__name__.lower()
        text = str(exc).lower()
        retryable_tokens = (
            "timeout",
            "ratelimit",
            "rate limit",
            "internalservererror",
            "apierror",
            "connection",
            "temporar",
            "server error",
            "unknown error",
        )
        return any(token in name or token in text for token in retryable_tokens)

    def generate(self, prompt, system_prompt=GENERATION_SYSTEM_PROMPT, temperature=0.1, max_tokens=1500):
        """Run one completion request with transient-error retries."""
        last_error = None
        for attempt in range(1, self.generation_max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                if not self._is_retryable_error(exc) or attempt >= self.generation_max_attempts:
                    raise
                wait_seconds = self.retry_backoff_seconds * (2 ** (attempt - 1))
                log(
                    f"[llm] transient generation error attempt={attempt}/{self.generation_max_attempts} "
                    f"wait={wait_seconds:.1f}s error={exc.__class__.__name__}: {exc}"
                )
                sleep(wait_seconds)
        raise last_error


def generate_answers(records, output_path, llm_client, max_workers=None):
    """Generate answers for a retrieval file in parallel and preserve order.

    If `output_path` already exists, completed records are reused and only the
    missing records are generated. Results are checkpointed after each finished
    record so a later retry can resume safely.
    """
    max_workers = max_workers or llm_client.max_concurrency
    output_path = Path(output_path) if output_path is not None else None
    log(f"[answer] start total={len(records)} workers={max_workers} model={llm_client.model}")

    results = [None] * len(records)
    if output_path is not None and output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            existing_by_group = {
                item.get("group_id"): item
                for item in existing
                if isinstance(item, dict) and item.get("group_id") and item.get("model_answer")
            }
            reused = 0
            for idx, record in enumerate(records):
                cached = existing_by_group.get(record["group_id"])
                if cached is not None:
                    results[idx] = cached
                    reused += 1
            if reused:
                log(f"[answer] resume found {reused}/{len(records)} completed answers in {output_path}")
        except Exception as exc:
            log(f"[answer] could not load existing answers from {output_path}: {exc}")

    def _one(record):
        prompt = build_generation_prompt(
            record["query"],
            record["prompt_context"],
            query_contract=record.get("query_contract", "theme-grounded"),
        )
        answer = llm_client.generate(prompt)
        return {
            "group_id": record["group_id"],
            "query": record["query"],
            "query_contract": record.get("query_contract", "theme-grounded"),
            "model_answer": answer,
            "stats": record["stats"],
        }

    pending = [(idx, record) for idx, record in enumerate(records) if results[idx] is None]
    if not pending:
        log("[answer] all answers already present; nothing to generate")
        if output_path is not None:
            output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"[answer] wrote {output_path}")
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_one, record): idx for idx, record in pending}
        completed = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                results[idx] = future.result()
            except Exception:
                if output_path is not None:
                    ordered_results = [item for item in results if item is not None]
                    output_path.write_text(json.dumps(ordered_results, ensure_ascii=False, indent=2), encoding="utf-8")
                    log(f"[answer] checkpointed partial results to {output_path} before raising")
                raise
            completed += 1
            log(
                f"[answer {completed}/{len(pending)}] {results[idx]['group_id']} "
                f"chars={len(results[idx]['model_answer'])}"
            )
            if output_path is not None:
                serializable_results = [item for item in results if item is not None]
                ordered_results = []
                for item in results:
                    if item is not None:
                        ordered_results.append(item)
                output_path.write_text(json.dumps(ordered_results, ensure_ascii=False, indent=2), encoding="utf-8")

    if output_path is not None:
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[answer] wrote {output_path}")
    return results
