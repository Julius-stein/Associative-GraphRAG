"""Answer-generation client and prompt builder."""

from concurrent.futures import ThreadPoolExecutor, as_completed

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
- Group the answer by the strongest recurring themes only when that actually matches the question.
- Use representative examples to keep each theme concrete and query-focused."""


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
                "- This query supports broader coverage, but only include aspects that are clearly relevant to the wording of the question.",
                "- Cover the main supported aspects before adding secondary ones, and stop once the query is answered.",
                "- Use the coverage checklist as a reminder, not as an obligation to expand the answer.",
            ]
        )
    extra_block = "\n".join(extra_constraints)
    contract_hint_block = _contract_template_hints(query_contract)
    return f"""Answer the query using the evidence package below.

Requirements:
- Write a concise but substantive query-focused summary.
- Answer the question in the form it asks for; do not default to a broad thematic survey.
- Start from the most direct answer, then add supporting synthesis only where it helps.
- Synthesize across sources instead of listing isolated facts, but keep concrete examples and named items when they are central to the query.
- Ignore isolated low-support peripheral items unless they clearly strengthen the answer.
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

    def generate(self, prompt, system_prompt=GENERATION_SYSTEM_PROMPT, temperature=0.1, max_tokens=1500):
        """Run one completion request."""
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


def generate_answers(records, output_path, llm_client, max_workers=None):
    """Generate answers for a retrieval file in parallel and preserve order."""
    max_workers = max_workers or llm_client.max_concurrency
    log(f"[answer] start total={len(records)} workers={max_workers} model={llm_client.model}")

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

    results = [None] * len(records)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_one, record): idx for idx, record in enumerate(records)}
        completed = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
            completed += 1
            log(
                f"[answer {completed}/{len(records)}] {results[idx]['group_id']} "
                f"chars={len(results[idx]['model_answer'])}"
            )

    if output_path is not None:
        output_path.write_text(__import__("json").dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[answer] wrote {output_path}")
    return results
