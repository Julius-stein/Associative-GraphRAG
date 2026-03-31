"""Answer-generation client and prompt builder."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from .config import load_llm_config
from .logging_utils import log


GENERATION_SYSTEM_PROMPT = """You are an expert at query-focused summarization.
Use the provided evidence package to synthesize a coherent answer to the query.
Prioritize multi-source aggregation, theme organization, and explicit uncertainty when evidence is thin.
Do not mention that the input came from a graph or a retrieval system unless the user asks.
Prefer breadth across well-supported themes over over-explaining one narrow thread."""


def build_generation_prompt(query, prompt_context, query_style="balanced"):
    """Build a QFS-oriented prompt from the evidence package and query style."""
    query_lower = query.lower()
    extra_constraints = []
    if any(term in query_lower for term in ["character", "characters", "passage", "passages", "narrative", "narratives"]):
        extra_constraints.append("- This query is about passage content. Prioritize people, events, themes, and historical or social forces over dataset or method descriptions.")
    if any(term in query_lower for term in ["historical", "history", "socio-political", "conflict", "conflicts"]):
        extra_constraints.append("- Emphasize comparative historical interpretation, not metadata about how the evidence was collected or modeled.")
    if query_style == "synthesis":
        extra_constraints.append("- Emphasize cross-source patterns, contrasts, and thematic structure.")
    elif query_style == "concrete":
        extra_constraints.append("- Prioritize concrete, practical, and directly supported points. Avoid broad contextual expansion unless it clearly helps answer the question.")
    extra_block = "\n".join(extra_constraints)
    return f"""Answer the query using the evidence package below.

Requirements:
- Write a concise but substantive query-focused summary.
- Organize the answer around major themes or aspects of the query.
- Synthesize across sources instead of listing isolated facts.
- Prefer themes supported by multiple groups, relations, or source chunks.
- Ignore isolated low-support peripheral items unless they clearly strengthen the answer.
- Focus on the substantive content of passages, characters, events, and themes.
- Unless the query explicitly asks about datasets, models, benchmarks, or methods, do not center the answer on metadata, annotation schemes, or NLP systems.
- If the evidence is partial or mixed, say so briefly.
- Do not fabricate details outside the evidence.
- Read the evidence package group-first rather than table-first.
- Treat each knowledge group dossier as one thematic slice of evidence.
- First identify the 2-4 knowledge groups that best match the query's main aspects.
- For each chosen group, read its summary and linked-source previews before consulting the full Sources table.
- Use the Sources table through linked source ids from the chosen groups, rather than summarizing the Root Chunks table by itself.
- Use Focused Entities and Focused Relations as optional indexes to clarify a group's content, not as a requirement to mention every listed item.
- You do not need to use every knowledge group. Prefer the best-supported groups and ignore peripheral groups that do not improve the answer.
- When multiple groups support the same theme, merge them into one synthesized point instead of repeating them separately.
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
        prompt = build_generation_prompt(record["query"], record["prompt_context"], query_style=record.get("query_style", "balanced"))
        answer = llm_client.generate(prompt)
        return {
            "group_id": record["group_id"],
            "query": record["query"],
            "query_style": record.get("query_style", "balanced"),
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
