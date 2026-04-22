"""Answer-generation client and prompt builder.

生成答案的 LLM 调用封装。
"""

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

def _theme_qfs_hints() -> str:
    return """Default answer shape for this query:
- Organize the answer into multiple query-relevant aspects/themes.
- Use representative examples to keep each aspect concrete and query-focused.
- Prefer broad but still evidence-grounded coverage over a narrow deep dive when the query asks about overall influence or emergence."""


def _theme_qfs_output_template() -> str:
    return """Theme answer shape:
- Output the answer using the following explicit structure:
  P1. Titles
  P2. Answer Outline
  P3. Queries, Summaries, and Evidence
  P4. Document Sections
  P5. Refinement
- In P1, list distinct aspect titles that jointly cover the strongest supported angles of the query.
- In P2, write one compact synthesis paragraph that directly answers the query and previews the main aspects.
- In P3, for each aspect, provide:
  ⟨Question⟩ one focused sub-question,
  ⟨Summary⟩ one compact evidence-grounded summary,
  ⟨Evidence⟩ the key supporting sources or concrete named anchors.
- In P4, briefly map the main supporting source regions or document bands when useful.
- In P5, refine the answer by noting how the aspects fit together and what uncertainty or gaps remain.
- Use the structure to force coverage: do not collapse several angles into one heading if the evidence supports distinct aspects.
## The structure is a tool to help you cover the query well, not a formality to check off. Use it when it helps, but do not feel compelled to fill every part if it does not fit the evidence. Always prioritize a clear, direct answer to the query that is well-grounded in the evidence.
"""


def _is_broad_qfs_query(query: str) -> bool:
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


def _is_action_planning_qfs_query(query: str) -> bool:
    query_lower = " ".join(query.lower().split())
    action_cues = (
        "how can",
        "how could",
        "how should",
        "how might",
        "what strategies",
        "what criteria",
        "what partnerships",
        "what assignments",
        "what methodologies",
        "what rights",
        "what prerequisites",
        "how do you measure",
        "how can you ensure",
        "how are educators",
    )
    planning_terms = (
        "education",
        "educational",
        "curriculum",
        "curriculums",
        "course",
        "courses",
        "student",
        "students",
        "workshop",
        "workshops",
        "module",
        "modules",
        "exhibition",
        "museum",
        "museums",
        "gallery",
        "galleries",
        "digital",
        "virtual",
        "partnership",
        "partnerships",
        "collaboration",
        "collaborations",
        "acquisition",
        "acquisitions",
    )
    return any(cue in query_lower for cue in action_cues) and any(term in query_lower for term in planning_terms)


def build_generation_prompt(query, prompt_context):
    """Build a QFS-oriented prompt from the evidence package.

    参数:
        query: 用户查询文本。
        prompt_context: 组织层构造的 evidence package 文本。
    返回:
        生成模型的完整 prompt 文本。

    构造给生成模型的问答提示，包含上下文与 QFS 任务提示。
    """
    extra_constraints = []
    if _is_broad_qfs_query(query):
        extra_constraints.append("Cover the strongest distinct evidence-backed aspects; include concrete named examples when available.")
    if _is_action_planning_qfs_query(query):
        extra_constraints.append("For planning/application questions, turn evidence into practical criteria, steps, risks, and evaluation measures.")
    extra_block = "\n".join(f"- {item}" for item in extra_constraints)
    return f"""Answer the query using only the evidence package.

Priorities, in order:
1. Directly answer the query.
2. Cover diverse evidence-backed aspects.
3. Synthesize sources into useful explanations or guidance.
4. Stay grounded; do not invent unsupported facts.

Use this compact structure:
P1. Titles: 4-7 aspect titles.
P2. Answer Outline: one direct synthesis paragraph.
P3. Evidence-backed Answer: one substantial section per aspect, with concrete examples, mechanisms, and source ids when useful.
P4. Source Coverage: briefly name the main evidence clusters used.
P5. Refinement: note uncertainty, gaps, or practical implications.

Use knowledge groups as a coverage map, not as mandatory headings. Ignore weak or irrelevant groups.
{extra_block}

Query:
{query}

Evidence Package:
{prompt_context}
"""


def build_multihop_qa_prompt(query, prompt_context):
    """Build a benchmark-style multi-hop QA prompt.

    Unlike the QFS prompt, this asks for a concise answer compatible with
    EM/F1 evaluation while still allowing one short evidence-grounded rationale.
    """
    return f"""Answer the multi-hop question using only the provided evidence.

Rules:
- The evidence is restricted to the benchmark candidate context for this question.
- Give the shortest correct answer possible on the first line.
- Then provide one brief explanation sentence if needed.
- Do not use the QFS P1-P5 structure.
- If the evidence does not support an answer, write: insufficient evidence.

Question:
{query}

Evidence:
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

    def generate(self, prompt, system_prompt=GENERATION_SYSTEM_PROMPT, temperature=0.1, max_tokens=2200):
        """Run one completion request with transient-error retries.

        参数:
            prompt: 用户提示文本。
            system_prompt: 系统角色提示内容。
            temperature: 生成温度。
            max_tokens: 最大生成长度。

        返回:
            LLM 生成的文本回复。
        """
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


def generate_one_answer_record(record, llm_client):
    """Generate one answer payload from one retrieval record.

    对单条检索记录生成最终答案，并返回带统计信息的结果。
    """
    if record.get("task_mode") == "multihop_qa":
        prompt = build_multihop_qa_prompt(record["query"], record["prompt_context"])
        answer = llm_client.generate(prompt, max_tokens=600)
    else:
        prompt = build_generation_prompt(
            record["query"],
            record["prompt_context"],
        )
        answer = llm_client.generate(prompt)
    return {
        "group_id": record["group_id"],
        "query": record["query"],
        "task_mode": record.get("task_mode", "qfs"),
        "context_constraint": record.get("context_constraint", "none"),
        "organization_mode": record.get("organization_mode", "qfs"),
        "model_answer": answer,
        "stats": record["stats"],
    }


def generate_answers(records, output_path, llm_client, max_workers=None):
    """Generate answers for a retrieval file in parallel and preserve order.

    If `output_path` already exists, completed records are reused and only the
    missing records are generated. Results are checkpointed after each finished
    record so a later retry can resume safely.

    参数:
        records: 检索结果记录列表，每条记录必须包含 group_id、query、prompt_context、stats 等字段。
        output_path: 输出 JSON 文件路径，用于保存/恢复生成结果。
        llm_client: LLM 客户端实例。
        max_workers: 并行生成线程数，默认使用客户端最大并发数。

    返回:
        生成后的答案记录列表。

    并行生成多个答案，支持断点重用与结果检查点保存。
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
        return generate_one_answer_record(record, llm_client)

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
