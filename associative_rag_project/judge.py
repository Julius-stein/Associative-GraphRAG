"""LLM-as-a-judge utilities.

用于对生成答案与基线答案进行对比评判的工具集。
"""

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import load_llm_config
from .embedding_client import OpenAICompatibleEmbeddingClient
from .logging_utils import log
from .data import load_graph_corpus
from .retrieval import BM25Index, DenseChunkIndex, HybridChunkRetriever


JUDGE_SYSTEM_PROMPT = """---Role---
You are an expert tasked with evaluating two answers to the same question.
Evaluate them as query-focused summarization answers.
"""


FG_RAG_EVALUATE_PROMPT = """---Goal---

You will evaluate two answers to the same question based on:

- **Comprehensiveness**
- **Diversity**
- **Empowerment**

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?

- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?

- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why.
Then select an Overall Winner for the QFS task. The better QFS answer should be more comprehensive, more diverse in coverage, and more empowering/useful to the reader while staying faithful to the question.

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
    "Comprehensiveness": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }} ,
    "Diversity": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }} ,
    "Empowerment": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }} ,
    "Overall Winner": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Summarize why this answer is the better QFS answer]" }}
}}
"""

SOURCE_COMPLIANCE_PROMPT = """---Goal---

You will evaluate two answers to the same source-bounded QFS question.

This is NOT the original QFS quality judge. Do not judge which answer is more polished or more comprehensive in general.
Judge which answer better follows a source-bounded answer contract:
- It should primarily answer from the provided corpus or document collection implied by the question.
- It may use external background knowledge only if clearly marked as external context.
- It should not present unsupported outside knowledge as if it came from the corpus.
- It should not be penalized merely for high-level synthesis when multiple pieces of evidence could jointly support an abstraction.
- It should acknowledge uncertainty or missing evidence when the corpus scope appears insufficient.

Evaluate the answers on:

- **Evidence Usefulness**: Which answer appears to make more meaningful use of source evidence rather than generic background knowledge?
- **Source Grounding**: Which answer is more careful about tying major conclusions to the source-bounded task? Do not require extractive wording.
- **Source Layering**: Which answer better separates corpus-supported conclusions, external background, and speculation/uncertainty?
- **Gap Awareness**: Which answer better acknowledges limits, missing evidence, or uncertainty instead of overclaiming?

For each criterion, choose the better answer (Answer 1, Answer 2, or Tie) and explain why.
Then select a Source Compliance Winner.

Question:
{input_query}

Answer 1:
{first_answer}

Answer 2:
{second_answer}

Output strict JSON in this format:

{{
    "Evidence Usefulness": {{ "Winner": "[Answer 1 or Answer 2 or Tie]", "Explanation": "[Provide explanation here]" }},
    "Source Grounding": {{ "Winner": "[Answer 1 or Answer 2 or Tie]", "Explanation": "[Provide explanation here]" }},
    "Source Layering": {{ "Winner": "[Answer 1 or Answer 2 or Tie]", "Explanation": "[Provide explanation here]" }},
    "Gap Awareness": {{ "Winner": "[Answer 1 or Answer 2 or Tie]", "Explanation": "[Provide explanation here]" }},
    "Source Compliance Winner": {{ "Winner": "[Answer 1 or Answer 2 or Tie]", "Explanation": "[Summarize which answer better follows the source-bounded QFS contract]" }}
}}
"""

CLAIM_SYSTEM_PROMPT = """You extract and verify corpus-checkable claims from QFS answers.
Return strict JSON only."""

CLAIM_EXTRACTION_PROMPT = """Goal:
- Extract every core, corpus-checkable claim needed to cover the substantive answer.
- Do not impose a maximum number of claims. Continue until the core factual content is covered.
- Prefer high recall over brevity, but skip trivial restatements and purely rhetorical framing.

Rules:
- Keep claims atomic, specific, and corpus-checkable.
- The answer may use a P1-P5 scaffold. Ignore P1/P2/P3/P4/P5 labels, Titles, Answer Outline, Queries/Summaries/Evidence headings, Document Sections, Refinement notes, source labels, and other formatting shell text.
- Ignore claims about the retrieval system, graph, evidence package structure, prompt design, or answer formatting.
- Prefer claims that materially affect whether the answer is trustworthy.
- Split multi-part claims into separate claims when possible.
- If a sentence is only framing or rhetorical, skip it.
- Return strict JSON with exactly this schema:
{{
  "claims": [
    {{
      "text": "...",
      "verifiable": true
    }}
  ]
}}

Question:
{query}

Answer:
{answer}
"""

CLAIM_PAIR_EXTRACTION_PROMPT = """Goal:
- Extract every core, corpus-checkable claim needed to cover each answer's substantive content.
- Do not impose a maximum number of claims. Continue until each answer's core factual content is covered.
- Prefer high recall over brevity, but skip trivial restatements and purely rhetorical framing.

Rules:
- Keep claims atomic, specific, and corpus-checkable.
- The answers may use a P1-P5 scaffold. Ignore P1/P2/P3/P4/P5 labels, Titles, Answer Outline, Queries/Summaries/Evidence headings, Document Sections, Refinement notes, source labels, and other formatting shell text.
- Ignore claims about the retrieval system, graph, evidence package structure, prompt design, or answer formatting.
- Prefer claims that materially affect whether the answer is trustworthy.
- Split multi-part claims into separate claims when possible.
- If a sentence is only framing or rhetorical, skip it.
- Return strict JSON with exactly this schema:
{{
  "answers": [
    {{
      "answer_id": "candidate",
      "claims": [
        {{
          "text": "...",
          "verifiable": true
        }}
      ]
    }},
    {{
      "answer_id": "baseline",
      "claims": [
        {{
          "text": "...",
          "verifiable": true
        }}
      ]
    }}
  ]
}}

Question:
{query}

Candidate answer:
{candidate_answer}

Baseline answer:
{baseline_answer}
"""

CLAIM_SUPPORT_PROMPT = """For each claim below, decide whether the retrieved corpus snippets support it.

Rules:
- Use only the provided snippets.
- "supported" means the snippets directly support the claim.
- "partially_supported" means the snippets support part of the claim but not the whole statement.
- "insufficient_evidence" means there is not enough information in the snippets to determine the claim's truthfulness.
- "contradicted" means the snippets explicitly contradict the claim.
- "verification_error" means there was an error in verifying the claim.
- Return strict JSON with exactly this schema:
{{
  "claim_assessments": [
    {{
      "answer_id": "candidate | baseline",
      "claim_index": 1,
      "claim": "...",
      "label": "supported | partially_supported | insufficient_evidence | contradicted | verification_error",
      "explanation": "..."
    }}
  ]
}}

Question:
{query}

Claims and retrieved snippets:
{payload}
"""

JSON_REPAIR_SYSTEM_PROMPT = """You repair malformed JSON.
Return only valid JSON. Do not add prose, Markdown, or code fences."""

JSON_REPAIR_PROMPT = """The previous model response was intended to be JSON but could not be parsed.

Parse error:
{error}

Invalid response:
{raw}

Return only the repaired JSON object that satisfies the requested schema.
"""


def _extract_json(text):
    """Recover JSON even if the model wraps it in prose or code fences.

    从模型回复中抽取 JSON 内容，忽略前后附加文本。
    """
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
        if fenced:
            try:
                return json.loads(fenced.group(1).strip())
            except Exception:
                pass
        decoder = json.JSONDecoder()
        for start, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[start:])
                return obj
            except Exception:
                continue
        raise


def _claim_support_report(groundedness):
    """Compact claim-label report for QFS judging, without evidence snippets."""
    if not groundedness:
        return json.dumps(
            {
                "claim_count": 0,
                "supported_claim_ratio": 0.0,
                "label_counts": {},
                "claims": [],
            },
            ensure_ascii=False,
        )
    claims = []
    for item in groundedness.get("claims", []):
        claims.append(
            {
                "claim_index": item.get("claim_index"),
                "label": item.get("label"),
                "claim": item.get("claim"),
            }
        )
    report = {
        "claim_count": groundedness.get("claim_count", len(claims)),
        "supported_claim_ratio": groundedness.get("supported_claim_ratio", 0.0),
        "label_counts": groundedness.get("label_counts", {}),
        "claims": claims,
    }
    return json.dumps(report, ensure_ascii=False, indent=2)


def build_judge_prompt(query, answer_a, answer_b):
    """Build the exact FG-RAG evaluation prompt template.

    参数:
        query: 评判的问题文本。
        answer_a: 第一个候选答案文本。
        answer_b: 第二个候选答案文本。
    返回:
        用于 LLM 判分的完整提示字符串。

    构造原始 QFS 质量判分提示，不注入 groundedness 标签。
    """
    return FG_RAG_EVALUATE_PROMPT.format(
        input_query=query,
        first_answer=answer_a,
        second_answer=answer_b,
    )


def build_source_compliance_prompt(query, answer_a, answer_b):
    """Build the separate source-bounded contract compliance judge prompt."""
    return SOURCE_COMPLIANCE_PROMPT.format(
        input_query=query,
        first_answer=answer_a,
        second_answer=answer_b,
    )


def _generate_json(prompt, llm_client, system_prompt, max_tokens=900, max_attempts=3):
    """Run JSON-only generation with parse retries and JSON repair.

    参数:
        prompt: 传给模型的用户提示文本。
        llm_client: 支持 generate() 的 LLM 客户端实例。
        system_prompt: 系统角色提示内容。
        max_tokens: 生成最大 token 数。
        max_attempts: 解析失败时的最大重试次数。

    返回:
        解析后的 JSON 对象。

    重复调用 LLM，直到成功解析出 JSON 或达到重试上限。
    """
    last_error = None
    current_prompt = prompt
    for attempt in range(1, max_attempts + 1):
        raw = ""
        try:
            raw = llm_client.generate(current_prompt, system_prompt=system_prompt, temperature=0.0, max_tokens=max_tokens)
            return _extract_json(raw)
        except Exception as exc:
            last_error = exc
            log(f"[judge] json parse failure attempt={attempt}/{max_attempts}: {exc}")
            if raw and attempt < max_attempts:
                current_prompt = JSON_REPAIR_PROMPT.format(error=str(exc), raw=raw)
                system_prompt = JSON_REPAIR_SYSTEM_PROMPT
    raise last_error


def build_claim_extraction_prompt(query, answer):
    return CLAIM_EXTRACTION_PROMPT.format(query=query, answer=answer)


def build_pair_claim_extraction_prompt(query, candidate_answer, baseline_answer):
    return CLAIM_PAIR_EXTRACTION_PROMPT.format(
        query=query,
        candidate_answer=candidate_answer,
        baseline_answer=baseline_answer,
    )


def build_claim_support_prompt(query, claim_packets):
    return CLAIM_SUPPORT_PROMPT.format(
        query=query,
        payload=json.dumps(claim_packets, ensure_ascii=False, indent=2),
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

WINRATE_TABLE_METRICS = [
    "Overall Winner",
    "Comprehensiveness",
    "Diversity",
    "Empowerment",
]

SOURCE_COMPLIANCE_KEYS = [
    "Evidence Usefulness",
    "Source Grounding",
    "Source Layering",
    "Gap Awareness",
    "Source Compliance Winner",
]

SOURCE_COMPLIANCE_TABLE_METRICS = [
    "Source Compliance Winner",
    "Evidence Usefulness",
    "Source Grounding",
    "Source Layering",
    "Gap Awareness",
]

CLAIM_DIAGNOSTIC_TABLE_METRICS = [
    "Claim Corpus Support",
]

DEFAULT_CLAIM_RETRIEVAL_TOP_K = 5


def _map_letter_winner(winner):
    if winner == "a":
        return "candidate"
    if winner == "b":
        return "baseline"
    return "tie"


def _extract_dimension_votes(verdict_ab, verdict_ba_raw):
    """Map both evaluation orders back into candidate/baseline vote counts."""
    return _extract_dimension_votes_for_keys(verdict_ab, verdict_ba_raw, CRITERIA_KEYS)


def _extract_dimension_votes_for_keys(verdict_ab, verdict_ba_raw, criteria_keys):
    """Map both evaluation orders back into candidate/baseline vote counts for arbitrary criteria."""
    mapped = {}
    for key in criteria_keys:
        winner_ab = _map_letter_winner(_normalize_winner(verdict_ab.get(key, {}).get("Winner", "")))
        winner_ba = _map_letter_winner(_map_swapped_winner(_normalize_winner(verdict_ba_raw.get(key, {}).get("Winner", ""))))
        votes = Counter([winner_ab, winner_ba])
        mapped[key] = {
            "candidate": votes["candidate"],
            "baseline": votes["baseline"],
            "tie": votes["tie"],
        }
    return mapped


def _parse_error_verdict(criteria_keys=None):
    """Tie verdict used only when pairwise evaluator JSON cannot be parsed."""
    criteria_keys = criteria_keys or CRITERIA_KEYS
    return {
        key: {
            "Winner": "Tie",
            "Explanation": "Evaluator output could not be parsed reliably; treated as a tie for robustness.",
        }
        for key in criteria_keys
    }


def load_judge_corpus_resources(corpus_dir):
    """Load dense chunk retrieval resources for claim-groundedness checks."""
    if not corpus_dir:
        return None
    _graph, chunk_store, index_dir = load_graph_corpus(Path(corpus_dir))
    dense_path = index_dir / "vdb_chunks.json"
    dense_index = DenseChunkIndex.load(dense_path) if dense_path.exists() else None
    bm25_index = BM25Index.build(chunk_store)
    embedding_client = OpenAICompatibleEmbeddingClient(load_llm_config())
    claim_retriever = HybridChunkRetriever(
        bm25_index=bm25_index,
        dense_index=dense_index,
        embedding_client=embedding_client,
        mode="dense",
    )
    return {
        "index_dir": str(index_dir),
        "chunk_store": chunk_store,
        "dense_index": dense_index,
        "embedding_client": embedding_client,
        "claim_retriever": claim_retriever,
    }


def _normalize_claim_label(raw_value):
    value = (raw_value or "").strip().lower()
    if value in {
        "supported",
        "partially_supported",
        "insufficient_evidence",
        "contradicted",
        "verification_error",
    }:
        return value
    if ("support" in value and value.startswith("un")) or "not supported" in value:
        return "insufficient_evidence"
    if "partial" in value:
        return "partially_supported"
    if "insufficient" in value or "not enough" in value:
        return "insufficient_evidence"
    if "contradict" in value or "refute" in value:
        return "contradicted"
    if "support" in value:
        return "supported"
    if "error" in value:
        return "verification_error"
    return "verification_error"


_SCAFFOLD_CLAIM_PATTERNS = (
    r"^p[1-7]\s*[\.:]",
    r"^titles?\s*[:\-]?$",
    r"^answer outline\s*[:\-]?$",
    r"^queries,\s*summaries,\s*and evidence\s*[:\-]?$",
    r"^document sections?\s*[:\-]?$",
    r"^refinement\s*[:\-]?$",
    r"^source\s+chunks?\s*[:\-]?$",
)


def _normalize_extracted_claim_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^\s*[-*]\s*", "", text)
    text = re.sub(r"^⟨[^⟩]+⟩\s*", "", text)
    text = re.sub(r"^\*\*Aspect\s*\d+[^*]*\*\*\s*:?\s*", "", text, flags=re.I)
    text = re.sub(r"^\*\*[^*]+\*\*\s*:?\s*", "", text)
    text = re.sub(r"\bkg-\d+\b", "", text, flags=re.I)
    text = re.sub(r"\bsrc-\d+\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" -|:;,.")
    return text


def _is_scaffold_claim(text):
    value = str(text or "").strip().lower()
    if not value:
        return True
    return any(re.search(pattern, value, flags=re.I) for pattern in _SCAFFOLD_CLAIM_PATTERNS)


def _claim_dedup_key(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _coerce_claim_items(payload):
    """Accept minor JSON shape drift without falling back to sentence splitting."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    raw_claims = payload.get("claims")
    if isinstance(raw_claims, list):
        return raw_claims
    if isinstance(raw_claims, dict):
        nested = raw_claims.get("items") or raw_claims.get("claims")
        if isinstance(nested, list):
            return nested
        return list(raw_claims.values())
    if payload.get("text") or payload.get("claim"):
        return [payload]
    for value in payload.values():
        if isinstance(value, list) and any(isinstance(item, (str, dict)) for item in value):
            return value
    return []


def _clean_claim_items(raw_claims):
    claims = []
    seen = set()

    for item in raw_claims:
        if isinstance(item, str):
            claim_text = item
            verifiable = True
        else:
            claim_text = item.get("text") or item.get("claim") or item.get("statement") or ""
            verifiable = bool(item.get("verifiable", True))
        claim_text = _normalize_extracted_claim_text(claim_text)

        if not claim_text or not verifiable or _is_scaffold_claim(claim_text):
            continue

        dedup_key = _claim_dedup_key(claim_text)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        claims.append(
            {
                "text": claim_text,
                "verifiable": True,
            }
        )
    return claims


def _extract_claims(query, answer, llm_client):
    prompt = build_claim_extraction_prompt(query, answer)
    try:
        payload = _generate_json(
            prompt,
            llm_client,
            CLAIM_SYSTEM_PROMPT,
            max_tokens=12000,
            max_attempts=4,
        )
    except Exception as exc:
        log(f"[judge] claim extraction failed: {exc}")
        return []

    return _clean_claim_items(_coerce_claim_items(payload))


def _coerce_pair_claim_items(payload):
    if not isinstance(payload, dict):
        return {"candidate": [], "baseline": []}
    grouped = {"candidate": [], "baseline": []}
    answers = payload.get("answers")
    if isinstance(answers, list):
        for answer_payload in answers:
            if not isinstance(answer_payload, dict):
                continue
            answer_id = str(answer_payload.get("answer_id") or answer_payload.get("id") or "").strip().lower()
            if answer_id not in grouped:
                continue
            grouped[answer_id].extend(_coerce_claim_items(answer_payload))
    for answer_id in grouped:
        raw_value = payload.get(answer_id)
        if raw_value is not None:
            grouped[answer_id].extend(_coerce_claim_items(raw_value if isinstance(raw_value, dict) else {"claims": raw_value}))
    return grouped


def _extract_pair_claims(query, candidate_answer, baseline_answer, llm_client):
    prompt = build_pair_claim_extraction_prompt(query, candidate_answer, baseline_answer)
    try:
        payload = _generate_json(
            prompt,
            llm_client,
            CLAIM_SYSTEM_PROMPT,
            max_tokens=20000,
            max_attempts=4,
        )
    except Exception as exc:
        log(f"[judge] pair claim extraction failed: {exc}")
        return {"candidate": [], "baseline": []}
    grouped = _coerce_pair_claim_items(payload)
    return {
        "candidate": _clean_claim_items(grouped["candidate"]),
        "baseline": _clean_claim_items(grouped["baseline"]),
    }


def _claim_retrieval_query(query, claim_text):
    return f"Question: {query}\nClaim to verify: {claim_text}"


def _dense_claim_hits(query, claim_texts, corpus_resources, top_k):
    dense_index = corpus_resources.get("dense_index")
    embedding_client = corpus_resources.get("embedding_client")
    if dense_index is None or embedding_client is None:
        raise ValueError("dense claim retriever is unavailable")
    queries = [_claim_retrieval_query(query, claim_text) for claim_text in claim_texts]
    if hasattr(embedding_client, "embed_texts"):
        vectors = embedding_client.embed_texts(queries)
    else:
        vectors = [embedding_client.embed_text(query_text) for query_text in queries]
    return [dense_index.search(vector, top_k=top_k) for vector in vectors]


def _claim_evidence_packets(
    query,
    claims,
    corpus_resources,
    top_k=DEFAULT_CLAIM_RETRIEVAL_TOP_K,
    snippet_words=220,
    answer_id=None,
):
    if not claims or not corpus_resources:
        return []
    chunk_store = corpus_resources["chunk_store"]
    claim_texts = [claim["text"] if isinstance(claim, dict) else str(claim) for claim in claims]
    retrieval_error = None
    try:
        hit_batches = _dense_claim_hits(query, claim_texts, corpus_resources, top_k=top_k)
    except Exception as exc:
        hit_batches = [[] for _ in claim_texts]
        retrieval_error = f"{exc.__class__.__name__}: {exc}"
        log(f"[judge] dense batch claim retrieval failed answer_id={answer_id or 'single'}: {exc}")
    packets = []
    for claim_index, (claim_text, hits) in enumerate(zip(claim_texts, hit_batches), start=1):
        snippets = []
        for rank, hit in enumerate(hits, start=1):
            chunk = chunk_store.get(hit["chunk_id"], {})
            content = " ".join(chunk.get("content", "").split())
            snippet = " ".join(content.split()[:snippet_words])
            snippets.append(
                {
                    "rank": rank,
                    "chunk_id": hit["chunk_id"],
                    "score_norm": round(float(hit.get("score_norm", 0.0)), 6),
                    "dense_score": round(float(hit.get("dense_score", 0.0)), 6),
                    "dense_score_norm": round(float(hit.get("dense_score_norm", 0.0)), 6),
                    "snippet": snippet,
                }
            )
        packet = {
            "claim_index": claim_index,
            "claim": claim_text,
            "retrieved_chunks": snippets,
            "retrieval_error": retrieval_error,
        }
        if answer_id:
            packet["answer_id"] = answer_id
        packets.append(packet)
    return packets


def _pair_claim_evidence_packets(
    query,
    pair_claims,
    corpus_resources,
    top_k=DEFAULT_CLAIM_RETRIEVAL_TOP_K,
    snippet_words=220,
):
    all_claims = []
    packet_keys = []
    for answer_id in ("candidate", "baseline"):
        for claim_index, claim in enumerate(pair_claims.get(answer_id, []), start=1):
            all_claims.append(claim)
            packet_keys.append((answer_id, claim_index))
    if not all_claims:
        return {"candidate": [], "baseline": []}
    flat_packets = _claim_evidence_packets(
        query=query,
        claims=all_claims,
        corpus_resources=corpus_resources,
        top_k=top_k,
        snippet_words=snippet_words,
    )
    grouped = {"candidate": [], "baseline": []}
    for packet, (answer_id, claim_index) in zip(flat_packets, packet_keys):
        packet["answer_id"] = answer_id
        packet["claim_index"] = claim_index
        grouped[answer_id].append(packet)
    return grouped


def _chunked(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _verification_error_assessment(packet, explanation):
    assessment = {
        "claim_index": packet.get("claim_index"),
        "claim": packet["claim"],
        "label": "verification_error",
        "explanation": explanation,
        "retrieved_chunks": packet.get("retrieved_chunks", []),
    }
    if packet.get("answer_id"):
        assessment["answer_id"] = packet["answer_id"]
    return assessment


def _parse_support_assessments(payload, claim_packets):
    raw_items = payload.get("claim_assessments", []) if isinstance(payload, dict) else []
    by_index = {
        (packet.get("answer_id", ""), int(packet["claim_index"])): packet
        for packet in claim_packets
    }
    by_claim = {_claim_dedup_key(packet["claim"]): packet for packet in claim_packets}
    seen = set()
    assessments = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        raw_index = item.get("claim_index")
        answer_id = str(item.get("answer_id") or "").strip().lower()
        packet = None
        try:
            if raw_index is not None:
                packet = by_index.get((answer_id, int(raw_index))) or by_index.get(("", int(raw_index)))
        except Exception:
            packet = None
        if packet is None:
            packet = by_claim.get(_claim_dedup_key(item.get("claim", "")))
        if packet is None:
            continue
        packet_key = (packet.get("answer_id", ""), int(packet["claim_index"]))
        if packet_key in seen:
            continue
        seen.add(packet_key)
        assessment = {
            "claim_index": int(packet["claim_index"]),
            "claim": packet["claim"],
            "label": _normalize_claim_label(item.get("label", "")),
            "explanation": str(item.get("explanation", "")).strip(),
            "retrieved_chunks": packet.get("retrieved_chunks", []),
        }
        if packet.get("answer_id"):
            assessment["answer_id"] = packet["answer_id"]
        assessments.append(assessment)
    for packet in claim_packets:
        packet_key = (packet.get("answer_id", ""), int(packet["claim_index"]))
        if packet_key not in seen:
            assessments.append(_verification_error_assessment(packet, "support_check_missing_assessment"))
    assessments.sort(key=lambda item: (str(item.get("answer_id") or ""), int(item.get("claim_index") or 0)))
    return assessments


def _assess_claim_support(query, claim_packets, llm_client, batch_size=16):
    if not claim_packets:
        return []
    assessments = []
    for packet_batch in _chunked(claim_packets, batch_size):
        retrieval_errors = [packet for packet in packet_batch if packet.get("retrieval_error")]
        assessable = [packet for packet in packet_batch if not packet.get("retrieval_error")]
        assessments.extend(
            _verification_error_assessment(packet, f"claim_retrieval_error: {packet['retrieval_error']}")
            for packet in retrieval_errors
        )
        if not assessable:
            continue
        prompt = build_claim_support_prompt(query, assessable)
        max_tokens = min(20000, 700 + 450 * len(assessable))
        try:
            payload = _generate_json(
                prompt,
                llm_client,
                CLAIM_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                max_attempts=3,
            )
            assessments.extend(_parse_support_assessments(payload, assessable))
        except Exception as exc:
            log(f"[judge] claim support check failed: {exc}")
            assessments.extend(
                _verification_error_assessment(packet, f"support_check_error:{exc.__class__.__name__}")
                for packet in assessable
            )
    assessments.sort(key=lambda item: (str(item.get("answer_id") or ""), int(item.get("claim_index") or 0)))
    return assessments


def _empty_answer_groundedness(claim_count=0, extraction_failed_or_empty=False):
    return {
        "claims": [],
        "claim_count": claim_count,
        "supported_claim_count": 0,
        "supported_claim_ratio": 0.0,
        "label_counts": {
            "supported": 0,
            "partially_supported": 0,
            "insufficient_evidence": 0,
            "contradicted": 0,
            "verification_error": 0,
        },
        "extraction_failed_or_empty": extraction_failed_or_empty,
    }


def _summarize_groundedness_assessments(assessments, extracted_claim_count=0):
    if not assessments:
        if extracted_claim_count:
            return _empty_answer_groundedness(claim_count=extracted_claim_count, extraction_failed_or_empty=False)
        return _empty_answer_groundedness(claim_count=0, extraction_failed_or_empty=True)

    counts = Counter()
    for item in assessments:
        label = item["label"]
        counts[label] += 1

    claim_count = len(assessments)
    supported_claim_ratio = counts["supported"] / max(claim_count, 1)
    label_counts = {
        "supported": counts["supported"],
        "partially_supported": counts["partially_supported"],
        "insufficient_evidence": counts["insufficient_evidence"],
        "contradicted": counts["contradicted"],
        "verification_error": counts["verification_error"],
    }

    return {
        "claims": assessments,
        "claim_count": claim_count,
        "supported_claim_count": counts["supported"],
        "supported_claim_ratio": round(supported_claim_ratio, 4),
        "label_counts": label_counts,
        "supported": counts["supported"],
        "partially_supported": counts["partially_supported"],
        "insufficient_evidence": counts["insufficient_evidence"],
        "contradicted": counts["contradicted"],
        "verification_error": counts["verification_error"],
        "extraction_failed_or_empty": False,
    }


def assess_answer_groundedness(
    query,
    answer,
    llm_client,
    corpus_resources,
):
    claims = _extract_claims(query, answer, llm_client)

    if not claims:
        return _empty_answer_groundedness(claim_count=0, extraction_failed_or_empty=True)

    claim_packets = _claim_evidence_packets(
        query=query,
        claims=claims,
        corpus_resources=corpus_resources,
        top_k=DEFAULT_CLAIM_RETRIEVAL_TOP_K,
    )
    assessments = _assess_claim_support(query, claim_packets, llm_client)
    return _summarize_groundedness_assessments(assessments, extracted_claim_count=len(claims))


def assess_pair_groundedness(query, candidate_answer, baseline_answer, llm_client, corpus_resources):
    pair_claims = _extract_pair_claims(query, candidate_answer, baseline_answer, llm_client)
    if not pair_claims["candidate"] and not pair_claims["baseline"]:
        return {
            "candidate": _empty_answer_groundedness(claim_count=0, extraction_failed_or_empty=True),
            "baseline": _empty_answer_groundedness(claim_count=0, extraction_failed_or_empty=True),
        }

    grouped_packets = _pair_claim_evidence_packets(
        query=query,
        pair_claims=pair_claims,
        corpus_resources=corpus_resources,
        top_k=DEFAULT_CLAIM_RETRIEVAL_TOP_K,
    )
    combined_packets = grouped_packets["candidate"] + grouped_packets["baseline"]
    combined_assessments = _assess_claim_support(query, combined_packets, llm_client, batch_size=16)
    grouped_assessments = {"candidate": [], "baseline": []}
    for assessment in combined_assessments:
        answer_id = assessment.get("answer_id")
        if answer_id in grouped_assessments:
            clean_assessment = dict(assessment)
            clean_assessment.pop("answer_id", None)
            grouped_assessments[answer_id].append(clean_assessment)
    return {
        "candidate": _summarize_groundedness_assessments(
            grouped_assessments["candidate"],
            extracted_claim_count=len(pair_claims["candidate"]),
        ),
        "baseline": _summarize_groundedness_assessments(
            grouped_assessments["baseline"],
            extracted_claim_count=len(pair_claims["baseline"]),
        ),
    }


def _groundedness_decision(candidate_groundedness, baseline_groundedness):
    return {
        "winner": "tie",
        "rule": "diagnostic only: claim labels are reported as counts and are not converted into a heuristic winner",
    }


def _empty_groundedness_accumulator():
    return Counter(
        {
            "answer_count": 0,
            "claim_count": 0,
            "supported_claim_count": 0,
            "partially_supported_claim_count": 0,
            "insufficient_evidence_claim_count": 0,
            "contradicted_claim_count": 0,
            "verification_error_claim_count": 0,
        }
    )


def _accumulate_groundedness(accumulator, groundedness):
    accumulator["answer_count"] += 1
    accumulator["claim_count"] += int(groundedness.get("claim_count", 0) or 0)
    accumulator["supported_claim_count"] += int(groundedness.get("supported_claim_count", 0) or 0)
    accumulator["partially_supported_claim_count"] += int(groundedness.get("partially_supported", 0) or 0)
    accumulator["insufficient_evidence_claim_count"] += int(groundedness.get("insufficient_evidence", 0) or 0)
    accumulator["contradicted_claim_count"] += int(groundedness.get("contradicted", 0) or 0)
    accumulator["verification_error_claim_count"] += int(groundedness.get("verification_error", 0) or 0)


def _summarize_groundedness_accumulator(accumulator):
    answer_count = max(int(accumulator["answer_count"]), 1)
    claim_count = int(accumulator["claim_count"])
    return {
        "answer_count": int(accumulator["answer_count"]),
        "claim_count": claim_count,
        "supported_claim_count": int(accumulator["supported_claim_count"]),
        "partially_supported_claim_count": int(accumulator["partially_supported_claim_count"]),
        "insufficient_evidence_claim_count": int(accumulator["insufficient_evidence_claim_count"]),
        "contradicted_claim_count": int(accumulator["contradicted_claim_count"]),
        "verification_error_claim_count": int(accumulator["verification_error_claim_count"]),
        "supported_claim_ratio": round(float(accumulator["supported_claim_count"]) / max(claim_count, 1), 4),
        "average_claim_count": round(claim_count / answer_count, 4),
        "average_supported_claim_count": round(float(accumulator["supported_claim_count"]) / answer_count, 4),
    }


def _generate_verdict(prompt, llm_client, max_attempts=3, criteria_keys=None):
    """Retry parsing a judge response a few times before returning a parse-error tie."""
    criteria_keys = criteria_keys or CRITERIA_KEYS
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _generate_json(prompt, llm_client, JUDGE_SYSTEM_PROMPT, max_tokens=900, max_attempts=1)
        except Exception as exc:
            log(f"[judge] parse failure attempt={attempt}/{max_attempts}: {exc}")
            last_error = exc
    log(f"[judge] returning tie verdict after repeated parse failures: {last_error}")
    return _parse_error_verdict(criteria_keys=criteria_keys)


def _winner_from_votes(votes):
    if votes["a"] > votes["b"] and votes["a"] > votes["tie"]:
        return "candidate"
    if votes["b"] > votes["a"] and votes["b"] > votes["tie"]:
        return "baseline"
    return "tie"


def _judge_quality_pair(query, candidate_answer, baseline_answer, llm_client):
    prompt_ab = build_judge_prompt(query, candidate_answer, baseline_answer)
    verdict_ab = _generate_verdict(prompt_ab, llm_client, criteria_keys=CRITERIA_KEYS)
    prompt_ba = build_judge_prompt(query, baseline_answer, candidate_answer)
    verdict_ba_raw = _generate_verdict(prompt_ba, llm_client, criteria_keys=CRITERIA_KEYS)

    overall_ab = _normalize_winner(verdict_ab.get("Overall Winner", {}).get("Winner", "Tie"))
    overall_ba = _map_swapped_winner(_normalize_winner(verdict_ba_raw.get("Overall Winner", {}).get("Winner", "Tie")))
    votes = Counter([overall_ab, overall_ba])
    llm_overall_winner = _winner_from_votes(votes)
    dimension_votes = _extract_dimension_votes_for_keys(verdict_ab, verdict_ba_raw, CRITERIA_KEYS)

    return {
        "judge_mode": "quality",
        "order_ab": verdict_ab,
        "order_ba": verdict_ba_raw,
        "mapped_overall_votes": {
            "candidate": votes["a"],
            "baseline": votes["b"],
            "tie": votes["tie"],
        },
        "llm_overall_winner": llm_overall_winner,
        "dimension_votes": dimension_votes,
        "final_winner": llm_overall_winner,
    }


def _judge_source_compliance_pair(query, candidate_answer, baseline_answer, llm_client):
    prompt_ab = build_source_compliance_prompt(query, candidate_answer, baseline_answer)
    verdict_ab = _generate_verdict(prompt_ab, llm_client, criteria_keys=SOURCE_COMPLIANCE_KEYS)
    prompt_ba = build_source_compliance_prompt(query, baseline_answer, candidate_answer)
    verdict_ba_raw = _generate_verdict(prompt_ba, llm_client, criteria_keys=SOURCE_COMPLIANCE_KEYS)

    overall_ab = _normalize_winner(verdict_ab.get("Source Compliance Winner", {}).get("Winner", "Tie"))
    overall_ba = _map_swapped_winner(_normalize_winner(verdict_ba_raw.get("Source Compliance Winner", {}).get("Winner", "Tie")))
    votes = Counter([overall_ab, overall_ba])
    source_compliance_winner = _winner_from_votes(votes)
    dimension_votes = _extract_dimension_votes_for_keys(verdict_ab, verdict_ba_raw, SOURCE_COMPLIANCE_KEYS)

    return {
        "judge_mode": "source_compliance",
        "order_ab": verdict_ab,
        "order_ba": verdict_ba_raw,
        "mapped_overall_votes": {
            "candidate": votes["a"],
            "baseline": votes["b"],
            "tie": votes["tie"],
        },
        "llm_overall_winner": source_compliance_winner,
        "dimension_votes": dimension_votes,
        "final_winner": source_compliance_winner,
    }


def _judge_claim_diagnostics_pair(query, candidate_answer, baseline_answer, llm_client, corpus_resources=None):
    pair_groundedness = assess_pair_groundedness(query, candidate_answer, baseline_answer, llm_client, corpus_resources)
    candidate_groundedness = pair_groundedness["candidate"]
    baseline_groundedness = pair_groundedness["baseline"]
    groundedness_decision = _groundedness_decision(candidate_groundedness, baseline_groundedness)
    winner = groundedness_decision["winner"]
    return {
        "judge_mode": "claim_diagnostics",
        "order_ab": {},
        "order_ba": {},
        "mapped_overall_votes": {
            "candidate": 1 if winner == "candidate" else 0,
            "baseline": 1 if winner == "baseline" else 0,
            "tie": 1 if winner == "tie" else 0,
        },
        "llm_overall_winner": winner,
        "dimension_votes": {
            "Claim Corpus Support": {
                "candidate": 1 if winner == "candidate" else 0,
                "baseline": 1 if winner == "baseline" else 0,
                "tie": 1 if winner == "tie" else 0,
            }
        },
        "corpus_groundedness": {
            "winner": winner,
            "decision": groundedness_decision,
            "candidate": candidate_groundedness,
            "baseline": baseline_groundedness,
        },
        "final_winner": winner,
    }


def judge_pair(query, candidate_answer, baseline_answer, llm_client, corpus_resources=None, judge_mode="quality"):
    """Judge one answer pair in both orders to reduce position bias.

    参数:
        query: 原始问题文本。
        candidate_answer: 算法候选答案文本。
        baseline_answer: 对照基线答案文本。
        llm_client: 用于调用 LLM 的客户端实例。
        corpus_resources: 可选的语料资源，仅用于 claim_diagnostics。
        judge_mode: quality、source_compliance 或 claim_diagnostics。

    返回:
        一个包含 AB/BA 判定、组织分析、维度投票和 groundedness 结果的判决字典。

    对候选答案/基线答案进行 AB/BA 两次评判，降低位置偏差。
    """
    if judge_mode == "quality":
        return _judge_quality_pair(query, candidate_answer, baseline_answer, llm_client)
    if judge_mode == "source_compliance":
        return _judge_source_compliance_pair(query, candidate_answer, baseline_answer, llm_client)
    if judge_mode == "claim_diagnostics":
        return _judge_claim_diagnostics_pair(query, candidate_answer, baseline_answer, llm_client, corpus_resources)
    raise ValueError(f"Unsupported judge_mode: {judge_mode}")


def run_winrate_judgement(
    questions,
    candidate_answers,
    baseline_answers,
    llm_client,
    output_path=None,
    max_workers=None,
    corpus_resources=None,
    judge_mode="quality",
):
    """Run pairwise judging and aggregate both overall and per-dimension stats.

    参数:
        questions: 问题文本列表。
        candidate_answers: 候选答案记录列表，每条记录包含 model_answer。
        baseline_answers: 基线答案记录列表，每条记录包含 model_answer。
        llm_client: LLM 客户端实例。
        output_path: 可选的判决结果输出文件路径。
        max_workers: 并行评判的线程数，默认为 llm_client.max_concurrency。
        corpus_resources: 可选的语料资源，仅用于 claim_diagnostics。
        judge_mode: quality、source_compliance 或 claim_diagnostics。

    返回:
        包含总体胜率、各维度统计、合同条件汇总和每条判决详细信息的结果载荷。

    对每个问题执行成对评判，并汇总最终胜率、维度统计与组织分析。
    """
    if len(candidate_answers) != len(baseline_answers):
        raise ValueError("Candidate and baseline answer counts do not match")
    if judge_mode not in {"quality", "source_compliance", "claim_diagnostics"}:
        raise ValueError(f"Unsupported judge_mode: {judge_mode}")
    max_workers = max_workers or llm_client.max_concurrency
    log(f"[judge] start total={len(candidate_answers)} mode={judge_mode} model={llm_client.model} workers={max_workers}")
    verdicts = [None] * len(candidate_answers)
    summary_counter = Counter()
    llm_summary_counter = Counter()
    criteria_keys = {
        "quality": CRITERIA_KEYS,
        "source_compliance": SOURCE_COMPLIANCE_KEYS,
        "claim_diagnostics": ["Claim Corpus Support"],
    }[judge_mode]
    dimension_counter = {
        key: Counter({"candidate": 0, "baseline": 0, "tie": 0}) for key in criteria_keys
    }
    corpus_groundedness_counter = Counter({"candidate": 0, "baseline": 0, "tie": 0})
    groundedness_accumulators = {
        "candidate": _empty_groundedness_accumulator(),
        "baseline": _empty_groundedness_accumulator(),
    }
    pairs = list(zip(questions, candidate_answers, baseline_answers))

    def _one(item):
        idx, (query, cand, base) = item
        verdict = judge_pair(
            query,
            cand["model_answer"],
            base["model_answer"],
            llm_client,
            corpus_resources=corpus_resources,
            judge_mode=judge_mode,
        )
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
            if "corpus_groundedness" in verdict_payload:
                corpus_groundedness_counter[verdict_payload["corpus_groundedness"]["winner"]] += 1
                _accumulate_groundedness(
                    groundedness_accumulators["candidate"],
                    verdict_payload["corpus_groundedness"]["candidate"],
                )
                _accumulate_groundedness(
                    groundedness_accumulators["baseline"],
                    verdict_payload["corpus_groundedness"]["baseline"],
                )
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
            "judge_mode": judge_mode,
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
        "verdicts": verdicts,
    }
    if judge_mode == "claim_diagnostics":
        payload["corpus_groundedness_claim_summary"] = {
            "candidate": _summarize_groundedness_accumulator(groundedness_accumulators["candidate"]),
            "baseline": _summarize_groundedness_accumulator(groundedness_accumulators["baseline"]),
        }
    if output_path is not None:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[judge] wrote {output_path}")
    return payload


def build_winrate_table_rows(payload):
    """Normalize overall + per-criterion win rates into row objects for table rendering."""
    rows = []
    overall = payload.get("summary", {})
    judge_mode = overall.get("judge_mode", "quality")
    overall_metric = {
        "quality": "Overall Winner",
        "source_compliance": "Source Compliance Winner",
        "claim_diagnostics": "Claim Corpus Support",
    }.get(judge_mode, "Overall Winner")
    rows.append(
        {
            "metric": overall_metric,
            "candidate_win_rate": overall.get("candidate_win_rate", 0.0),
            "baseline_win_rate": overall.get("baseline_win_rate", 0.0),
            "tie_rate": round(overall.get("ties", 0) / max(overall.get("total", 1), 1), 4),
            "total_votes": overall.get("total", 0),
        }
    )
    criteria_summary = payload.get("criteria_summary", {})
    if judge_mode == "source_compliance":
        table_metrics = SOURCE_COMPLIANCE_TABLE_METRICS[1:]
    elif judge_mode == "claim_diagnostics":
        table_metrics = []
    else:
        table_metrics = WINRATE_TABLE_METRICS[1:]
    for metric_name in table_metrics:
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
