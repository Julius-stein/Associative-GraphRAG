"""Adaptive controller for association strength.

Current design principle:
- start from the non-adaptive baseline as the default behavior
- compute a linear alpha for analysis and light steering only
- let adaptive control nudge graph expansion budgets around the baseline
- do not let adaptive control fully take over rounds, group count, or source
  budget, because those large global changes were too brittle in practice
"""

import math

import networkx as nx

from .common import safe_mean, tokenize


OVERVIEW_CUES = {
    "overall",
    "main",
    "compare",
    "comparative",
    "comparison",
    "broader",
    "overview",
    "patterns",
    "pattern",
    "themes",
    "theme",
    "trends",
    "trend",
    "across",
    "different",
    "various",
    "role",
    "impact",
    "influence",
    "factors",
    "ways",
}

FOCUSED_CUES = {
    "which",
    "when",
    "reason",
    "where",
    "who",
    "specific",
    "exact",
    "section",
    "sections",
    "list",
    "steps",
    "signs",
    "cost",
    "costs",
    "examples",
}


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _phrase_hits(query_lower, cues):
    return sum(1 for cue in cues if cue in query_lower)


def _normalize_linear(value, low, high, inverse=False):
    """Map a raw feature into [0, 1] with an optional inverse direction."""
    if high <= low:
        return 0.0
    ratio = (value - low) / (high - low)
    ratio = _clamp(ratio)
    return 1.0 - ratio if inverse else ratio


def _normalized_entropy(values):
    """Entropy normalized to [0, 1]."""
    if not values:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    probs = [value / total for value in values if value > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in probs)
    max_entropy = math.log(len(probs))
    return entropy / max(max_entropy, 1e-12)


def _score_for_concentration(item):
    """Pick the most informative retrieval score available on a candidate hit."""
    score = item.get("dense_score")
    if score is None:
        score = item.get("retrieval_score")
    if score is None:
        score = item.get("score_norm", 0.0)
    return float(score)


def compute_candidate_retrieval_dispersion(candidate_chunk_hits, top_n=12, temperature=12.0):
    """Measure how many chunks are truly competitive at retrieval time.

    Unlike root-only dispersion, this looks earlier in the pipeline before the
    reranked top-k chunks flatten out. We convert the top candidate scores into
    a softmax distribution and compute an effective number of competitive chunks.
    """
    candidate_chunk_hits = candidate_chunk_hits[:top_n]
    if not candidate_chunk_hits:
        return {
            "candidate_chunk_dispersion": 0.0,
            "candidate_chunk_concentration": 1.0,
            "effective_candidate_chunk_count": 0.0,
        }

    raw_scores = [_score_for_concentration(item) for item in candidate_chunk_hits]
    max_score = max(raw_scores)
    scaled = [math.exp((score - max_score) * temperature) for score in raw_scores]
    total = sum(scaled)
    if total <= 0:
        scaled = [1.0 for _ in candidate_chunk_hits]
        total = float(len(candidate_chunk_hits))
    probs = [score / total for score in scaled]
    effective_count = 1.0 / max(sum(prob * prob for prob in probs), 1e-12)
    if len(probs) <= 1:
        dispersion = 0.0
    else:
        dispersion = (effective_count - 1.0) / max(len(probs) - 1.0, 1e-12)
    dispersion = _clamp(dispersion)
    return {
        "candidate_chunk_dispersion": round(dispersion, 6),
        "candidate_chunk_concentration": round(1.0 - dispersion, 6),
        "effective_candidate_chunk_count": round(effective_count, 6),
    }


def compute_retrieval_cliff(candidate_chunk_hits, top_n=15):
    """Measure pre-rerank retrieval concentration from the candidate pool.

    We intentionally compute this before root reranking because reranking tends
    to flatten the already-strong top chunks. The goal is to detect whether the
    *initial* retrieval landscape has a single dominant chunk neighborhood or a
    broad set of similarly relevant candidates.
    """
    if not candidate_chunk_hits:
        return {
            "retrieval_cliff": 0.0,
            "retrieval_flatness": 1.0,
            "retrieval_cliff_topn": top_n,
        }
    scores = []
    for item in candidate_chunk_hits[:top_n]:
        score = item.get("dense_score")
        if score is None:
            score = item.get("retrieval_score", item.get("score_norm", 0.0))
        scores.append(float(score))
    if len(scores) == 1:
        return {
            "retrieval_cliff": 1.0,
            "retrieval_flatness": 0.0,
            "retrieval_cliff_topn": len(scores),
        }
    top_score = scores[0]
    tail = scores[1:]
    tail_mean = safe_mean(tail) if tail else scores[1]
    scale = max(abs(top_score), 1e-9)
    cliff = _clamp((top_score - tail_mean) / scale)
    return {
        "retrieval_cliff": round(cliff, 6),
        "retrieval_flatness": round(1.0 - cliff, 6),
        "retrieval_cliff_topn": len(scores),
    }


def compute_query_intent_profile(query):
    """Estimate query scope from generic discourse cues only."""
    query_lower = query.lower()
    tokens = set(tokenize(query))
    overview_hits = _phrase_hits(query_lower, OVERVIEW_CUES) + int("how" in tokens) + int("why" in tokens)
    focused_hits = _phrase_hits(query_lower, FOCUSED_CUES)

    if focused_hits >= overview_hits + 1 and focused_hits > 0:
        style = "concrete"
    elif overview_hits >= focused_hits + 1 and overview_hits > 0:
        style = "synthesis"
    else:
        style = "balanced"
    return {
        "overview_hits": overview_hits,
        "focused_hits": focused_hits,
        "style": style,
    }


def compute_root_graph_features(root_nodes, root_edges, root_chunk_hits, node_to_chunks, edge_to_chunks):
    """Summarize the root graph structure for logging and later analysis."""
    root_graph = nx.Graph()
    root_graph.add_nodes_from(root_nodes)
    root_graph.add_edges_from(root_edges)
    if not root_nodes:
        return {
            "root_density": 0.0,
            "fragmentation": 1.0,
            "largest_component_ratio": 0.0,
            "component_count": 0,
        }

    components = list(nx.connected_components(root_graph))
    largest_component_ratio = max((len(component) for component in components), default=0) / max(len(root_nodes), 1)
    root_density = len(root_edges) / max(len(root_nodes), 1)

    return {
        "root_density": round(root_density, 6),
        "fragmentation": round(1.0 - largest_component_ratio, 6),
        "largest_component_ratio": round(largest_component_ratio, 6),
        "component_count": len(components),
    }


def build_adaptive_controller(
    query,
    root_nodes,
    root_edges,
    root_chunk_hits,
    candidate_chunk_hits,
    node_to_chunks,
    edge_to_chunks,
    base_cfg,
):
    """Convert query + root-graph signals into a per-query budget profile."""
    intent = compute_query_intent_profile(query)
    graph_features = compute_root_graph_features(
        root_nodes=root_nodes,
        root_edges=root_edges,
        root_chunk_hits=root_chunk_hits,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
    )
    candidate_features = compute_candidate_retrieval_dispersion(candidate_chunk_hits)
    retrieval_features = compute_retrieval_cliff(candidate_chunk_hits)

    # Main axis: broad questions tend to produce sparser root graphs.
    density_signal = _normalize_linear(
        graph_features["root_density"],
        low=0.30,
        high=0.75,
        inverse=True,
    )
    # Auxiliary correction: fragmented roots justify a bit more association.
    fragmentation_signal = _normalize_linear(
        graph_features["fragmentation"],
        low=0.50,
        high=0.95,
        inverse=False,
    )
    # Query-form is only a weak prior now.
    if intent["style"] == "synthesis":
        style_prior = 0.08
    elif intent["style"] == "concrete":
        style_prior = -0.08
    else:
        style_prior = 0.0

    # Retrieval cliff stays as a brake rather than a driver.
    retrieval_cliff = retrieval_features["retrieval_cliff"]
    cliff_brake = 0.18 * retrieval_cliff

    # Relative linear alpha around the non-adaptive midpoint 0.5.
    # This keeps the feature semantics interpretable while preventing alpha
    # from dominating the whole pipeline.
    association_strength = (
        0.50
        + 0.26 * (density_signal - 0.50)
        + 0.12 * (fragmentation_signal - 0.50)
        + style_prior
        - cliff_brake
    )
    association_strength = _clamp(association_strength, low=0.15, high=0.85)

    # Bridge search can tolerate a bit more spread than semantic expansion,
    # especially when the root graph is fragmented, but keep the adjustment
    # close to the baseline.
    bridge_strength = _clamp(
        association_strength + 0.06 * (fragmentation_signal - 0.50),
        low=0.15,
        high=0.90,
    )
    semantic_strength = association_strength

    # Convert alpha into residual budget changes around the no-adaptive base.
    bridge_delta = bridge_strength - 0.50
    semantic_delta = semantic_strength - 0.50
    path_scale = 1.0 + 0.40 * bridge_delta
    semantic_scale = 1.0 + 0.50 * semantic_delta
    score_tightening = 1.0 - 0.45 * semantic_delta

    adapted = {
        # Keep global structure fixed. Adaptive control should not decide how
        # many alternation rounds the whole pipeline gets.
        "association_rounds": base_cfg["association_rounds"],
        "path_budget": max(4, round(base_cfg["path_budget"] * path_scale)),
        "semantic_edge_budget": max(6, round(base_cfg["semantic_edge_budget"] * semantic_scale)),
        "semantic_node_budget": max(4, round(base_cfg["semantic_node_budget"] * semantic_scale)),
        "semantic_edge_min_score": round(base_cfg["semantic_edge_min_score"] * score_tightening, 6),
        "semantic_node_min_score": round(base_cfg["semantic_node_min_score"] * score_tightening, 6),
        # Keep the final evidence-presentation shape stable for fairer
        # comparisons; adaptive control only adjusts retrieval-time expansion.
        "group_limit": base_cfg["group_limit"],
        "max_source_chunks": base_cfg["max_source_chunks"],
        "max_source_word_budget": base_cfg["max_source_word_budget"],
    }

    return {
        "query_style": intent["style"],
        "association_strength": round(association_strength, 6),
        "alpha_components": {
            "density_signal": round(density_signal, 6),
            "fragmentation_signal": round(fragmentation_signal, 6),
            "style_prior": round(style_prior, 6),
            "cliff_brake": round(cliff_brake, 6),
            "bridge_strength": round(bridge_strength, 6),
        },
        "intent": intent,
        "graph_features": graph_features,
        "candidate_features": candidate_features,
        "retrieval_features": retrieval_features,
        "adapted_budgets": adapted,
    }
