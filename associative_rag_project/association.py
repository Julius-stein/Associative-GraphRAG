"""Graph association logic.

The association stage is explicitly treated as a divergence stage.

Current design uses two structural relations:
- graph adjacency over entity-relation edges
- provenance adjacency over chunk <-> node/edge links, plus neighboring chunks

On top of these two structures, the code performs two kinds of association:
- structural association: reward candidates that create new bridges
- semantic/coverage association: reward candidates that introduce new evidence

This yields a practical 2 x 2 scheme:
- graph-bridge association
- chunk-bridge association
- graph-coverage association
- chunk-coverage association
"""

from collections import Counter

import networkx as nx

from .common import edge_key
from .retrieval import normalize_relation_category, score_root_edges, score_root_nodes, support_score


def build_root_components(root_nodes, root_edges):
    """Materialize connected components for the current root graph."""
    root_graph = nx.Graph()
    root_graph.add_nodes_from(root_nodes)
    root_graph.add_edges_from(root_edges)
    components = list(nx.connected_components(root_graph))
    node_to_component = {}
    for component_index, nodes in enumerate(components):
        for node_id in nodes:
            node_to_component[node_id] = component_index
    return root_graph, components, node_to_component


def _covered_chunk_ids(current_nodes, current_edges, node_to_chunks, edge_to_chunks):
    covered = set()
    for node_id in current_nodes:
        covered.update(node_to_chunks.get(node_id, set()))
    for edge_id in current_edges:
        covered.update(edge_to_chunks.get(edge_id, set()))
    return covered


def _path_chunk_ids(path, node_to_chunks, edge_to_chunks):
    chunk_ids = set()
    for node_id in path:
        chunk_ids.update(node_to_chunks.get(node_id, set()))
    for index in range(len(path) - 1):
        chunk_ids.update(edge_to_chunks.get(edge_key(path[index], path[index + 1]), set()))
    return chunk_ids


def _bounded_gain(value, cap=5):
    return min(value, cap)


def _path_bridge_signature(path, node_to_component):
    components = {
        node_to_component.get(path[0]),
        node_to_component.get(path[-1]),
    }
    return len({item for item in components if item is not None})


def _graph_bridge_association(
    graph,
    current_nodes,
    current_edges,
    top_nodes,
    top_edges,
    node_to_chunks,
    edge_to_chunks,
    max_hop,
    path_budget,
):
    """Expand by graph paths that connect disconnected evidence regions.

    Primary objective:
    - bridge separate current components

    Tie-break objectives:
    - introduce more new supporting chunks
    - prefer shorter paths
    """
    if not current_nodes:
        return {
            "current_components": [],
            "selected_paths": [],
            "selected_path_nodes": [],
            "selected_path_edges": [],
        }

    _, components, node_to_component = build_root_components(current_nodes, current_edges)
    current_edge_set = set(current_edges)
    covered_chunks = _covered_chunk_ids(current_nodes, current_edges, node_to_chunks, edge_to_chunks)

    seed_nodes = []
    seen = set()
    for item in top_nodes:
        if item["id"] not in seen:
            seed_nodes.append(item["id"])
            seen.add(item["id"])
    for item in top_edges:
        for node_id in item["edge"]:
            if node_id not in seen:
                seed_nodes.append(node_id)
                seen.add(node_id)

    candidate_paths = []
    dedupe_paths = set()
    for source_id in seed_nodes:
        source_component = node_to_component.get(source_id)
        paths = nx.single_source_shortest_path(graph, source_id, cutoff=max_hop)
        for target_id in current_nodes:
            if target_id == source_id:
                continue
            target_component = node_to_component.get(target_id)
            if source_component is None or target_component is None or source_component == target_component:
                continue
            path = paths.get(target_id)
            if path is None or len(path) <= 1:
                continue
            canonical = tuple(path) if tuple(path) < tuple(reversed(path)) else tuple(reversed(path))
            if canonical in dedupe_paths:
                continue
            path_edges = {edge_key(path[index], path[index + 1]) for index in range(len(path) - 1)}
            if path_edges.issubset(current_edge_set):
                continue
            dedupe_paths.add(canonical)
            path_chunks = _path_chunk_ids(path, node_to_chunks, edge_to_chunks)
            new_source_count = _bounded_gain(len(path_chunks - covered_chunks))
            candidate_paths.append(
                {
                    "path": path,
                    "bridge_gain": _path_bridge_signature(path, node_to_component),
                    "new_source_count": new_source_count,
                    "path_length": len(path) - 1,
                }
            )

    candidate_paths.sort(
        key=lambda item: (
            -item["bridge_gain"],
            -item["new_source_count"],
            item["path_length"],
            item["path"],
        )
    )
    selected_paths = candidate_paths[:path_budget]
    selected_nodes = set()
    selected_edges = set()
    for item in selected_paths:
        selected_nodes.update(item["path"])
        selected_edges.update(edge_key(item["path"][index], item["path"][index + 1]) for index in range(len(item["path"]) - 1))
    return {
        "current_components": [sorted(component) for component in components],
        "selected_paths": selected_paths,
        "selected_path_nodes": sorted(selected_nodes),
        "selected_path_edges": sorted(selected_edges),
    }


def _chunk_component_ids(chunk_id, current_nodes, chunk_to_nodes, chunk_to_edges, node_to_component):
    component_ids = set()
    for node_id in chunk_to_nodes.get(chunk_id, set()):
        if node_id in current_nodes and node_id in node_to_component:
            component_ids.add(node_to_component[node_id])
    for edge_id in chunk_to_edges.get(chunk_id, set()):
        for node_id in edge_id:
            if node_id in current_nodes and node_id in node_to_component:
                component_ids.add(node_to_component[node_id])
    return component_ids


def _chunk_bridge_touch(chunk_id, current_nodes, current_edges, chunk_to_nodes, chunk_to_edges, chunk_neighbors, covered_band):
    touched = 0
    touched += len(set(chunk_to_nodes.get(chunk_id, set())) & set(current_nodes))
    touched += len(set(chunk_to_edges.get(chunk_id, set())) & set(current_edges))
    touched += len(set(chunk_neighbors.get(chunk_id, set())) & set(covered_band))
    return touched


def _chunk_bridge_association(
    current_nodes,
    current_edges,
    root_chunk_ids,
    root_chunk_score_lookup,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    chunk_budget,
):
    """Expand by chunk-side structural adjacency.

    A candidate chunk is considered a structural bridge when it is reachable via
    chunk neighborhoods/provenance and helps connect multiple currently active
    components or strengthens the current frontier with new chunk-supported
    graph content.
    """
    if chunk_budget <= 0:
        return {
            "selected_chunks": [],
            "selected_chunk_nodes": [],
            "selected_chunk_edges": [],
        }

    _, _, node_to_component = build_root_components(current_nodes, current_edges)
    covered_chunks = _covered_chunk_ids(current_nodes, current_edges, node_to_chunks, edge_to_chunks)
    covered_band = _expand_chunk_band(covered_chunks, chunk_neighbors)
    candidate_chunks = set(covered_band)
    for chunk_id in list(covered_band):
        candidate_chunks.update(chunk_neighbors.get(chunk_id, set()))
    candidate_chunks -= covered_chunks

    scored_chunks = []
    current_node_set = set(current_nodes)
    current_edge_set = set(current_edges)
    root_chunk_id_set = set(root_chunk_ids)
    for chunk_id in candidate_chunks:
        chunk_nodes = set(chunk_to_nodes.get(chunk_id, set()))
        chunk_edges = set(chunk_to_edges.get(chunk_id, set()))
        new_nodes = chunk_nodes - current_node_set
        new_edges = chunk_edges - current_edge_set
        if not new_nodes and not new_edges:
            continue
        component_ids = _chunk_component_ids(
            chunk_id=chunk_id,
            current_nodes=current_nodes,
            chunk_to_nodes=chunk_to_nodes,
            chunk_to_edges=chunk_to_edges,
            node_to_component=node_to_component,
        )
        bridge_gain = 1 if len(component_ids) >= 2 else 0
        frontier_touch = _bounded_gain(
            _chunk_bridge_touch(
                chunk_id=chunk_id,
                current_nodes=current_nodes,
                current_edges=current_edges,
                chunk_to_nodes=chunk_to_nodes,
                chunk_to_edges=chunk_to_edges,
                chunk_neighbors=chunk_neighbors,
                covered_band=covered_band,
            )
        )
        new_source_count = _bounded_gain(len(_expand_chunk_band({chunk_id}, chunk_neighbors) - covered_band))
        root_overlap = 1 if chunk_id in root_chunk_id_set else 0
        root_band_alignment = max(
            [root_chunk_score_lookup.get(neighbor_id, 0.0) for neighbor_id in _expand_chunk_band({chunk_id}, chunk_neighbors)],
            default=0.0,
        )
        scored_chunks.append(
            {
                "chunk_id": chunk_id,
                "bridge_gain": bridge_gain,
                "frontier_touch": frontier_touch,
                "new_source_count": new_source_count,
                "new_node_count": len(new_nodes),
                "new_edge_count": len(new_edges),
                "root_overlap": root_overlap,
                "root_band_alignment": round(root_band_alignment, 6),
                "node_ids": sorted(chunk_nodes),
                "edge_ids": sorted(chunk_edges),
            }
        )

    scored_chunks.sort(
        key=lambda item: (
            -item["bridge_gain"],
            -item["frontier_touch"],
            -item["new_source_count"],
            -(item["new_node_count"] + item["new_edge_count"]),
            -item["root_overlap"],
            -item["root_band_alignment"],
            item["chunk_id"],
        )
    )
    selected_chunks = scored_chunks[:chunk_budget]
    selected_nodes = set()
    selected_edges = set()
    for item in selected_chunks:
        selected_nodes.update(item["node_ids"])
        selected_edges.update(item["edge_ids"])
    return {
        "selected_chunks": selected_chunks,
        "selected_chunk_nodes": sorted(selected_nodes),
        "selected_chunk_edges": sorted(selected_edges),
    }


def bridge_association(
    graph,
    current_nodes,
    current_edges,
    root_chunk_ids,
    root_chunk_score_lookup,
    top_nodes,
    top_edges,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    max_hop,
    path_budget,
):
    """Expand structurally using both graph paths and chunk-side bridges."""
    graph_bridge_budget = max(1, path_budget // 2)
    chunk_bridge_budget = max(1, path_budget - graph_bridge_budget)
    graph_output = _graph_bridge_association(
        graph=graph,
        current_nodes=current_nodes,
        current_edges=current_edges,
        top_nodes=top_nodes,
        top_edges=top_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        max_hop=max_hop,
        path_budget=graph_bridge_budget,
    )
    chunk_output = _chunk_bridge_association(
        current_nodes=current_nodes,
        current_edges=current_edges,
        root_chunk_ids=root_chunk_ids,
        root_chunk_score_lookup=root_chunk_score_lookup,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_neighbors=chunk_neighbors,
        chunk_budget=chunk_bridge_budget,
    )
    selected_nodes = set(graph_output["selected_path_nodes"]) | set(chunk_output["selected_chunk_nodes"])
    selected_edges = set(graph_output["selected_path_edges"]) | set(chunk_output["selected_chunk_edges"])
    return {
        "current_components": graph_output["current_components"],
        "selected_paths": graph_output["selected_paths"],
        "selected_path_nodes": sorted(selected_nodes),
        "selected_path_edges": sorted(selected_edges),
        "selected_chunks": chunk_output["selected_chunks"],
        "selected_chunk_nodes": chunk_output["selected_chunk_nodes"],
        "selected_chunk_edges": chunk_output["selected_chunk_edges"],
    }


def build_current_relation_categories(edge_ids, graph):
    """Collect coarse relation labels from the current edge set."""
    categories = []
    for edge_id in edge_ids:
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
        categories.append(normalize_relation_category(edge_data))
    return categories


def _expand_chunk_band(chunk_ids, chunk_neighbors):
    expanded = set(chunk_ids)
    for chunk_id in list(chunk_ids):
        expanded.update(chunk_neighbors.get(chunk_id, set()))
    return expanded


def _chunk_coverage_association(
    graph,
    current_nodes,
    current_edges,
    current_categories,
    root_chunk_ids,
    root_chunk_score_lookup,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    chunk_budget,
):
    """Expand by informative adjacent/provenance chunk bands."""
    if chunk_budget <= 0:
        return {"selected_chunks": [], "selected_chunk_nodes": [], "selected_chunk_edges": []}

    covered_chunks = _covered_chunk_ids(current_nodes, current_edges, node_to_chunks, edge_to_chunks)
    covered_band = _expand_chunk_band(covered_chunks, chunk_neighbors)
    current_node_set = set(current_nodes)
    current_edge_set = set(current_edges)
    root_chunk_id_set = set(root_chunk_ids)

    candidate_chunks = set()
    for chunk_id in covered_band:
        candidate_chunks.update(chunk_neighbors.get(chunk_id, set()))
    candidate_chunks -= covered_band

    scored_chunks = []
    for chunk_id in candidate_chunks:
        chunk_nodes = set(chunk_to_nodes.get(chunk_id, set()))
        chunk_edges = set(chunk_to_edges.get(chunk_id, set()))
        new_nodes = chunk_nodes - current_node_set
        new_edges = chunk_edges - current_edge_set
        if not new_nodes and not new_edges:
            continue
        new_relation_count = 0
        for edge_id in new_edges:
            edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
            category = normalize_relation_category(edge_data)
            if category not in current_categories:
                new_relation_count += 1
        new_source_count = _bounded_gain(len(_expand_chunk_band({chunk_id}, chunk_neighbors) - covered_band))
        root_overlap = 1 if chunk_id in root_chunk_id_set else 0
        root_band_alignment = max(
            [root_chunk_score_lookup.get(neighbor_id, 0.0) for neighbor_id in _expand_chunk_band({chunk_id}, chunk_neighbors)],
            default=0.0,
        )
        scored_chunks.append(
            {
                "chunk_id": chunk_id,
                "new_source_count": new_source_count,
                "new_node_count": len(new_nodes),
                "new_edge_count": len(new_edges),
                "new_relation_count": new_relation_count,
                "root_overlap": root_overlap,
                "root_band_alignment": round(root_band_alignment, 6),
                "node_ids": sorted(chunk_nodes),
                "edge_ids": sorted(chunk_edges),
            }
        )

    scored_chunks.sort(
        key=lambda item: (
            -(item["new_node_count"] + item["new_edge_count"]),
            -item["new_relation_count"],
            -item["new_source_count"],
            -item["root_overlap"],
            -item["root_band_alignment"],
            item["chunk_id"],
        )
    )
    selected_chunks = scored_chunks[:chunk_budget]
    selected_nodes = set()
    selected_edges = set()
    for item in selected_chunks:
        selected_nodes.update(item["node_ids"])
        selected_edges.update(item["edge_ids"])
    return {
        "selected_chunks": selected_chunks,
        "selected_chunk_nodes": sorted(selected_nodes),
        "selected_chunk_edges": sorted(selected_edges),
    }


def coverage_association(
    graph,
    current_nodes,
    current_edges,
    root_chunk_ids,
    root_chunk_score_lookup,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    semantic_edge_budget,
    semantic_node_budget,
    semantic_edge_min_score,
    semantic_node_min_score,
):
    """Expand by evidence gain rather than local query similarity.

    Primary signal:
    - how many new supporting chunks / chunk-band items are introduced

    Tie-break signals:
    - whether a new relation category is introduced
    - whether the candidate still overlaps with rooted evidence
    """
    current_nodes = set(current_nodes)
    current_edges = set(current_edges)
    current_categories = set(build_current_relation_categories(current_edges, graph))
    covered_chunks = _covered_chunk_ids(current_nodes, current_edges, node_to_chunks, edge_to_chunks)
    covered_chunk_band = _expand_chunk_band(covered_chunks, chunk_neighbors)
    root_chunk_id_set = set(root_chunk_ids)

    candidate_edges = set()
    candidate_nodes = set()
    candidate_chunks = set(covered_chunk_band)

    for node_id in current_nodes:
        for neighbor_id in graph.neighbors(node_id):
            ek = edge_key(node_id, neighbor_id)
            if ek not in current_edges:
                candidate_edges.add(ek)
            if neighbor_id not in current_nodes:
                candidate_nodes.add(neighbor_id)

    for chunk_id in covered_chunk_band:
        candidate_nodes.update(node_id for node_id in chunk_to_nodes.get(chunk_id, set()) if node_id not in current_nodes)
        candidate_edges.update(edge_id for edge_id in chunk_to_edges.get(chunk_id, set()) if edge_id not in current_edges)
        candidate_chunks.update(chunk_neighbors.get(chunk_id, set()))

    scored_edges = []
    for edge_id in candidate_edges:
        edge_data = graph.get_edge_data(edge_id[0], edge_id[1]) or {}
        source_chunks = set(edge_to_chunks.get(edge_id, set()))
        local_source_chunks = source_chunks & candidate_chunks if candidate_chunks else source_chunks
        expanded_band = _expand_chunk_band(local_source_chunks, chunk_neighbors)
        new_source_count = _bounded_gain(len(expanded_band - covered_chunk_band))
        category = normalize_relation_category(edge_data)
        new_relation = 1 if category not in current_categories else 0
        _, chunk_alignment = support_score(source_chunks, root_chunk_score_lookup)
        root_overlap = len(source_chunks & root_chunk_id_set)
        coverage_gain = new_source_count if new_source_count > 0 else new_relation
        if coverage_gain < semantic_edge_min_score:
            continue
        if coverage_gain <= 0:
            continue
        scored_edges.append(
            {
                "edge": edge_id,
                "score": float(coverage_gain),
                "coverage_gain": float(coverage_gain),
                "new_source_count": new_source_count,
                "new_relation": new_relation,
                "root_overlap": root_overlap,
                "chunk_alignment": round(chunk_alignment, 6),
                "category": category,
                "keywords": edge_data.get("keywords", ""),
                "description": edge_data.get("description", ""),
                "source_chunk_ids": sorted(source_chunks),
            }
        )
    scored_edges.sort(
        key=lambda item: (
            -item["coverage_gain"],
            -item["new_relation"],
            -item["root_overlap"],
            -item["chunk_alignment"],
            item["edge"],
        )
    )
    selected_edges = scored_edges[:semantic_edge_budget]

    expanded_nodes = set()
    for item in selected_edges:
        expanded_nodes.update(item["edge"])

    scored_nodes = []
    for node_id in candidate_nodes:
        node_data = graph.nodes[node_id]
        source_chunks = set(node_to_chunks.get(node_id, set()))
        local_source_chunks = source_chunks & candidate_chunks if candidate_chunks else source_chunks
        expanded_band = _expand_chunk_band(local_source_chunks, chunk_neighbors)
        new_source_count = _bounded_gain(len(expanded_band - covered_chunk_band))
        relation_categories = set()
        bridge_strength = 0
        for neighbor_id in graph.neighbors(node_id):
            if neighbor_id in current_nodes or neighbor_id in expanded_nodes:
                bridge_strength += 1
            edge_data = graph.get_edge_data(node_id, neighbor_id) or {}
            relation_categories.add(normalize_relation_category(edge_data))
        new_relation_count = len(relation_categories - current_categories)
        _, chunk_alignment = support_score(source_chunks, root_chunk_score_lookup)
        root_overlap = len(source_chunks & root_chunk_id_set)
        coverage_gain = new_source_count if new_source_count > 0 else new_relation_count
        if coverage_gain < semantic_node_min_score:
            continue
        if coverage_gain <= 0 and bridge_strength <= 0:
            continue
        scored_nodes.append(
            {
                "id": node_id,
                "score": float(coverage_gain),
                "coverage_gain": float(coverage_gain),
                "new_source_count": new_source_count,
                "new_relation_count": new_relation_count,
                "bridge_strength": bridge_strength,
                "root_overlap": root_overlap,
                "chunk_alignment": round(chunk_alignment, 6),
                "entity_type": node_data.get("entity_type", "UNKNOWN"),
                "description": node_data.get("description", ""),
                "source_chunk_ids": sorted(source_chunks),
            }
        )
    scored_nodes.sort(
        key=lambda item: (
            -item["coverage_gain"],
            -item["bridge_strength"],
            -item["root_overlap"],
            -item["chunk_alignment"],
            item["id"],
        )
    )
    selected_nodes = scored_nodes[:semantic_node_budget]

    semantic_chunk_budget = max(1, min(semantic_edge_budget, semantic_node_budget) // 2)
    chunk_output = _chunk_coverage_association(
        graph=graph,
        current_nodes=current_nodes,
        current_edges=current_edges,
        current_categories=current_categories,
        root_chunk_ids=root_chunk_ids,
        root_chunk_score_lookup=root_chunk_score_lookup,
        chunk_to_nodes=chunk_to_nodes,
        chunk_to_edges=chunk_to_edges,
        node_to_chunks=node_to_chunks,
        edge_to_chunks=edge_to_chunks,
        chunk_neighbors=chunk_neighbors,
        chunk_budget=semantic_chunk_budget,
    )

    final_node_ids = set(current_nodes)
    final_edge_ids = set(current_edges)
    for item in selected_edges:
        final_edge_ids.add(item["edge"])
        final_node_ids.update(item["edge"])
    for item in selected_nodes:
        final_node_ids.add(item["id"])
    final_node_ids.update(chunk_output["selected_chunk_nodes"])
    final_edge_ids.update(chunk_output["selected_chunk_edges"])

    return {
        "selected_edges": selected_edges,
        "selected_nodes": selected_nodes,
        "selected_chunks": chunk_output["selected_chunks"],
        "final_nodes": sorted(final_node_ids),
        "final_edges": sorted(final_edge_ids),
    }


def build_node_role_sets(root_nodes, structural_nodes, semantic_nodes):
    """Tag final nodes by how they entered the graph for later presentation."""
    roles = {}
    for node_id in root_nodes:
        roles[node_id] = "root"
    for node_id in structural_nodes:
        if node_id not in roles:
            roles[node_id] = "structural"
    for node_id in semantic_nodes:
        if node_id not in roles:
            roles[node_id] = "semantic"
    return roles


def build_edge_role_sets(root_edges, structural_edges, semantic_edges):
    """Tag final edges by origin: root, structural bridge, or coverage expansion."""
    roles = {}
    for edge_id in root_edges:
        roles[edge_id] = "root"
    for edge_id in structural_edges:
        if edge_id not in roles:
            roles[edge_id] = "structural"
    for item in semantic_edges:
        edge_id = item["edge"]
        if edge_id not in roles:
            roles[edge_id] = "semantic"
    return roles


def expand_associative_graph(
    query,
    graph,
    root_nodes,
    root_edges,
    root_chunk_ids,
    root_chunk_score_lookup,
    chunk_to_nodes,
    chunk_to_edges,
    node_to_chunks,
    edge_to_chunks,
    chunk_neighbors,
    top_root_nodes,
    top_root_edges,
    max_hop,
    path_budget,
    semantic_edge_budget,
    semantic_node_budget,
    association_rounds,
    semantic_edge_min_score,
    semantic_node_min_score,
):
    """Run alternating bridge-association and coverage-association rounds."""
    current_nodes = set(root_nodes)
    current_edges = set(root_edges)
    all_structural_nodes = set()
    all_structural_edges = set()
    all_semantic_nodes = set()
    all_semantic_edges = []
    round_outputs = []
    last_structural_output = {
        "current_components": [],
        "selected_paths": [],
        "selected_path_nodes": [],
        "selected_path_edges": [],
    }

    for round_index in range(1, max(association_rounds, 1) + 1):
        scored_current_nodes = score_root_nodes(
            query=query,
            root_nodes=current_nodes,
            graph=graph,
            node_to_chunks=node_to_chunks,
            root_chunk_score_lookup=root_chunk_score_lookup,
        )
        scored_current_edges = score_root_edges(
            query=query,
            root_edges=current_edges,
            graph=graph,
            edge_to_chunks=edge_to_chunks,
            root_chunk_score_lookup=root_chunk_score_lookup,
        )
        structural_output = bridge_association(
            graph=graph,
            current_nodes=current_nodes,
            current_edges=current_edges,
            root_chunk_ids=root_chunk_ids,
            root_chunk_score_lookup=root_chunk_score_lookup,
            top_nodes=scored_current_nodes[:top_root_nodes],
            top_edges=scored_current_edges[:top_root_edges],
            chunk_to_nodes=chunk_to_nodes,
            chunk_to_edges=chunk_to_edges,
            node_to_chunks=node_to_chunks,
            edge_to_chunks=edge_to_chunks,
            chunk_neighbors=chunk_neighbors,
            max_hop=max_hop,
            path_budget=path_budget,
        )
        structural_new_nodes = set(structural_output["selected_path_nodes"]) - current_nodes
        structural_new_edges = set(structural_output["selected_path_edges"]) - current_edges
        current_nodes = current_nodes.union(structural_output["selected_path_nodes"])
        current_edges = current_edges.union(structural_output["selected_path_edges"])

        semantic_output = coverage_association(
            graph=graph,
            current_nodes=current_nodes,
            current_edges=current_edges,
            root_chunk_ids=root_chunk_ids,
            root_chunk_score_lookup=root_chunk_score_lookup,
            chunk_to_nodes=chunk_to_nodes,
            chunk_to_edges=chunk_to_edges,
            node_to_chunks=node_to_chunks,
            edge_to_chunks=edge_to_chunks,
            chunk_neighbors=chunk_neighbors,
            semantic_edge_budget=semantic_edge_budget,
            semantic_node_budget=semantic_node_budget,
            semantic_edge_min_score=semantic_edge_min_score,
            semantic_node_min_score=semantic_node_min_score,
        )
        semantic_new_nodes = set(semantic_output["final_nodes"]) - current_nodes
        semantic_new_edges = set(semantic_output["final_edges"]) - current_edges
        current_nodes = set(semantic_output["final_nodes"])
        current_edges = set(semantic_output["final_edges"])

        all_structural_nodes.update(structural_new_nodes)
        all_structural_edges.update(structural_new_edges)
        all_semantic_nodes.update(semantic_new_nodes)
        all_semantic_edges.extend(item for item in semantic_output["selected_edges"] if item["edge"] in semantic_new_edges)
        round_outputs.append(
            {
                "round": round_index,
                "structural_path_count": len(structural_output["selected_paths"]),
                "structural_chunk_bridge_count": len(structural_output.get("selected_chunks", [])),
                "structural_added_node_count": len(structural_new_nodes),
                "structural_added_edge_count": len(structural_new_edges),
                "semantic_added_node_count": len(semantic_new_nodes),
                "semantic_added_edge_count": len(semantic_new_edges),
                "semantic_chunk_coverage_count": len(semantic_output.get("selected_chunks", [])),
                "current_node_count": len(current_nodes),
                "current_edge_count": len(current_edges),
                "structural_path_preview": [
                    {
                        "path": item["path"],
                        "score": item["bridge_gain"],
                        "new_source_count": item["new_source_count"],
                        "path_length": item["path_length"],
                    }
                    for item in structural_output["selected_paths"][:3]
                ],
                "structural_chunk_preview": [
                    {
                        "chunk_id": item["chunk_id"],
                        "bridge_gain": item["bridge_gain"],
                        "frontier_touch": item["frontier_touch"],
                        "new_source_count": item["new_source_count"],
                    }
                    for item in structural_output.get("selected_chunks", [])[:3]
                ],
                "semantic_edge_preview": [
                    {
                        "edge": item["edge"],
                        "score": item["coverage_gain"],
                        "category": item["category"],
                        "new_source_count": item["new_source_count"],
                        "new_relation": item["new_relation"],
                    }
                    for item in semantic_output["selected_edges"][:3]
                ],
                "semantic_node_preview": [
                    {
                        "id": item["id"],
                        "score": item["coverage_gain"],
                        "bridge_strength": item["bridge_strength"],
                        "new_source_count": item["new_source_count"],
                    }
                    for item in semantic_output["selected_nodes"][:3]
                ],
                "semantic_chunk_preview": [
                    {
                        "chunk_id": item["chunk_id"],
                        "new_node_count": item["new_node_count"],
                        "new_edge_count": item["new_edge_count"],
                        "new_source_count": item["new_source_count"],
                    }
                    for item in semantic_output.get("selected_chunks", [])[:3]
                ],
            }
        )
        last_structural_output = structural_output
        if not structural_new_nodes and not structural_new_edges and not semantic_new_nodes and not semantic_new_edges:
            break

    return {
        "final_nodes": sorted(current_nodes),
        "final_edges": sorted(current_edges),
        "all_structural_nodes": sorted(all_structural_nodes),
        "all_structural_edges": sorted(all_structural_edges),
        "all_semantic_nodes": sorted(all_semantic_nodes),
        "all_semantic_edges": all_semantic_edges,
        "rounds": round_outputs,
        "last_structural_output": last_structural_output,
    }
