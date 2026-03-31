"""Rebuild prompt_context from an existing retrieval file without rerunning retrieval.

This utility is meant for clean organization-side ablations: it preserves the
existing retrieval output and simply prepends a group-dossier view built from
stored knowledge-group metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_group_summary(group):
    nodes = group.get("nodes", [])
    themes = [theme for theme in group.get("relation_themes", []) if theme]
    previews = group.get("source_previews", [])
    lead_entities = ", ".join(nodes[:3]) if nodes else "the retrieved evidence"
    parts = [f"This group centers on {lead_entities}."]
    if themes:
        parts.append(f"It mainly connects evidence through {', '.join(themes[:3])}.")
    if group.get("supporting_chunk_ids"):
        parts.append(f"It is supported by {len(group['supporting_chunk_ids'])} chunks.")
    if previews:
        parts.append(f"Representative evidence: {previews[0].get('preview', '')[:180]}")
    return " ".join(parts)


def _build_dossier_block(record):
    sections = ["-----Knowledge Group Dossiers-----"]
    for group in record.get("knowledge_groups", []):
        key_entities = " | ".join(group.get("nodes", [])[:6]) or "n/a"
        key_relations = group.get("edges", [])[:5]
        sections.append(
            "\n".join(
                [
                    f"[{group.get('group_id', 'kg-xx')}] score={group.get('group_score', 0):.4f} "
                    f"nodes={group.get('node_count', 0)} edges={group.get('edge_count', 0)}",
                    f"Summary: {_build_group_summary(group)}",
                    f"Themes: {' | '.join(group.get('relation_themes', [])) or 'n/a'}",
                    f"Key Entities: {key_entities}",
                    "Key Relations:",
                    *(
                        [f"- {edge[0]} -> {edge[1]}" for edge in key_relations]
                        if key_relations
                        else ["- n/a"]
                    ),
                    "Linked Source Previews:",
                    "```csv",
                    "chunk_id,\tpreview",
                    *[
                        f"{item.get('chunk_id','')},\t{item.get('preview','').replace(chr(10), ' ')}"
                        for item in group.get("source_previews", [])[:4]
                    ],
                    "```",
                ]
            )
        )
    return "\n".join(sections).strip()


def repack_prompt_context(input_path: Path, output_path: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError("Expected a retrieval JSON with top-level 'results'.")

    for record in payload["results"]:
        dossier = _build_dossier_block(record)
        original = record.get("prompt_context", "")
        record["prompt_context"] = f"{dossier}\n\n{original}".strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Repack an existing retrieval file with group dossiers.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()
    repack_prompt_context(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
