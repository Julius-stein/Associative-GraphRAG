import argparse
import json
from pathlib import Path

import networkx as nx


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: Consolas, "SFMono-Regular", monospace;
      background: #f6f3ea;
      color: #1d1d1b;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 320px 1fr 360px;
      height: 100vh;
    }}
    .panel {{
      padding: 14px;
      overflow: auto;
      border-right: 1px solid #d5cfbf;
      background: #fbf8f1;
    }}
    .panel:last-child {{
      border-right: 0;
      border-left: 1px solid #d5cfbf;
    }}
    #network {{
      height: 100vh;
      background:
        radial-gradient(circle at 20% 20%, rgba(212,188,126,0.18), transparent 25%),
        radial-gradient(circle at 80% 10%, rgba(89,131,146,0.14), transparent 20%),
        linear-gradient(180deg, #f5f0e2 0%, #efe7d6 100%);
    }}
    h1, h2 {{
      margin: 0 0 10px 0;
      font-size: 16px;
    }}
    .meta {{
      font-size: 12px;
      line-height: 1.5;
      color: #564f45;
      margin-bottom: 12px;
    }}
    input {{
      width: 100%;
      box-sizing: border-box;
      padding: 10px 12px;
      border: 1px solid #b9ae98;
      background: #fffdf8;
      margin-bottom: 12px;
      border-radius: 8px;
    }}
    .item {{
      padding: 10px;
      margin-bottom: 8px;
      border: 1px solid #ddd4c4;
      border-radius: 8px;
      background: #fffdf8;
      cursor: pointer;
    }}
    .item:hover {{
      border-color: #8d7750;
    }}
    .name {{
      font-weight: 700;
      font-size: 12px;
      margin-bottom: 6px;
      word-break: break-word;
    }}
    .sub {{
      font-size: 11px;
      color: #6b6357;
      word-break: break-word;
    }}
    .detail-block {{
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid #ddd4c4;
    }}
    .detail-title {{
      font-weight: 700;
      margin-bottom: 8px;
      font-size: 12px;
    }}
    .detail-text {{
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .empty {{
      color: #7b7468;
      font-size: 12px;
      line-height: 1.6;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="panel">
      <h1>{title}</h1>
      <div class="meta">
        nodes shown: {shown_nodes}<br>
        edges shown: {shown_edges}<br>
        source graph: {source_nodes} nodes / {source_edges} edges
      </div>
      <input id="search" placeholder="Search node label">
      <div id="nodeList"></div>
    </aside>
    <main id="network"></main>
    <aside class="panel">
      <h2>Selection</h2>
      <div id="details" class="empty">Click a node or edge to inspect details.</div>
    </aside>
  </div>
  <script>
    const rawNodes = {nodes_json};
    const rawEdges = {edges_json};
    const nodeMap = new Map(rawNodes.map(n => [n.id, n]));
    const edgeMap = new Map(rawEdges.map(e => [e.id, e]));

    const nodes = new vis.DataSet(rawNodes);
    const edges = new vis.DataSet(rawEdges);
    const container = document.getElementById('network');
    const nodeList = document.getElementById('nodeList');
    const details = document.getElementById('details');
    const search = document.getElementById('search');

    const network = new vis.Network(container, {{ nodes, edges }}, {{
      autoResize: true,
      interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
      physics: {{
        barnesHut: {{
          gravitationalConstant: -7000,
          springLength: 130,
          springConstant: 0.025
        }},
        stabilization: {{ iterations: 250 }}
      }},
      nodes: {{
        shape: 'dot',
        scaling: {{ min: 6, max: 28 }},
        font: {{ size: 12, face: 'Consolas' }},
        borderWidth: 1
      }},
      edges: {{
        width: 1,
        color: {{ color: '#8e8576', highlight: '#8d5d1d' }},
        smooth: false,
        font: {{ size: 10, align: 'top' }}
      }}
    }});

    function renderNodeList(items) {{
      nodeList.innerHTML = '';
      items.forEach(node => {{
        const div = document.createElement('div');
        div.className = 'item';
        div.innerHTML = `
          <div class="name">${{node.label}}</div>
          <div class="sub">degree: ${{node.degree}} | type: ${{node.entity_type || 'UNKNOWN'}}</div>
        `;
        div.onclick = () => {{
          network.selectNodes([node.id]);
          network.focus(node.id, {{ scale: 1.1, animation: true }});
          showNode(node.id);
        }};
        nodeList.appendChild(div);
      }});
    }}

    function showNode(id) {{
      const node = nodeMap.get(id);
      const connected = network.getConnectedEdges(id).map(eid => edgeMap.get(eid)).filter(Boolean);
      details.innerHTML = `
        <div class="detail-block">
          <div class="detail-title">Node</div>
          <div class="detail-text"><strong>${{node.label}}</strong></div>
        </div>
        <div class="detail-block">
          <div class="detail-title">Meta</div>
          <div class="detail-text">type: ${{node.entity_type || 'UNKNOWN'}}\ndegree: ${{node.degree}}\nsource_id: ${{node.source_id || ''}}</div>
        </div>
        <div class="detail-block">
          <div class="detail-title">Description</div>
          <div class="detail-text">${{node.description || ''}}</div>
        </div>
        <div class="detail-block">
          <div class="detail-title">Connected edges (${{connected.length}})</div>
          <div class="detail-text">${{connected.slice(0, 20).map(e => `${{e.from_label}} -> ${{e.to_label}} | ${{e.keywords || ''}}`).join('\\n')}}</div>
        </div>
      `;
    }}

    function showEdge(id) {{
      const edge = edgeMap.get(id);
      details.innerHTML = `
        <div class="detail-block">
          <div class="detail-title">Edge</div>
          <div class="detail-text"><strong>${{edge.from_label}}</strong> -> <strong>${{edge.to_label}}</strong></div>
        </div>
        <div class="detail-block">
          <div class="detail-title">Meta</div>
          <div class="detail-text">weight: ${{edge.weight}}\nkeywords: ${{edge.keywords || ''}}\nsource_id: ${{edge.source_id || ''}}</div>
        </div>
        <div class="detail-block">
          <div class="detail-title">Description</div>
          <div class="detail-text">${{edge.description || ''}}</div>
        </div>
      `;
    }}

    network.on('click', params => {{
      if (params.nodes.length) {{
        showNode(params.nodes[0]);
      }} else if (params.edges.length) {{
        showEdge(params.edges[0]);
      }}
    }});

    search.addEventListener('input', () => {{
      const q = search.value.trim().toLowerCase();
      if (!q) {{
        renderNodeList(rawNodes.slice(0, 200));
        return;
      }}
      const filtered = rawNodes.filter(n => n.label.toLowerCase().includes(q)).slice(0, 200);
      renderNodeList(filtered);
    }});

    renderNodeList(rawNodes.slice(0, 200));
  </script>
</body>
</html>
"""


def color_for_type(entity_type: str) -> str:
    base = abs(hash(entity_type or "UNKNOWN")) % 0xFFFFFF
    return f"#{base:06x}"


def build_subgraph(graph: nx.Graph, max_nodes: int) -> nx.Graph:
    ranked_nodes = [node for node, _ in sorted(graph.degree, key=lambda x: x[1], reverse=True)[:max_nodes]]
    return graph.subgraph(ranked_nodes).copy()


def export_html(graph_path: Path, output_path: Path, max_nodes: int):
    graph = nx.read_graphml(graph_path)
    subgraph = build_subgraph(graph, max_nodes=max_nodes)

    nodes = []
    for node_id, attrs in sorted(subgraph.nodes(data=True), key=lambda x: subgraph.degree(x[0]), reverse=True):
        label = str(node_id).strip('"')
        entity_type = attrs.get("entity_type", "").strip('"')
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "value": max(subgraph.degree(node_id), 1),
                "degree": int(subgraph.degree(node_id)),
                "entity_type": entity_type,
                "description": attrs.get("description", ""),
                "source_id": attrs.get("source_id", ""),
                "color": color_for_type(entity_type),
            }
        )

    edges = []
    for idx, (src, tgt, attrs) in enumerate(subgraph.edges(data=True)):
        edges.append(
            {
                "id": f"e{idx}",
                "from": src,
                "to": tgt,
                "from_label": str(src).strip('"'),
                "to_label": str(tgt).strip('"'),
                "label": "",
                "weight": attrs.get("weight", ""),
                "keywords": attrs.get("keywords", ""),
                "description": attrs.get("description", ""),
                "source_id": attrs.get("source_id", ""),
            }
        )

    html = HTML_TEMPLATE.format(
        title=graph_path.parent.name + " graph view",
        shown_nodes=subgraph.number_of_nodes(),
        shown_edges=subgraph.number_of_edges(),
        source_nodes=graph.number_of_nodes(),
        source_edges=graph.number_of_edges(),
        nodes_json=json.dumps(nodes, ensure_ascii=False),
        edges_json=json.dumps(edges, ensure_ascii=False),
    )
    output_path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="Path to graphml file")
    parser.add_argument("--output", required=True, help="Output html path")
    parser.add_argument("--max-nodes", type=int, default=180, help="Number of top-degree nodes to visualize")
    args = parser.parse_args()
    export_html(Path(args.graph), Path(args.output), max_nodes=args.max_nodes)


if __name__ == "__main__":
    main()
