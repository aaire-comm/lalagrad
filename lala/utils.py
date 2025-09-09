from .lala import  Matrice

def export_html(root, filename="graph.html"):
    nodes, edges = [], []
    visited = set()

    def visit(node, is_root=False):
        if node in visited or node is None: return
        visited.add(node)
        if isinstance(node, Matrice):
            if is_root:
                nodes.append({"id": id(node), "label": str(node) if node.label is None else node.label, "shape": "box", "color": "#aa5555"})
            elif node.requires_grad:
                nodes.append({"id": id(node), "label": str(node) if node.label is None else node.label, "shape": "box"})
            else:
                nodes.append({"id": id(node), "label": str(node) if node.label is None else node.label, "shape": "box", "color": "yellow"})
            if node.grad_fn is not None:
                nodes.append({"id": id(node.grad_fn), "label": node.grad_fn.name, "color": "green", "shape": "circle"})
                edges.append({"from": id(node), "to": id(node.grad_fn)})
                for parent in node.grad_fn.operands:
                    edges.append({"from": id(parent), "to": id(node.grad_fn)})
                    visit(parent)
        else:
            nodes.append({"id": id(node), "label": str(node), "shape": "box"})

    visit(root, True)

    html = f"""
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    </head>
    <body>
      <div id="mynetwork" style="height: 600px;"></div>
      <div id="detail" style="position: fixed; width: 400px; height: 400px; top:0; left: 0; border: black; backgroung-color: 'red'"/>
      <script>
        var selectedNode;
        var nodes = new vis.DataSet({nodes});
        var edges = new vis.DataSet({edges});
        var container = document.getElementById('mynetwork');
        var details = document.getElementById('mynetwork');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{layout: {{ hierarchical: {{ direction: "LR", nodeSpacing: 150, levelSeparation: 200 }} }}, nodes: {{ shape: "box" }}, edges: {{ arrows: "to"}}, physics: false}}
        var network = new vis.Network(container, data, options);

        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                selectedNode = node
                h1 = document.createElement('h1')
                h1.text = node.label
                details.addChild()
            }}
        }});

      </script>
    </body>
    </html>
    """
    with open(filename, "w") as f:
        f.write(html)
