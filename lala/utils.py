from math import prod

def view(shape, list, offset=0):
    return [list[i] if len(shape)==1 else view(shape[1:], list) for i in range(shape[0])] 

#TODO: Implement broadcast of shapes according to the pytorch broadcast rule
def get_broadcast_shape(s0, s1):
    pass
def _to_python_list(arr, shape, off=0): 
    return [arr[off: off+shape[0]][i] for i in range(shape[0])] if len(shape) == 1 else [_to_python_list(arr, shape[1: ], i*prod(shape[1:])) for i in range(shape[0])]


def _get_list_shape(l):
    return _get_list_shape(l[0]) + (len(l),)if isinstance(l[0], list) else (len(l),)

def get_list_shape(l): return tuple(reversed(_get_list_shape(l)))

def graph_html(root, filename="graph.html"):
    from .tensor import Tensor
    nodes, edges = [], []
    visited = set()

    def visit(node, is_root=False):
        if node in visited or node is None: return
        visited.add(node)
        if isinstance(node, Tensor):
            if is_root:
                nodes.append({"id": id(node), "label": str(node) if node.label is None else node.label, "shape": "box", "color": "#aa5555"})
            elif node.requires_grad:
                nodes.append({"id": id(node), "label": str(node) if node.label is None else node.label, "shape": "box"})
            else:
                nodes.append({"id": id(node), "label": str(node) if node.label is None else node.label + str(node.shape), "shape": "box", "color": "yellow"})
            if node.src is not None:
                nodes.append({"id": id(node.src), "label": node.src.name, "color": "green", "shape": "circle"})
                edges.append({"from": id(node), "to": id(node.src)})
                for parent in node.src.operands:
                    edges.append({"from": id(parent), "to": id(node.src)})
                    visit(parent)
        else:
            nodes.append({"id": id(node), "label": str(node), "shape": "box", "color": "#ff5555"})

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
