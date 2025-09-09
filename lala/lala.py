"""
Function types:
    Unary -> take a Matrice return Matrice of the same shape
    Binary -> take a Matrice and return a Matrice
    SReduce (Scalar reduce) take a Matrice and return single element Matrice (implicite autograd allowed)
"""

from typing import List, Optional
import math


def _transpose(data): return [[row[i] for row in data] for i in range(len(data[0]))]

def _dot(v1, v2): return sum([a * b for a, b in zip(v1, v2)])




    
class Function:
    def __init__(self, name, type_, *args):
        self.name = name
        self.op_type = type_
        self.operands = args
        
        self.grad = None

    def __call__(self): return Matrice(self.forward(), grad_fn=self, requires_grad=any(operand.requires_grad for operand in self.operands))
    
    def backward(self, upstream_m):
        for operand in self.operands:
            if isinstance(operand, Matrice) and operand.requires_grad:
                if self.op_type == "Binary":
                    grad =   Matrice(self.gradient(operand))
                elif self.op_type == "View":
                    grad =   Matrice(self.gradient(upstream_m))
                elif self.op_type == "Unary" or self.op_type == "SReduce":
                    grad =   Matrice(self.gradient())
                else:
                    grad =   Matrice(self.gradient(operand, upstream_m))

                if upstream_m is not None and self.op_type != "View":
                    grad *= upstream_m

                if operand.grad is None:
                    operand.grad = grad
                else: 
                    operand.grad += grad

                operand.grad.requires_grad = None
                operand.grad.grad_fun = None

                operand.backward(operand.grad)



class Mean(Function):
    def __init__(self, *args): super().__init__("MEAN", "SReduce", *args)

    def forward(self):
        size = self.operands[0].numel()
        print("mean operands:", self.operands)
        return [[sum(sum(row) for row in self.operands[0].data)/size]]
    
    def gradient(self):
        m = self.operands[0]
        return [[1/m.numel() for _ in row] for row in m.data]
        
        
class Transpose(Function):
    def __init__(self, *args): 
        super().__init__("Transpose", "VIEW", *args)

    def forward(self): 
        m = self.operands[0]
        return _transpose(m.data)
    
    def gradient(self, upstream_m):
        return _transpose(upstream_m.data)
        
    
class Add(Function):
    def __init__(self, *args):
        super().__init__("Add", "Binary", *args)
    
    def forward(self):
        return [[e1 + e2 for e1, e2 in zip(row1, row2) ] for row1, row2 in zip(*(operand.data for operand in self.operands)) ] 
    

    def gradient(self, w_r_t): 
        assert w_r_t in self.operands, "w_r_t is not an operand of this grad_fn"
        return  [[1 for _ in row] for row in w_r_t.data]
    
class Sub(Function):
    def __init__(self, *args):
        super().__init__("Sub", "Binary", *args)
    
    def forward(self):
        return [[e1 - e2 for e1, e2 in zip(row1, row2) ] for row1, row2 in zip(*(operand.data for operand in self.operands)) ] 
    

    def gradient(self, w_r_t): 
        assert w_r_t in self.operands, "w_r_t is not an operand of this grad_fn"
        a, b = self.operands
        if w_r_t is a:
            return  [[1 for _ in row] for row in w_r_t.data]
        else:
            return  [[-1 for _ in row] for row in w_r_t.data]
            
    

class Sum(Function):
    def __init__(self, *args): 
        super().__init__("Sum", "SReduce" if args[1] is None else "Unary", *args)
        


    def forward(self, dim=None): 
        m, dim = self.operands
        print(m.data)
        if dim is None:
            return [[sum([sum(row) for row in m.data])]]
        elif dim == 0:
            _t = m.transpose()
            return _transpose([[sum(row)]for row in _t.data])
        else:
            return _transpose([[sum(row)]for row in m.data])
    
    def gradient(self):
        rows, cols =  self.operands[0].shape
        return [[1 for _ in range(cols)] for __ in range(rows)]

        

class Mul(Function):
    def __init__(self, *args): super().__init__("ElMul", "Binary", *args)

    def forward(self): return [[e1 * e2 for e1, e2 in zip(row1, row2) ] for row1, row2 in zip(*(operand.data for operand in self.operands)) ]
    def gradient(self, w_r_t): return self.operands[0].data if w_r_t is self.operands[1] else self.operands[0].data


class ScalarPower(Function):
    def __init__(self, *args): super().__init__("ElPow", "Unary", *args)

    def forward(self): 
        m, p = self.operands
        return [[e ** p for e in raw] for raw in m.data]
        
    def gradient(self): 
        m = self.operands[0]
        return m.smul(self.operands[1]).data

    

class ScalarMul(Function):
    def __init__(self, *args): super().__init__("SMul", "Unary", *args)

    def forward(self): 
        m, p = self.operands
        return [[e * p for e in raw] for raw in m.data]
        
    def gradient(self, w_r_t): return [[self.operands[1] for _ in raw] for raw in w_r_t.data]

        
class Matmul(Function):
    _matmul = lambda m1, m2: [[_dot(row, col) for col in m2] for row in m1 ]
    
    def __init__(self, *args):
        super().__init__("MMul", "MBinary", *args)
        
    def forward(self):
        m1, m2 = self.operands
        m2_t = m2.transpose()
        return [[_dot(row, col) for col in m2_t.data] for row in m1.data]

    def gradient(self, w_r_t, upstream_m): 
        a, b = self.operands
        if w_r_t is a:
            assert upstream_m.shape[1] == b.shape[1]
            B_T = b.transpose().data
            return Matmul._matmul(upstream_m.data, B_T)
        else:
            assert upstream_m.shape[0] == a.shape[0]
            A_T = a.transpose().data
            return Matmul._matmul(A_T, upstream_m.data)

        
        


class Matrice: 
    def __init__(self, data: List[int], grad_fn: Optional[Function]=None, label=None,  requires_grad=False):
        self.data, self.grad_fn, self.requires_grad = data, grad_fn, requires_grad
        self.shape = (len(data), len(data[0]))
        self.grad = None
        self.label = label
    
    def __repr__(self):
        return f"Matrice(shape=<{self.shape}> grad_fn=<{None if self.grad_fn is None else self.grad_fn.name}>)"

    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0], f"matmul of invalid shapes {self.shape, other.shape}"
        grad_fn = Matmul(self, other)
        return grad_fn()

    def dot(self, other): return (self * other).sum(1)

    def __add__(self, other): return Add(self, other)()
    def __sub__(self, other): return Sub(self, other)()
    def __mul__(self, other): return Mul(self, other)()
        

    def transpose(self): return Transpose(self)()

    def smul(self, scalar): return ScalarMul(self, scalar)()

    def spow(self, scalar): return ScalarPower(self, scalar)()

    def mean(self, dim=None): return Mean(self, dim)()

    def sum(self, dim=None): return Sum(self, dim)()

    def numel(self): return math.prod(self.shape)

    def backward(self, upstream_m=None):
        assert upstream_m is not None or self.grad_fn.op_type == "SReduce", "implicit backward only defined for single elemnt matrices"
        assert self.requires_grad, "matrice doesn't requre_grad"
        if not self.grad_fn:
            return 
        self.grad_fn.backward(upstream_m)


def export_html(root, filename="graph.html"):
    nodes, edges = [], []
    visited = set()

    def visit(node, is_root=False):
        if node in visited or node is None: return
        visited.add(node)
        if isinstance(node, Matrice):
            if is_root:
                nodes.append({"id": id(node), "label": str(node) if node.label is None else node.label, "shape": "box", "color": "#aa5555"})
            else:
                nodes.append({"id": id(node), "label": str(node) if node.label is None else node.label, "shape": "box"})
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
