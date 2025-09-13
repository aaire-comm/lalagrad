"""
Function types:
    Unary -> take a Matrice return Matrice of the same shape
    Binary -> take a Matrice and return a Matrice
    SReduce (Scalar reduce) take a Matrice and return single element Matrice (implicite autograd allowed)
"""
import ctypes
from typing import List, Optional
import math
from .utils import graph_html
        

    
class Function:
    def __init__(self, name, type_, *args):
        self.name = name
        self.op_type = type_
        self.operands = args
        
        self.grad = None

    @classmethod
    def elem_wise_validate(cls, fn):
        shape = fn.operands[0].shape
        return all(shape == op.shape for op in fn.operands), f"Incompatable shape for function {fn.name}"


    def validate(self):
        return True, ""

    def __call__(self): 
        assertion, msg = self.validate()
        assert assertion, msg
        return Matrice(self.forward(), grad_fn=self, requires_grad=any(operand.requires_grad if isinstance(operand, Matrice) else False for operand in self.operands))
    
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

                if upstream_m is not None and self.op_type != "View" and self.name != "MMul":
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
        return [[sum(sum(row) for row in self.operands[0].data)/size]]
    
    def gradient(self):
        m = self.operands[0]
        return [[1/m.numel() for _ in row] for row in m.data]
        
        
class Transpose(Function):
    def __init__(self, *args): 
        super().__init__("Transpose", "View", *args)

    def forward(self): 
        m = self.operands[0]
        return _transpose(m.data)
    
    def gradient(self, upstream_m):
        return _transpose(upstream_m.data)
        
    
class Add(Function):
    def __init__(self, *args):
        super().__init__("Add", "Binary", *args)
    
    def validate(self):
        a, b = self.operands
        return a.shape == b.shape, f"Matrices for different shape for {self.name}"
    
    def forward(self):
        return [[e1 + e2 for e1, e2 in zip(row1, row2) ] for row1, row2 in zip(*(operand.data for operand in self.operands)) ] 
    

    def gradient(self, w_r_t): 
        assert w_r_t in self.operands, "w_r_t is not an operand of this grad_fn"
        return  [[1 for _ in row] for row in w_r_t.data]
    
class Sub(Function):
    def __init__(self, *args):
        super().__init__("Sub", "Binary", *args)
    
    def validate(self): 
        return Function.elem_wise_validate(self)
        
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
        
    def gradient(self): 
        w_r_t, scalar = self.operands
        return [[scalar for _ in raw] for raw in w_r_t.data]

class Relu(Function):
    def __init__(self, *args): super().__init__("Relu", "Unary", *args)

    def forward(self):
        m = self.operands[0]
        return [[e if e > 0 else 0 for e in row] for row in m.data]
    
    def gradient(self):
        m = self.operands[0]
        return [[1 if e > 0 else 0 for e in row] for row in m.data]

    
        
class Matmul(Function):
    _matmul = lambda m1, m2: [[_dot(row, col) for col in _transpose(m2)] for row in m1 ]
    
    def __init__(self, *args):
        super().__init__("MMul", "MBinary", *args)
        
    def forward(self):
        m1, m2 = self.operands
        m2_t = m2.transpose()
        return [[_dot(row, col) for col in m2_t.data] for row in m1.data]


    def gradient(self, w_r_t, upstream_m): 
        lhs, rhs = self.operands
        
        if w_r_t is lhs:
            lgrad = Matrice(Matmul._matmul(upstream_m.data, rhs.transpose().data))
            if len(lhs.shape) < len(lgrad.shape):
                lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
            return lgrad.data
        else:
            rgrad = Matrice(Matmul._matmul(lhs.transpose().data, upstream_m.data))
            if len(rhs.shape) < len(rgrad.shape):
                rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
            
            return rgrad.data


class Matrice: 
    def __init__(self, data: List[int], shape=None, grad_fn: Optional[Function]=None, label=None,  requires_grad=False):
        self.data, self.grad_fn, self.requires_grad = data, grad_fn, requires_grad
        self.shape = (len(data), len(data[0])) if shape is None else shape
        self.grad = None
        self.label = label
    
    def __repr__(self):
        return f"Matrice(shape=<{self.shape}> grad_fn=<{None if self.grad_fn is None else self.grad_fn.name}>)"
    
    def visualize(self, file_name="graph.html"):
        graph_html(self, Matrice, filename=file_name)
        print(f"Computaion Graph exported as {file_name}")

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


def _transpose(data): return [[row[i] for row in data] for i in range(len(data[0]))]

def _dot(v1, v2): return sum([a * b for a, b in zip(v1, v2)])

