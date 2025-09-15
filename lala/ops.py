from .utils import *
from .dtype import int32
from .blob import Blob

def _transpose(data):
    return [[row[i] for row in data] for i in range(len(data[0]))]

def _dot(v1, v2): return sum([a * b for a, b in zip(v1, v2)])


class Operation:
    def __init__(self, name, type_, op_lib, *args):
        self.name = name
        self.op_type = type_
        self.operands = args
        self.grad = None
        self.op_lib = op_lib

    @classmethod
    def elem_wise_validate(cls, fn):
        shape = fn.operands[0].shape
        return all(shape == op.shape for op in fn.operands), f"Incompatable shape for Operation {fn.name}"


    def validate(self):
        return True, ""

    def __call__(self): 
        assertion, msg = self.validate()
        assert assertion, msg
        dtype = max(operand.dtype for operand in self.operands)
        return Blob(data=self.forward(), grad_fn=self, requires_grad=any(operand.requires_grad if isinstance(operand, Blob) else False for operand in self.operands))
    
    def backward(self, upstream_m):
        for operand in self.operands:
            if isinstance(operand, Blob) and operand.requires_grad:
                if self.op_type == "Binary":
                    grad =   self.gradient(operand)
                elif self.op_type == "View":
                    grad =   self.gradient(upstream_m)
                elif self.op_type == "Unary" or self.op_type == "SReduce":
                    grad =   self.gradient()
                else:
                    grad =   self.gradient(operand, upstream_m)

                if upstream_m is not None and self.op_type != "View" and self.name != "MMul":
                    grad *= upstream_m

                if operand.grad is None:
                    operand.grad = grad
                else: 
                    operand.grad += grad

                operand.grad.requires_grad = None
                operand.grad.grad_fun = None

                operand.backward(operand.grad)


class Mean(Operation):
    def __init__(self, *args): super().__init__("MEAN", "SReduce", *args)

    def forward(self):
        size = self.operands[0].numel()
        return [[sum(sum(row) for row in self.operands[0].data)/size]]
    
    def gradient(self):
        m = self.operands[0]
        return [[1/m.numel() for _ in row] for row in m.data]
        
        
class Transpose(Operation):
    def __init__(self, *args): 
        super().__init__("Transpose", "View", *args)

    def forward(self): 
        m = self.operands[0]
        return _transpose(m.data)
    
    def gradient(self, upstream_m):
        return _transpose(upstream_m.data)
        
    
class Add(Operation):
    def __init__(self, *args):
        super().__init__("Add", "Binary", *args)
    
    def validate(self):
        a, b = self.operands
        return a.shape == b.shape, f"Blobs for different shape for {self.name}"
    
    def forward(self):
        rhs = self.operands[0]
        return [[e1 + e2 for e1, e2 in zip(row1, row2) ] for row1, row2 in zip(*(operand.data for operand in self.operands)) ] 
    

    def gradient(self, w_r_t): 
        assert w_r_t in self.operands, "w_r_t is not an operand of this grad_fn"
        return  [[1 for _ in row] for row in w_r_t.data]
    
class Sub(Operation):
    def __init__(self, *args):
        super().__init__("Sub", "Binary", *args)
    
    def validate(self): 
        return Operation.elem_wise_validate(self)
        
    def forward(self):
        return [[e1 - e2 for e1, e2 in zip(row1, row2) ] for row1, row2 in zip(*(operand.data for operand in self.operands)) ] 
    

    def gradient(self, w_r_t): 
        assert w_r_t in self.operands, "w_r_t is not an operand of this grad_fn"
        a, b = self.operands
        if w_r_t is a:
            return  [[1 for _ in row] for row in w_r_t.data]
        else:
            return  [[-1 for _ in row] for row in w_r_t.data]
            
    

class Sum(Operation):
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

        

class Mul(Operation):
    def __init__(self, *args): super().__init__("ElMul", "Binary", *args)

    def forward(self): return [[e1 * e2 for e1, e2 in zip(row1, row2) ] for row1, row2 in zip(*(operand.data for operand in self.operands)) ]
    def gradient(self, w_r_t): return self.operands[0].data if w_r_t is self.operands[1] else self.operands[0].data


class ScalarPower(Operation):
    def __init__(self, *args): super().__init__("ElPow", "Unary", *args)

    def forward(self): 
        m, p = self.operands
        return [[e ** p for e in raw] for raw in m.data]
        
    def gradient(self): 
        m = self.operands[0]
        return m.smul(self.operands[1]).data

    

class ScalarMul(Operation):
    def __init__(self, *args): super().__init__("SMul", "Unary", *args)

    def forward(self): 
        m, p = self.operands
        return [[e * p for e in raw] for raw in m.data]
        
    def gradient(self): 
        w_r_t, scalar = self.operands
        return [[scalar for _ in raw] for raw in w_r_t.data]

class Relu(Operation):
    def __init__(self, *args): super().__init__("Relu", "Unary", *args)

    def forward(self):
        m = self.operands[0]
        res = Blob.empty(*m.shape, dtype=int32)
        lib.relu_int(m._data.ptr, res._data.ptr, res._data.size)
        return res
    
    def gradient(self):
        m = self.operands[0]
        return [[1 if e > 0 else 0 for e in row] for row in m.data]

    
        
class Matmul(Operation):
    _matmul = lambda m1, m2: [[_dot(row, col) for col in _transpose(m2)] for row in m1 ]
    
    def __init__(self, *args):
        super().__init__("MMul", "MBinary", *args)
        
    def forward(self):
        m1, m2 = self.operands
        numel = prod((m1.shape[0], m2.shape[1]))
        res = Blob(numel=numel, dtype=int32)
        lib.matmul_int(m1.shape[0], m1.shape[1], m2.shape[1], m1._ptr(), m2.ptr(), res._ptr())
        return res


    def gradient(self, w_r_t, upstream_m): 
        lhs, rhs = self.operands
        
        if w_r_t is lhs:
            lgrad = Blob(Matmul._matmul(upstream_m.data, rhs.transpose().data))
            return lgrad.data
        else:
            rgrad = Blob(Matmul._matmul(lhs.transpose().data, upstream_m.data))
            return rgrad.data
        
class ViewOp(Operation):
    def __init__(self, name, type_, *args):
        super().__init__("View", "View", *args)
    
    def forward(self):
        return 

    def gradient(self):
        return 

