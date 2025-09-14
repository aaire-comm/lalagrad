
"""
Function types:
    Unary -> take a Matrice return Matrice of the same shape
    Binary -> take a Matrice and return a Matrice
    SReduce (Scalar reduce) take a Matrice and return single element Matrice (implicite autograd allowed)
"""
import ctypes
from typing import List, Optional, Union, Tuple, Any
import math
from .utils import graph_html, view, get_list_shape, _to_list
from .c.lib_loader import lib, libc
import numpy as np
import time



    
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
        return Matrice(data=self.forward(), grad_fn=self, requires_grad=any(operand.requires_grad if isinstance(operand, Matrice) else False for operand in self.operands))
    
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
        res = Matrice.empty(*m.shape, dtype=int32)
        lib.relu_int(m._data.ptr, res._data.ptr, res._data.size)
        return res
    
    def gradient(self):
        m = self.operands[0]
        return [[1 if e > 0 else 0 for e in row] for row in m.data]

    
        
class Matmul(Function):
    _matmul = lambda m1, m2: [[_dot(row, col) for col in _transpose(m2)] for row in m1 ]
    
    def __init__(self, *args):
        super().__init__("MMul", "MBinary", *args)
        
    def forward(self):
        m1, m2 = self.operands
        res = Matrice(m1.shape[0], m2.shape[1], dtype=int32)
        lib.matmul_int(m1.shape[0], m1.shape[1], m2.shape[1], m1._data.ptr, m2._data.ptr, res._data.ptr)
        return res


    def gradient(self, w_r_t, upstream_m): 
        lhs, rhs = self.operands
        
        if w_r_t is lhs:
            lgrad = Matrice(Matmul._matmul(upstream_m.data, rhs.transpose().data))
            return lgrad.data
        else:
            rgrad = Matrice(Matmul._matmul(lhs.transpose().data, upstream_m.data))
            return rgrad.data
        
class ViewOp(Function):
    def __init__(self, name, type_, *args):
        super().__init__("View", "View", *args)
    
    def forward(self):
        return 

    def gradient(self):
        return 


class Dtype:
    def __init__(self, name: str, bytes: int, base: Union[ctypes.c_int, ctypes.c_float]):
        self.name = name
        self.bytes = bytes
        self.base = base
        

int32  = Dtype("int32", 4, ctypes.c_int)
float32  = Dtype("float32", 4, ctypes.c_float)


class Blob:
    def __init__(self, ptr: Optional[ctypes.POINTER]=None, dtype: Dtype=float32, size: Optional[int]=None, zero_init=True):
        base = dtype.base
        self.size = size
        if ptr is None:
            if zero_init:
                ptr = (base * size)()
            else:
                ptr = libc.malloc(dtype.bytes*size)

        self.ptr = ctypes.cast(ptr, ctypes.POINTER(base))        


    
    @classmethod
    def from_list(cls, _from: List[List[Any]], dtype=float32):
        np_dtype, base = (np.float32, ctypes.c_float) if dtype is float32 else (np.int32, ctypes.c_int)
        arr = np.array(_from, dtype=np_dtype, order="C")
        ptr = arr.ctypes.data_as(ctypes.POINTER(base))
        len_ = len(arr)
        del arr
        return cls(ptr, size=len_, dtype=dtype)
    
    def __getitem__(self, index):
        return self.ptr[index]
    
    def fill(self, value):
        lib.fill_int(self.ptr, value, self.size)
        return self
    
    def tolist(self, shape):
        list_ = _to_list(self.ptr, shape)
        return view(shape, list_)

class View(tuple):
    def __new__(cls, iterable):
        return super().__new__(cls, iterable)
    


class Matrice: 
    def __init__(self, *args, data: Optional[Union[List[List[int]] | "Matrice"]]=None, dtype=float32, grad_fn: Optional[Function]=None, label=None,  requires_grad=False):
        # assert len(args) == 2, "Only Matrices (Tensors of dim=2 are supported)"
        assert (not requires_grad) or (dtype is float32), "requires_grad allowed for dtype=float32"
        if data is not None:
            if isinstance(data, list):
                assert all(isinstance(d, list) for d in data), "invalid data"
                self._data = Blob.from_list(data, dtype=dtype)
                self.shape = View(get_list_shape(data))
            elif isinstance(data, Matrice):
                self._data = data._data
                self.shape = data.shape if len(args)==0 else View(args)
            elif isinstance(data, Blob):
                self._data = data
                assert len(args) != 0, "shape is required when passing a Blob"
                self.shape = View(args)

            else: 
                raise TypeError("data must be a 2d list or a matrice instance")
        else: 
            assert len(args) > 0, "shape or data is required"
            self._data = Blob(size=math.prod(args), dtype=dtype)
            self.shape = View(args)
            
        self.label = label
        
        self.data = data, 

        self.dtype = dtype
        self.size = self._data.size

        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None

    @classmethod
    def empty(cls, *args, dtype=float32, requires_grad=False):
        blob = Blob(dtype=dtype, size=math.prod(args), zero_init=False)
        return cls(*args, data=blob, dtype=dtype, requires_grad=requires_grad)
    
    @classmethod
    def fill(cls, *args, value=0, dtype=float32, requires_grad=False):
        blob = Blob(dtype=dtype, size=math.prod(args)).fill(value)
        return cls(*args, data=blob, dtype=dtype, requires_grad=requires_grad)
    
    @classmethod
    def rand(cls, *args, value=0, dtype=float32, requires_grad=False):
        blob = Blob(dtype=dtype, size=math.prod(args))
        if dtype is float32:
            lib.rand_(blob.ptr, int(time.time()), blob.size)
        else:
            lib.rand_int(blob.ptr, int(time.time()), blob.size)
        return cls(*args, data=blob, dtype=dtype, requires_grad=requires_grad)



    def __repr__(self):
        return f"Matrice(shape=<{self.shape}> grad_fn=<{None if self.grad_fn is None else self.grad_fn.name}>)"
    
    def view(self, *args):
        assert math.prod(args) == math.prod(self.shape), f"can't view {self.shape} as {args}"
        new = Matrice(*args, data=self._data, dtype=self.dtype, requires_grad=self.requires_grad)
        return new

    def __getitem__(self, slices):
        assert len(slices) <= len(self.shape)
        
        for dim in range(len(self.shape)):
            if len(slices) > 1:
                return self[slices[dim:]]
            return self._data[slices[0:2]]
    
    def visualize(self, file_name="graph.html"):
        graph_html(self, Matrice, filename=file_name)
        print(f"Computaion Graph exported as {file_name}")

    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0], f"matmul of invalid shapes {self.shape, other.shape}"
        return Matmul(self, other)()

    def dot(self, other): return (self * other).sum(1)

    def __add__(self, other): return Add(self, other)()
    def __sub__(self, other): return Sub(self, other)()

    def tolist(self): 
        return self._data.tolist(self.shape)
        

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


def _transpose(data):
    return [[row[i] for row in data] for i in range(len(data[0]))]

def _dot(v1, v2): return sum([a * b for a, b in zip(v1, v2)])

