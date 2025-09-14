
"""
Function types:
    Unary -> take a Tensor return Tensor of the same shape
    Binary -> take a Tensor and return a Tensor
    SReduce (Scalar reduce) take a Tensor and return single element Tensor (implicite autograd allowed)
"""
import ctypes
from typing import List, Optional, Union, Tuple, Any
import math
from .utils import graph_html, get_list_shape
from .c.lib_loader import lib, libc
from .ops import *
from .blob import Blob
from lala.view import View

from lala.dtype import float32
import time
        


class Tensor: 
    def __init__(self, *args, data: Optional[Union[List[List[int]] | "Tensor"]]=None, dtype=float32, grad_fn: Optional[Function]=None, label=None,  requires_grad=False):
        assert (not requires_grad) or (dtype is float32), "requires_grad allowed for dtype=float32"
        if data is not None:
            if isinstance(data, list):
                assert all(isinstance(d, list) for d in data), "invalid data"
                self._data = Blob.from_list(data, dtype=dtype)
                self.shape = View(get_list_shape(data))
            elif isinstance(data, Tensor):
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
            self._data = Blob(nbytes=math.prod(args), dtype=dtype)
            self.shape = View(args)
            
        self.label = label
        
        self.data = data, 

        self.dtype = dtype
        self.nbytes = self._data.nbytes

        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None

    @classmethod
    def empty(cls, *args, dtype=float32, requires_grad=False):
        blob = Blob(dtype=dtype, nbytes=math.prod(args), zero_init=False)
        return cls(*args, data=blob, dtype=dtype, requires_grad=requires_grad)
    
    @classmethod
    def fill(cls, *args, value=0, dtype=float32, requires_grad=False):
        blob = Blob(dtype=dtype, nbytes=math.prod(args)).fill(value)
        return cls(*args, data=blob, dtype=dtype, requires_grad=requires_grad)
    
    @classmethod
    def rand(cls, *args, value=0, dtype=float32, requires_grad=False):
        blob = Blob(dtype=dtype, nbytes=math.prod(args))
        if dtype is float32:
            lib.rand_(blob.ptr, int(time.time()), blob.nbytes)
        else:
            lib.rand_int(blob.ptr, int(time.time()), blob.nbytes)
        return cls(*args, data=blob, dtype=dtype, requires_grad=requires_grad)



    def __repr__(self):
        return f"Tensor(shape=<{self.shape}> grad_fn=<{None if self.grad_fn is None else self.grad_fn.name}>)"
    
    def view(self, *args):
        assert math.prod(args) == math.prod(self.shape), f"can't view {self.shape} as {args}"
        new = Tensor(*args, data=self._data, dtype=self.dtype, requires_grad=self.requires_grad)
        return new

    def __getitem__(self, slices):
        assert len(slices) <= len(self.shape)
        
        for dim in range(len(self.shape)):
            if len(slices) > 1:
                return self[slices[dim:]]
            return self._data[slices[0:2]]
    
    def visualize(self, file_name="graph.html"):
        graph_html(self, Tensor, filename=file_name)
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


