
"""
Operation types:
    Unary -> take a Tensor return Tensor of the same shape
    Binary -> take a Tensor and return a Tensor
    SReduce (Scalar reduce) take a Tensor and return single element Tensor (implicite autograd allowed)
"""
from typing import List, Optional, Union
import math
from .utils import graph_html, get_list_shape
from .ops import *
from .blob import Blob
from lala.view import View
import numpy as np
from typing import Any
from .dtype import Dtype
import ctypes

from lala.dtype import float32

def isTensor(obj): return isinstance(obj, Tensor)

class Tensor: 
    def __init__(self, *args, data: Optional[Union[List[List[int]] | "Tensor" | Blob]]=None, dtype=float32, grad_fn: Optional[Operation]=None, label=None,  requires_grad=False):
        _ptr = None
        assert (not requires_grad) or (dtype is float32), "requires_grad allowed for dtype=float32"
        if data is not None:
            if isinstance(data, list):
                #use numpy to conver the list to a Contiguous memory block and get a void pointer to it
                #delete the numpy object (we don't need it)
                #TODO: Implement our own list to buffer of dtyper in C
                assert all(isinstance(d, list) for d in data), "invalid data"
                np_dtype = np.float32 if dtype is float32 else np.int32
                arr = np.array(data, dtype=np_dtype, order="C")
                nbytes = arr.size * dtype.bytes
                ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_void))
                del arr     #delte the numpy object (after taking getting the pointer)
                blob = Blob(ptr=ptr, nbytes=nbytes)
                self.shape = View(get_list_shape(data))

            #you can also just pass a Storage Blob and build a tensor on top
            #it doesn't copy so any change to this the tensor changes the storage 
            #ultimately chaging the data of any other tensor base on that passed blob
            elif isinstance(data, Blob):
                assert len(args) != 0, "shape is required when passing a Blob"
                self.shape = View(args)
                blob = data
            else: 
                raise TypeError("data must be a 2d list or a matrice instance")
        else: 
            assert len(args) > 0, "shape or data is required"
            self.shape = View(args)
            blob = Blob(nbytes==math.prod(args)*self.shape.numel())
            
        self.label = label
        self.storage = blob
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None

    @classmethod
    def empty(cls, *args, dtype=float32, requires_grad=False):
        blob = Blob(nbytes=math.prod(args))
        return cls(*args, data=blob, dtype=dtype, requires_grad=requires_grad)
    
    @classmethod
    def fill(cls, *args, value, dtype=float32, requires_grad=False):
        blob = Blob(dtype=dtype, nbytes=math.prod(args))
        return cls(*args, data=blob, dtype=dtype, requires_grad=requires_grad)
    
    @classmethod
    def rand(cls, *args, dtype=float32, requires_grad=False):
        assert len(args), "shape args required"
        b = Blob(nbytes=math.prod(args)*dtype.bytes)
        return cls(*args, data=b, dtype=dtype, requires_grad=requires_grad)
    
    @classmethod
    def zeros(cls, *args, dtype=float32, requires_grad=False):
        b = Blob(nbytes=dtype.bytes * math.prod(args), zero_init=True)
        return cls(data=b, *args, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def zeros_like(cls, t: "Tensor", dtype=float32,requires_grad=False):
        b = Blob(nbytes=t.storage.nbytes, zero_init=True)
        return cls(data=b, *t.shape, dtype=dtype, requires_grad=requires_grad)
    
    @classmethod
    def ones_like(cls, t: "Tensor", dtype=float32,requires_grad=False):
        blob = Blob(nbytes=t.storage.nbytes, fill=1)
        return cls(*t.shape, data=blob, dtype=dtype, requires_grad=requires_grad)
    

    def clone(self):
        clone_ = Tensor.empty(*self.shape, dtype=self.dtype, requires_grad=self.requires_grad)
        self.storage._clone(clone_.storage)
        return clone_

    def _ptr(self): return self.__ptr

    def __repr__(self):
        return f"Tensor(shape=<{self.shape}> grad_fn=<{None if self.grad_fn is None else self.grad_fn.name}>)"
    
    def view(self, *args):
        assert math.prod(args) == math.prod(self.shape), f"can't view {self.shape} as {args}"
        new = Tensor(*args, data=self._data, dtype=self.dtype, requires_grad=self.requires_grad)
        return new

    def to(self, dtype: Dtype):
        assert self.numel()

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
        return self.storage._to_python_list(self.shape)

    def numel(self): return self.shape.numel()

    def backward(self, upstream_m=None):
        assert upstream_m is not None or self.grad_fn.op_type == "SReduce", "implicit backward only defined for single elemnt matrices"
        assert self.requires_grad, "matrice doesn't requre_grad"
        if not self.grad_fn:
            return 
        self.grad_fn.backward(upstream_m)


