
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
from .dtype import Dtype
from lala._C import ffi
from lala.dtype import float32

def isTensor(obj): return isinstance(obj, Tensor)

class Tensor: 
    def __init__(self, *args, data: Optional[Union[List[List[int]] | "Tensor" | Blob]]=None, dtype=float32, grad_fn: Optional[Operation]=None, label=None,  requires_grad=False):
        assert (not requires_grad) or (dtype is float32), "requires_grad allowed for dtype=float32"
        if data is not None:
            if isinstance(data, list):
                #use numpy to conver the list to a Contiguous memory block and get a void pointer to it
                #delete the numpy object (we don't need it)
                #TODO: Implement our own list to buffer of dtyper in C
                np_dtype = np.float32 if dtype is float32 else np.int32
                arr = np.array(data, dtype=np_dtype, order="C")

                self.shape = View(arr.shape)
                nbytes = arr.size * dtype.bytes
                ptr = arr.ctypes.data

                del arr #free the memory held by numpy object (NOT the buffer just the PyObject)
                blob = Blob(ptr=ptr, nbytes=nbytes)

            #you can also just pass a Storage Blob and build a tensor on top
            #it doesn't copy so any change to this the tensor changes the storage 
            #ultimately chaging the data of any other tensor base on that passed blob
            elif isinstance(data, Blob):
                self.shape = View(args)
                blob = data
            else: 
                raise TypeError("data must be a 2d list or a matrice instance")
        else: 
            assert len(args) > 0, "shape or data is required"
            self.shape = View(args)
            blob = Blob(math.prod(args)*self.shape.numel())
            
        self.label = label
        self.storage = blob
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None

    @classmethod
    def empty(cls, *args, dtype=float32, requires_grad=False):
        blob = Blob(nbytes=math.prod(args)*dtype.bytes)
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
    def ones(cls, *args, dtype=float32, requires_grad=False):
        b = Blob(nbytes=dtype.bytes * math.prod(args), fill=1)
        return cls(data=b, *args, dtype=dtype, requires_grad=requires_grad)
    
    @classmethod
    def ones_like(cls, t: "Tensor", dtype=float32,requires_grad=False):
        blob = Blob(nbytes=t.storage.nbytes, fill=1)
        new = cls(*t.shape, data=blob, dtype=dtype, requires_grad=requires_grad)
        new.fill(1).detach()
        return new
    
    #this is just a dummy place holder to be used where we need to use a tensor
    @classmethod
    def dummy(cls, label: str="Dummy"):
        return cls(0, label=label)
    
    def detach(self):
        assert self.grad_fn is None, "trying to detach an unattached tensor"
        self.grad_fn.detach(self)
        self.grad_fn = None
        

    def clone(self):
        clone_ = Tensor.empty(*self.shape, dtype=self.dtype, requires_grad=self.requires_grad)
        self.storage._clone(clone_.storage)
        return clone_

    def __repr__(self):
        return f"Tensor(shape=<{self.shape}> grad_fn=<{None if self.grad_fn is None else self.grad_fn.name}>)"
    
    def view(self, *args): 
        print(args)
        return ViewOp(self, View(args))()
    
    def to_(self, dtype: Dtype):
        assert self.storage.nbytes / dtype.bytes >= self.numel(), "the dtype provided won't feet in the storage"
        self.dtype = dtype

        
    def to(self, dtype: Dtype):
        new  = self.clone()
        new.to_(dtype)
        return new
    
    def clone(self):
        new_b = Blob(self.storage.nbytes)
        return Tensor(*self.shape, data=new_b, dtype=self.dtype, label=str(self.label)+"-copy", requires_grad=self.requires_grad)

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
    def __mul__(self, other): return Mul(self, other)()

    def sum(self, dim=None): return Sum(self, dim)()
    def mean(self, dim=None): return Mean(self, dim)()
    def smul(self, scalar): return ScalarMul(self, scalar)()

    def get_item(self):
        assert len(self.shape) == 1 and self.shape[0] == 1, "get_item only defined for scalar "
        return self.storage._get_pointer(self.dtype.ptr_t)[0]

    def tolist(self): 
        if len(self.shape):
            return self.storage._to_python_list(self.shape, self.dtype.ptr_t)
        return self.get_item()

    def numel(self): return self.shape.numel()

    def backward(self, upstream_m=None):
        assert upstream_m is not None or self.grad_fn.op_type == "SReduce", "implicit backward only defined for single elemnt matrices"
        assert self.requires_grad, "matrice doesn't requre_grad"
        if not self.grad_fn:
            return 
        self.grad_fn.backward(upstream_m)


