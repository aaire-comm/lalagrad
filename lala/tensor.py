
"""
Operation types:
    Unary -> take a Tensor return Tensor of the same shape
    Binary -> take a Tensor and return a Tensor
    SReduce (Scalar reduce) take a Tensor and return single element Tensor (implicite autograd allowed)
"""
from typing import List, Optional, Union
import math
from .utils import graph_html
from .ops import *
from .blob import Blob
import numpy as np
from .dtype import Dtype
from lala._C import ffi
from lala.dtype import float32

def isTensor(obj): return isinstance(obj, Tensor)

class Tensor: 
    def __init__(self, *args, data: Optional[Union[List[List[int]] | "Tensor" | Blob]]=None, dtype=float32, grad_fn: Optional[Operation]=None, label=None,  src: Optional[Operation]=None, requires_grad=False):
        assert (not requires_grad) or (dtype is float32), "requires_grad allowed for dtype=float32"
        assert src is None or isinstance(src, Operation), "A Tensor src can only be an Op or None "
        self.src = src
        if data is not None:
            if isinstance(data, list):
                #use numpy to conver the list to a Contiguous memory block and get a void pointer to it
                #delete the numpy object (we don't need it)
                #TODO: Implement our own list to buffer of dtyper in C
                np_dtype = np.float32 if dtype is float32 else np.int32
                arr = np.array(data, dtype=np_dtype, order="C")
                self.shape = arr.shape
                nbytes = arr.size * dtype.bytes
                ptr = arr.ctypes.data

                del arr #free the memory held by numpy object (NOT the buffer just the PyObject)
                blob = Blob(ptr=ptr, nbytes=nbytes)

            #you can also just pass a Storage Blob and build a tensor on top
            #it doesn't copy so any change to this the tensor changes the storage 
            #ultimately chaging the data of any other tensor base on that passed blob
            elif isinstance(data, Blob):
                self.shape = args
                blob = data
            elif isinstance(data, int):
                self.shape = ()
                dtype = int32
                blob = Blob(nbytes=4)
                blob._get_pointer("int*")[0] = data
            elif isinstance(data, float):
                self.shape = ()
                dtype = int32
                blob = Blob(nbytes=4)
                blob._get_pointer("float*")[0] = data

            else: 
                raise TypeError("data must be a 2d list or a matrice instance")
        else: 
            assert len(args) > 0, "shape or data is required"
            self.shape = args
            blob = Blob(math.prod(args)*self.numel())
            
        self.label = label
        self.dims = len(self.shape)
        self.storage = blob
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None
        
        if self.dim:
            s = [1]
            r = tuple(reversed(self.shape))
            for i in range(self.dims - 1):
                s.append(s[i] * r[i])
                self.strides = tuple(reversed(s))
        else:
            self.strides = ()

    @property
    def T(self):
        return self.transpose()

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
    
    def is_scalar(self):
        return len(self.shape) == 0
    
    def view(self, *args): 
        if len(args):
            return ViewOp(self, View(args))()
        return ViewOp(self, self.shape)()  
        
    
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
        graph_html(self, filename=file_name)
        print(f"Computaion Graph exported as {file_name}")

    

    def dot(self, other): return (self * other).sum(1)

    def __add__(self, other): return Add(self, other)()
    def __sub__(self, other): return Sub(self, other)()
    def __mul__(self, other): return Mul(self, other)()
    def __matmul__(self, other): return Matmul(self, other)()

    def sum(self, dim=None): return Sum(self, dim)()
    def mean(self, dim=None): return Mean(self, dim)()
    def smul(self, scalar): return ScalarMul(self, scalar)()

    def transpose(self, dim0, dim1): return Transpose(self, dim0, dim1)()
    def broadcast_to(self, *args): 
        assert len(args) > len(self.shape), "broadcast shape must be greater than the tensor's shape"
        rev = reversed(args)
        res = self.clone()

        
    
    def spow(self, exp): return ScalarPower(self, exp)()

    def get_item(self):
        assert not len(self.shape), "get_item only defined for scalar "
        return self.storage._get_pointer(self.dtype.ptr_t)[0]

    def tolist(self): 
        if len(self.shape):
            return self.storage._to_python_list(self.shape, self.dtype.ptr_t)
        return self.get_item()

    def numel(self): return math.prod(self.shape)
    def dim(self): return self.dims
    def stride(self): return self.strides

    def backward(self, upstream_m: Optional["Tensor"]=None):
        assert self.requires_grad, "matrice doesn't requre_grad"
            
        if self.src:
            #TODO: Better to check if tensor is Scalar than relay on op_type

            if not upstream_m and self.src.op_type != "SReduce":
                raise RuntimeError("implicit backward only defined for single elemnt matrices")
            self.src.backward(upstream_m)
        else:
            assert upstream_m, "implicit backward only defined for single elemnt matrices"
            assert self.shape == upstream_m.shape, "invalid shape upstream tensor"
            self.grad = upstream_m


