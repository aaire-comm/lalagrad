
"""
Operation types:
    Unary -> take a Tensor return Tensor of the same shape
    Binary -> take two Tensors and return a Tensor
    SReduce (Scalar reduce) take a Tensor and return single element Tensor (implicit autograd allowed)
"""

from .utils import *
from .dtype import int32, Null, Dtype
from typing import Tuple

from .blob import Blob
from ._C import ops
import math

#TODO: Break operations into multile

class Operation:
    def __init__(self, name, type_, *args):
        from .tensor import  Tensor
        self.name = name
        self.op_type = type_
        self.op_dtype = max(operand.dtype if isinstance(operand, Tensor) else Null for operand in args)
        
        #cast every operand to the highes level operand
        casted_operands = []
        for operand in args:
            casted_v = operand.to(self.op_dtype) if isinstance(operand, Tensor) and operand.dtype is not self.op_dtype else operand
            casted_operands.append(casted_v)

        self.operands = tuple(casted_operands)
        self.requires_grad =  any(operand.requires_grad if isinstance(operand, Tensor) else False for operand in self.operands)
        



    
    def detach(self, operand):
        assert operand in self.operands, f"operand not attached to {self}"
        operands = list(self.operands)
        operands.remove(operand)
        self.operands =tuple(operands)

    def __call__(self): 
        from lala.tensor import Tensor

        data, shape, strides = self.forward()

        #Don't include in the Comp graph unless it requires gradien
        if self.requires_grad:
            src = self
        else:
            src = None
        return Tensor(*shape, data=data, strides=strides, src=src, dtype=self.op_dtype, requires_grad=self.requires_grad)
    

    def backward(self, upstream_m):
        from lala.tensor import Tensor
        for operand in self.operands:
            if isinstance(operand, Tensor) and operand.requires_grad:
                if self.op_type == "BinaryOp":
                    grad_b =   self.gradient(operand)
                elif self.op_type == "ViewOp":
                    grad_b =   self.gradient(upstream_m)
                elif self.op_type == "UnOp" or self.op_type == "ReduceOp":
                    grad_b =   self.gradient()
                else:
                    grad_b =   self.gradient(operand, upstream_m)

                grad = Tensor(*operand.shape, data=grad_b)
                #Remember VewOps don't participate in the chan rule. They just do a reverse op on the upstream and pass it up
                if upstream_m is not None and self.op_type != "ViewOp" and self.name != "MMul":
                    grad *= upstream_m

                #Set the grad or accumulate if the tensor already has a grad 
                if operand.grad is None:
                    operand.grad = grad
                else: 
                    operand.grad += grad

                #continue up the autodiffer chain
                operand.backward(operand.grad)


"""
We break down ops into these 5 ops so we can optimize the forward, broadcasting and backward optimized accordingly ...

"""

class UnOps(Operation): 
    """
    These are operantions on a tensor that take a tensor and do the same op 
    (function) on every element
    ex. smul. spow, sadd, ssub
    like ops with other types in python int, bool. float
    """
    def __init__(self, name, lhs, other): 
        super().__init__(name, "UnOp", lhs, other)


class BinaryOps(Operation): 
    """
    These are Ops that take two tensors and return a tensor of the same shape
    ex. +, -, ** and most other ops

    """
    def __init__(self, name, *args): 
        super().__init__(name, "BinaryOp", *args)


class ReduceOps(Operation):
    def __init__(self, name, *args): super().__init__(name, "ReduceOp", *args)

    """
    There are ops that return a tensor of smaller shape
    ex. mean(<dim>), sum(<dim>), along some dim
    """

class ViewOps(Operation): 
    """
    These are just different ways of looking at the storage(memory)
    ex. transpose, view, expand...
    The gradient of ViewOps is just the reverse if the changes
    """, 
    def __init__(self, name, t, new_shape): super().__init__(name, "ViewOp", t, new_shape)


#=======================================here are all the ops supported by  lalagrad=============================


class Mean(ReduceOps):
    def __init__(self, tensor, dim): 
        super().__init__("MEAN", tensor, dim)

    def forward(self):
        m, dim = self.operands
        res = Blob(m.dtype.bytes)
        #TODO: Make the mean kernel take a dimension
        ops[self.op_dtype.name].mean_t(m.storage, res)
        return res, (), None
    
    def gradient(self):
        m = self.operands[0]
        grad_b = Blob(nbytes=m.storage.nbytes, fill=1/m.numel())
        return grad_b
        
   

class ScalarPower(UnOps):
    def __init__(self, *args): 
        super().__init__("ElPow", *args)

    def forward(self): 
        lhs, exp = self.operands
        res = Blob(nbytes=lhs.storage.nbytes)
        ops[self.op_dtype.name].exp(lhs.storage, exp, res)
        return res, lhs.shape, None
        
    def gradient(self): 
        lhs, exp = self.operands
        grad_b = Blob(nbytes=lhs.storage.nbytes)
        ops[self.op_dtype.name].mul_s(lhs.storage, exp, grad_b)
        return grad_b

    

class ScalarMul(UnOps):
    def __init__(self, *args): 
        super().__init__("SMul", *args)

    def forward(self): 
        lhs, s = self.operands
        res_b = Blob(lhs.storage.nbytes)
        ops[self.op_dtype.name].mul_s(lhs.storage, s,  res_b)

        return res_b, lhs.shape, None
        
    def gradient(self): 
        w_r_t, scalar = self.operands
        grad_b = Blob(nbytes=w_r_t.storage.nbytes, fill=scalar)
        return grad_b

    
class Add(BinaryOps):
    def __init__(self, lhs, rhs):
        super().__init__("Add", lhs, rhs)
    
    def forward(self):
        rhs, lhs = self.operands
        res = Blob(nbytes=rhs.storage.nbytes)
        ops[self.op_dtype.name].add_t(rhs.storage, lhs.storage, res)
        return res, rhs.shape, None

    def gradient(self, w_r_t): 
        assert w_r_t in self.operands, "w_r_t is not an operand of this grad_fn"
        grad_b = Blob(nbytes=w_r_t.storage.nbytes, fill=1.0)
        return  grad_b
    



class Sub(BinaryOps):
    def __init__(self, *args):
        super().__init__("Sub", *args)
    
    def forward(self):
        rhs, lhs = self.operands
        res = Blob(nbytes=rhs.storage.nbytes)
        ops[self.op_dtype.name].sub_t(rhs.storage, lhs.storage, res)
        return res, rhs.shape, None
    

    def gradient(self, w_r_t): 
        assert w_r_t in self.operands, "w_r_t is not an operand of this grad_fn"
        lhs, _ = self.operands
        if w_r_t is lhs:
            fill_v = 1.0
        else:
            fill_v = -1.0
        return Blob(lhs.storage._get_size(), fill=fill_v)
            


class Sum(ReduceOps):
    """
    Sum elements of a tensor along a dim
    The gradient is just 1s of the same shape
    """
    def __init__(self, tensor, dim=None): 
        super().__init__("Sum", tensor, dim)
        

    def forward(self, dim=None): 
        m, dim = self.operands
        #TODO: remove the condition and write a generic kernel that handles this
        if dim is None:
            rhs, _ = self.operands
            res = Blob(nbytes=m.dtype.bytes)
            ops[self.op_dtype.name].sum_t(rhs.storage, res)
            return res, (), None
            #TODO: Implement sum allong dim with strides
        else: 
            raise NotImplementedError("Sum along a dim not Implemented yet")
    
    def gradient(self):
        lhs, dim = self.operands
        grad_b = Blob(lhs.storage._get_size(), fill=1.0)
        return grad_b

        

class Mul(BinaryOps):
    """
    Does element-wise multiplication
    backward with respect to one of the operands is the other
    """
    def __init__(self, *args): super().__init__("ElMul", *args)

    def forward(self): 
        rhs, lhs = self.operands
        res = Blob(nbytes=rhs.storage.nbytes)
        ops[self.op_dtype.name].mul_t(rhs.storage, lhs.storage, res)
        return res, rhs.shape, None
    
    def gradient(self, w_r_t): 
        lhs, rhs = self.operands
        other_b = rhs.storage if lhs is w_r_t else lhs.storage
        grad_b = Blob(nbytes=other_b.nbytes)
        other_b._copy(grad_b)
        return grad_b



class Relu(Operation):
    """
    Does the relu activation on on each element of a tensor
    the gradient of relu is just relu
    """
    def __init__(self, *args): super().__init__("Relu", *args)

    def forward(self):
        m = self.operands[0]
        res = Blob.empty(*m.shape, dtype=int32)
        
        return res, m.shape, None 
    
    def gradient(self):
        m = self.operands[0]
        return [[1 if e > 0 else 0 for e in row] for row in m.data]

        
class Transpose(ViewOps):
    """exchanges the elements of two dims"""
    def __init__(self, lhs, dim0: int, dim1: int): 
        super().__init__("Transpose", lhs, (dim0, dim1))

    def forward(self): 
        m, dims = self.operands
        new_shape = list(m.shape)
        a = new_shape[dims[0]]
        new_shape[dims[0]] = new_shape[dims[1]]
        new_shape[dims[1]] = a
        new_stride = list(m.stride())
        a = new_stride[dims[0]]
        new_stride[dims[0]] = new_stride[dims[1]]
        new_stride[dims[1]] = a
        return m.storage, tuple(new_shape), tuple(new_stride)
    
    def gradient(self, upstream_m):
        dims = self.operands[1]
        return upstream_m.transpose(dims[1], dims[0])
     
        
class View(ViewOps):
    """
    This is a litteral different view of a tensor. 
    No elemt postion swap is done just just reinterpret the buffer
    """
    def __init__(self, lhs, new_shape):
        super().__init__("View",  lhs, new_shape)
    
    def forward(self):
        t, new_shape = self.operands
        return t.storage, new_shape, None

    def gradient(self):
        lhs = self.operands
        grad_b = Blob(lhs.storage.nbytes)
        grad_b._copy(lhs.storage)
        return grad_b


class BroadCast(ViewOps):
    def __init__(self, lhs, new_shape: Tuple[int]):
        super().__init__("Broadcast", lhs, new_shape)

    def forward(self):
        lhs, shape = self.operands
        new_dims = len(shape) - len(lhs.shape)
        assert new_dims >= 0, "can't broadcast to a smaller dim shape"
        assert shape != lhs.shape, "can't broadcast to old shape"
        assert all(shape[new_dims+i] == lhs.shape[i] or lhs.shape[i]==1  for i in range(lhs.dim())), "new shape can ony replace size 1 dims from existing dims"
        new_shape, new_stride = shape, tuple(0 for _ in range(new_dims)) + lhs.stride()
        return lhs.storage, new_shape, new_stride


#this does casting to a dtype
#for now that is just to float32
#doesn't support backward
class CastOp:
    @staticmethod
    def forward(lhs, res, dtype: Dtype):
        ops[dtype.name].cast(lhs.storage, res.storage, lhs.numel())


        
class Matmul(Operation):
    """
    Matmul doesn't fit in the any of the above ops as  my may return tensors of different and
    also the gradient is different from other Binary ops

    This is also the only op that returns a pointer for the result tensor in case of 
    batched matmul as the mem size and allocation are handled in the kernel
    """
    
    def __init__(self, *args):
        super().__init__("MMul", "MBinary", *args)
        
    def forward(self):
        lhs, rhs = self.operands
        #this dims are along which we do matmuls 
        #common_dims is > 0 for batch matmul and 0 for single matmul
        rhs_rows, rhs_cols = rhs.shape[-2:]
        lhs_rows, lhs_cols = lhs.shape[-2:]
        assert lhs_cols == rhs_rows
        common_dims = lhs.dims - 2 
        
        res_shape = (lhs_rows, rhs_cols)
        res_strides = None
        if common_dims:
            common_shapes = lhs.shape[:common_dims]
            assert common_shapes == rhs.shape[:common_dims]
            res_shape = common_shapes + res_shape
            rhs_strides = rhs.stride()
            lhs_strides = lhs.stride()

            res_strides = tuple(max(s0, s1) for s0, s1 in zip(lhs_strides[:common_dims], rhs_strides[:common_dims])) + (rhs_cols, 1)
            p = ops[self.op_dtype.name].batch_matmul(lhs.storage, rhs.storage, common_shapes, lhs_strides, rhs_strides, res_strides, common_dims, lhs_rows, lhs_cols, rhs_cols)
            res = Blob(ptr=p, nbytes=math.prod(res_shape)*self.op_dtype.bytes)
        else:
            res = Blob(nbytes=math.prod(res_shape)*self.op_dtype.bytes,  zero_init=True)
            ops[self.op_dtype.name].matmul(lhs.storage, rhs.storage, res, lhs.shape[-2], lhs.shape[-1], rhs.shape[-1])
        return res, res_shape, res_strides


    def gradient(self, w_r_t, upstream_m): 
        lhs, rhs = self.operands

        if w_r_t is lhs:
            rhs_t = rhs.transpose(0, 1)
            lgrad_b = Blob(upstream_m.shape[0]*rhs_t.shape[1]*self.op_dtype.bytes)
            ops[self.op_dtype.name].matmul(upstream_m.storage, rhs_t.storage, lgrad_b, *upstream_m.shape, rhs_t.shape[1])

            return lgrad_b
        else:

            lhs_t = lhs.transpose(0, 1)
            rgrad_b = Blob(lhs_t.shape[0]*upstream_m.shape[1]*self.op_dtype.bytes)
            ops[self.op_dtype.name].matmul(lhs_t.storage, upstream_m.storage,  rgrad_b, *lhs_t.shape, upstream_m.shape[1])

            return rgrad_b
