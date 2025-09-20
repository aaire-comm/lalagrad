from .utils import *
from .dtype import int32, Null, Dtype

from .blob import Blob
from ._C import ops
import math

def _transpose(data):
    return [[row[i] for row in data] for i in range(len(data[0]))]

def _dot(v1, v2): return sum([a * b for a, b in zip(v1, v2)])


class Operation:
    def __init__(self, name, type_, *args):
        from .tensor import  Tensor
        self.name = name
        self.op_type = type_
        self.operands = args
        for operand in args:
            if isinstance(operand, Tensor):
                operand.grad_fn = self
                
        #we need to know the operation dtype
        self.op_dtype = max(operand.dtype if isinstance(operand, Tensor) else Null for operand in self.operands)
        #cast all tensors to the higher dtype
        for operand in self.operands:
            if isinstance(operand, Tensor) and operand.dtype is not self.op_dtype:
                operand = operand.to(self.op_dtype)
                
        self.requires_grad =  any(operand.requires_grad if isinstance(operand, Tensor) else False for operand in self.operands)
        


    @classmethod
    def elem_wise_validate(cls, fn):
        shape = fn.operands[0].shape
        return all(shape == op.shape for op in fn.operands), f"Incompatable shape for Operation {fn.name}"


    def validate(self):
        return True, ""
    
    def detach(self, operand):
        from .tensor import Tensor
        assert operand in self.operands, f"operand not attached to {self}"
        self.operands = (t if t is not operand else Tensor.dummy() for t in self.operands)

    def __call__(self): 
        from lala.tensor import Tensor
        assertion, msg = self.validate()
        assert assertion, msg

        data, shape = self.forward()

        #Don't include in the Comp graph unless it requires gradien
        if self.requires_grad:
            src = self
        else:
            src = None
        return Tensor(*shape, data=data, src=src, dtype=self.op_dtype, requires_grad=self.requires_grad)
    

    def backward(self, upstream_m):
        from lala.tensor import Tensor
        for operand in self.operands:
            if isinstance(operand, Tensor) and operand.requires_grad:
                if self.op_type == "Binary":
                    grad_b =   self.gradient(operand)
                    print(grad_b)
                elif self.op_type == "View":
                    grad_b =   self.gradient(upstream_m)
                elif self.op_type == "Unary" or self.op_type == "SReduce":
                    grad_b =   self.gradient()
                else:
                    grad_b =   self.gradient(operand, upstream_m)

                grad = Tensor(*operand.shape, data=grad_b)
                print(grad)

                # if upstream_m is not None and self.op_type != "View" and self.name != "MMul":
                #     grad *= upstream_m

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
        m, _ = self.operands
        res = Blob(m.dtype.bytes)
        ops[self.op_dtype.name].mean_t(m.storage, res)
        return res, ()
    
    def gradient(self):
        m = self.operands[0]
        grad_b = Blob(nbytes=m.storage.nbytes, fill=1/m.numel())
        return grad_b
        
        
class Transpose(Operation):
    def __init__(self, *args): 
        super().__init__("Transpose", "View", *args)

    def forward(self): 
        m, dims = self.operands
        new_shape = m.shape
        a = new_shape[dims[0]]
        new_shape[dims[0]] = new_shape[dims[1]]
        new_shape[dims[1]] = a
        return m.storage, new_shape
    
    def gradient(self, upstream_m):
        dims = self.operands[1]
        return upstream_m.transpose(dims[1], dims[0])
        
    
class Add(Operation):
    def __init__(self, *args):
        super().__init__("Add", "Binary", *args)
    
    def validate(self):
        a, b = self.operands
        return a.shape == b.shape, f"Blobs for different shape for {self.name}"
    
    def forward(self):
        rhs, lhs = self.operands
        res = Blob(nbytes=rhs.storage.nbytes)
        ops[self.op_dtype.name].add_t(rhs.storage, lhs.storage, res)
        return res, rhs.shape

    def gradient(self, w_r_t): 
        assert w_r_t in self.operands, "w_r_t is not an operand of this grad_fn"
        grad_b = Blob(nbytes=w_r_t.storage.nbytes, fill=1)
        return  grad_b
    

class Sub(Operation):
    def __init__(self, *args):
        super().__init__("Sub", "Binary", *args)
    
    def validate(self): 
        return Operation.elem_wise_validate(self)
        
    def forward(self):
        rhs, lhs = self.operands
        res = Blob(nbytes=rhs.storage.nbytes)
        ops[self.op_dtype.name].sub_t(rhs.storage, lhs.storage, res)
        return res, rhs.shape
    

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
            rhs, _ = self.operands
            res = Blob(nbytes=m.dtype.bytes)
            ops[self.op_dtype.name].sum_t(rhs.storage, res)
            return res, ()

            #TODO: Implement sum allong dim with strides
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

    def forward(self): 
        rhs, lhs = self.operands
        res = Blob(nbytes=rhs.storage.nbytes)
        ops[self.op_dtype.name].mul_t(rhs.storage, lhs.storage, res)
        return res, rhs.shape
    
    def gradient(self, w_r_t): 
        #Here we can save mem by just sending a ref to the other tensor its self
        lhs, rhs = self.operands
        other_b = rhs.storage if lhs is w_r_t else lhs.storage
        grad_b = Blob(nbytes=other_b.nbytes)
        other_b._copy(grad_b)
        return grad_b


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
        res_b = Blob(m.storage.nbytes)
        return res_b
        
    def gradient(self): 
        w_r_t, scalar = self.operands
        grad_b = Blob(nbytes=w_r_t.storage.nbytes, fill=scalar)
        return grad_b

class Relu(Operation):
    def __init__(self, *args): super().__init__("Relu", "Unary", *args)

    def forward(self):
        m = self.operands[0]
        res = Blob.empty(*m.shape, dtype=int32)
        
        return res, 
    
    def gradient(self):
        m = self.operands[0]
        return [[1 if e > 0 else 0 for e in row] for row in m.data]

    
        
class Matmul(Operation):
    _matmul = lambda m1, m2: [[_dot(row, col) for col in _transpose(m2)] for row in m1 ]
    
    def __init__(self, *args):
        super().__init__("MMul", "MBinary", *args)
        
    def forward(self):
        t0, t1 = self.operands
        dim0, dim1 = t0.dim(), t1.dim()

        #handle broadcasting to th right dimention
        brdcst_shape0 = t0.shape
        brdcst_shape1 = t0.shape
        if not dim0 :
            brdcst_shape0 = (1, 1)
        if not dim1:
            brdcst_shape1 = (1, 1)
        
        if dim0 == 1:
            brdcst_shape0 = (1, brdcst_shape0)
        if dim1 == 1:
            brdcst_shape1 = (1, brdcst_shape1)


        res_shape = (1, 1)
        res = Blob(nbytes=math.prod(res_shape)*self.op_dtype.bytes,  zero_init=True)
        ops[self.op_dtype.name].matmul(t0.storage, t1.storage, t0.stride(), t1.stride() )
        return res, res_shape


    def gradient(self, w_r_t, upstream_m): 
        lhs, rhs = self.operands
        
        if w_r_t is lhs:
            lgrad = Blob(Matmul._matmul(upstream_m.data, rhs.transpose().data))
            return lgrad.data
        else:
            rgrad = Blob(Matmul._matmul(lhs.transpose().data, upstream_m.data))
            return rgrad.data
        
class ViewOp(Operation):
    def __init__(self, *args):
        super().__init__("View", "View", *args)
    
    def forward(self):
        t, shape = self.operands
        assert t.shape == shape , f"Invalid input size"
        return t.storage, shape

    def gradient(self):
        return 

class ViewOp(Operation):
    def __init__(self, *args):
        super().__init__("Broadcast", "View", *args)
    
    def forward(self):
        t, shape = self.operands
        assert t.shape == shape , f"Invalid input size"
        return t.storage, shape

    def gradient(self):
        return 



#this does casting to a dtype
#for now that is just to float32
#doesn't support backward
class CastOp:
    @staticmethod
    def forward(lhs, res, dtype: Dtype):
        ops[dtype.name].cast(lhs.storage, res.storage, lhs.numel())



