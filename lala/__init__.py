from .tensor import Tensor
from .dtype import float32, int32, int8, int16


def empty(*args, dtype=float32, requires_grad=False):
    return Tensor.empty(*args, dtype=dtype, requires_grad=requires_grad)

def zeros(*args, dtype=float32, requires_grad=False):
    return Tensor.zeros(*args, dtype=dtype, requires_grad=requires_grad)

def ones(*args, dtype=float32, requires_grad=False):
    return Tensor.ones(*args, dtype=dtype, requires_grad=requires_grad)


def zeros_like(*args, dtype=float32, requires_grad=False):
    return Tensor.zeros_like(*args, dtype=dtype, requires_grad=requires_grad)

