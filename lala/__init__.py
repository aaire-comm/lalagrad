from .tensor import Tensor
from .dtype import float32, int32, int8, int16


def tensor(data, dtype=float32, label=None, requires_grad=False):
    return Tensor(data=data, dtype=dtype, label=label, requires_grad=requires_grad)

def empty(*args, dtype=float32, label=None, requires_grad=False):
    return Tensor.empty(*args, dtype=dtype, label=label, requires_grad=requires_grad)

def zeros(*args, dtype=float32, label=None, requires_grad=False):
    return Tensor.zeros(*args, dtype=dtype, label=label, requires_grad=requires_grad)

def ones(*args, dtype=float32, label=None, requires_grad=False):
    return Tensor.ones(*args, dtype=dtype, label=label, requires_grad=requires_grad)

def rand(*args, label=None, requires_grad=False):
    #This only generates random float32 in (-1, 1) range
    return Tensor.rand(*args, dtype=float32, label=label, requires_grad=requires_grad)



def zeros_like(*args, dtype=float32, label=None, requires_grad=False):
    return Tensor.zeros_like(*args, dtype=dtype, label=label, requires_grad=requires_grad)


def fill(*args, value,  dtype=None, label=None, requires_grad=False):
    return Tensor.fill(*args, value=value, dtype=dtype, label=label, requires_grad=requires_grad)