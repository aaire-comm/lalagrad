from .tensor import Matrice
from typing import Tuple
import numpy as np

class Parameter(Matrice):
    def __init__(self, shape: Tuple[int], label=None, grad_fn=None):
        self.w = Matrice(np.random.rand(*shape).tolist(), label=label, requires_grad=True)
        self.label = label

    
    def step(self):
        assert self.w.grad is not None, "No gradiend found for parameter"
        grad_fn = self.w.grad_fn
        self.w = self.w - self.w.grad
        self.w.grad_fn = grad_fn

    
    def __call__(self, x):
        y = x @ self.w
        return y
    

    def zero_grad(self):
        self.w.grad = None




