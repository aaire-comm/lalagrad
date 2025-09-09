from .matrice import Matrice
from typing import Tuple
import numpy as np

class Parameter(Matrice):
    def __init__(self, shape: Tuple[int], label=None, grad_fn=None):
        rand = np.random.rand(*shape).tolist()
        super().__init__(rand, label=label, requires_grad=True)
    
    def step(self):
        assert self.grad is not None, "No gradiend found for parameter"
        self = self - self.grad
    
    def __call__(self, x):
        y = x @ self
        return y

    def zero_grad(self):
        self.grad = None