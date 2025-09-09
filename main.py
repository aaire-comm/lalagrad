from typing import Tuple, Callable
from lala.lala import Matrice
from lala.utils import export_html
import numpy as np
        


class Parameter(Matrice):
    def __init__(self, shape: Tuple[int], label=None, grad_fn=None):
        rand = np.random.rand(*shape).tolist()
        super().__init__(rand, label=label, requires_grad=True)
    
    def step(self):
        assert self.grad is not None, "No gradiend found for parameter"
        print(self.grad.data)
        self = self - self.grad
    
    def __call__(self, x):
        y = x @ self
        return y

    def zero_grad(self):
        self.grad = None

class Model:
    def __init__(self, loss_fn: Callable):
        self.l1 = Parameter((3, 2), label="Layer1")
        self.l2 = Parameter((2, 1), label="Layer2")
        self.l3 = Parameter((1, 1), label="Layer3")
        self.l4 = Parameter((1, 1), label="Layer4")
        self.loss_fn = loss_fn

    def __call__(self, x):
        y = self.l1(x)
        y = self.l2(y)
        y = self.l3(y)
        y = self.l4(y)
        return y
    
    def batch_train(self, data):
        for inputs, target in data:
            logits = self(inputs)
            loss = self.loss_fn(logits, target)
            loss.backward()
        self.step()
        return loss


    
    def step(self):
        self.l1.step()
        self.l2.step()

        self.l1.zero_grad()
        self.l2.zero_grad()


def loss_fn(logits: Matrice, target: Matrice):
    return (logits - target).spow(2).mean()

m = Model(loss_fn=loss_fn)
b = m.batch_train(
    [
    (
    Matrice([[1, 2,  3]], label="Input1"),
    Matrice([[0]], label="Target1")
    )
    ]
)
export_html(b)