from typing import Tuple, Callable
from lala.matrice import Matrice
from lala.nn import Parameter
from lala.utils import graph_html
import numpy as np
        


class Model:
    def __init__(self, loss_fn: Callable):
        self.l1 = Parameter((3, 2), label="Layer1")
        self.l2 = Parameter((2, 1), label="Layer2")
        self.l3 = Parameter((1, 1), label="Layer3")
        self.l4 = Parameter((1, 2), label="Layer4")
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
            # loss.backward()
            loss.label = "Loss"
        # self.step()
        return loss


    
    def step(self):
        self.l1.step()
        self.l2.step()

        self.l1.zero_grad()
        self.l2.zero_grad()


def pred(w: Matrice, bias: Matrice, w2: Matrice, bias2: Matrice, inputs: Matrice):
    logits = (inputs @ w) + bias
    logits = (logits @ w2 ) + bias2
    return logits

import numpy as np

rand = np.random.rand(3, 2).tolist()

w1 = Matrice(rand, label="Weight1", requires_grad=True)
b1 = Matrice(np.random.rand(1, 2), label="Bias1", requires_grad=True)

rand = np.random.rand(2, 3).tolist()

w2 = Matrice(rand, label="Weight2", requires_grad=True)
b2 = Matrice(np.random.rand(1, 3), label="Bias1", requires_grad=True)



input_ = Matrice(np.random.rand(1, 3), label="Input1")
target = Matrice(input_.data, label="Target1")


for i in range(30):
    logits = pred(w1, b1, w2, b2, input_)
    loss = (logits - target).mean()
    loss.backward()

    w1 = w1 - w1.grad
    b1 = b1 - b1.grad
    if not i:
        graph_html(loss)
    print(loss.data)

