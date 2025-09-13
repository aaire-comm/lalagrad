from typing import Tuple, Callable
from lala.matrice import Matrice
from lala.nn import Parameter
from lala.utils import graph_html
import numpy as np
        


class Model:
    def __init__(self, loss_fn: Callable):
        self.l1 = Parameter((2, 3), label="Layer1")
        self.l2 = Parameter((3, 5), label="Layer2")
        self.l3 = Parameter((5, 3), label="Layer3")
        self.l4 = Parameter((3, 2), label="Layer4")

        self.loss_fn = loss_fn

    def __call__(self, x):
        y = self.l1(x)
        y = self.l2(y)
        y = self.l3(y)
        y = self.l4(y)
        
        return y
    
    def batch_train(self, data):
        
        for i, dt in enumerate(data):
            inputs, target = dt
           
            logits = self(inputs)
            print(logits.data)
            loss = self.loss_fn(logits, target)

            loss.backward()
            loss.label = "Loss"
            if not i:
                print(loss.data)

        self.step()

        print(f"bach {i} loss:", loss.data)
        return loss


    
    def step(self):
        self.l1.step()
        self.l2.step()
        self.l3.step()
        self.l4.step()

        self.l1.zero_grad()
        self.l2.zero_grad()
        self.l3.zero_grad()
        self.l4.zero_grad()



def loss_fn(logits, target):
    return (logits - target).spow(2).mean()


inp= np.random.rand(2, 3)
tar = inp * 2
model = Model(loss_fn=loss_fn)
loss = model.batch_train([
    (Matrice([[2, 1]]), Matrice([[2, 1]])) for _ in  range(19)
])



loss.visualize()
    
