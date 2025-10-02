import lala
import os
import numpy as np
import random
print(f"PID: {os.getpid()}")

l = lala.tensor([[[random.random() for i in range(2)] for j in range(2)] for r in range(2)], dtype=lala.float32, requires_grad=True)
l2 = lala.tensor([[random.random() for i in range(2)] for r in range(2)], dtype=lala.float32, requires_grad=True)


l6 = l @ l2.expand(2, *l2.shape)
l7 = l6.mean()


import torch


l1 = torch.tensor(l.tolist(), dtype=torch.float32, requires_grad=True)
l3 = torch.tensor(l2.tolist(), dtype=torch.float32, requires_grad=True)


l8 = l1 @ l3.expand(2, *l3.shape)
l10 = l8.mean()

l10.backward()

# print(l1.grad)
print(np.array(l.tolist()))
print("**********************************")
print(np.array(l2.T.tolist()))
print(np.array(l2.tolist()))
print("**********************************")
print(np.array(l6.tolist()))
print("**********************************")
print(np.array(l8.tolist()) == np.array(l6.tolist()))

print()
