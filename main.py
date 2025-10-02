import lala
import os
import numpy as np
import random
print(f"PID: {os.getpid()}")

l = lala.tensor([[random.random() for i in range(1)] for j in range(2)] , dtype=lala.float32, requires_grad=True)
l2 = lala.tensor([[random.random() for i in range(1)] for r in range(2)], dtype=lala.float32, requires_grad=True)


l6 = l @ l2.T
l7 = l6.mean()

# l7.backward()

import torch


l1 = torch.tensor(l.tolist(), dtype=torch.float32, requires_grad=True)
l3 = torch.tensor(l2.T.tolist(), dtype=torch.float32, requires_grad=True)


l8 = l1 @ l3
l10 = l8.mean()

l10.backward()

print(np.array(l8.tolist()) - np.array(l6.tolist()))

print()
