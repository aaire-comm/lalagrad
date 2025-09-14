from lala import float32, int32
from lala import Tensor


a = Tensor.fill(4, 3, 2, 1, value=-1, dtype=int32)


b = a.view(2, 3, 1, 4)

import torch

print(torch.tensor(b.tolist()).shape)