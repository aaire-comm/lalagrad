# from typing import Tuple, List, Callable, Optional
# import lala
# from lala.functional import relu
# # """
# # #implement a simple Attention block (Not masked)
# # #as of the 2015 (Attention is all you need paper from Google)
# # #Single Head Attention

# # def mean_square_loss(logits: lala.Tensor, target: lala.Tensor):
# #     mse  = (logits - target).spow(2).mean()
# #     return mse

# # class Attention:
# #     def __init__(self, token_len: int, embeding_dim: int, attention_score_dim, loss_fn: Optional[Callable[[lala.Tensor, lala.Tensor], lala.Tensor]]=None):
# #         self.input_dim = (token_len, embeding_dim)
# #         self.wq = lala.empty(embeding_dim, attention_score_dim, requires_grad=True)
# #         self.wk = lala.empty(embeding_dim, attention_score_dim, requires_grad=True)
        
# #         self.wv = lala.empty()
# #         self.loss_fn = loss_fn if loss_fn is not None else mean_square_loss

# #     def forward(self, x: lala.Tensor):
# #         assert x.shape == self.input_dim, f"Input must be of shape {self.input_dim}"
# #         Q = x @ self.wq
# #         K = x @ self.wk
# #         V = x @ self.wv

# #         ap = Q @ K.T #Attention pattern/Attention Score

# #         out = relu(ap @ V)
# #         return out
    
# #     def bach_train(self, bach_data: List[Tuple[lala.Tensor, lala.Tensor]]):
# #         batch_loss = lala.Tensor(0)
# #         for  data in bach_data:
# #             print(data)
# #             input_, target = data

# #             logits = self.forward(input_)
# #             loss = self.loss_fn(logits, target)
# #             batch_loss += loss
# #             loss.detach()

# #             bach_data

# #         batch_loss.backward()
# #         return batch_loss


# # #Train an attention block to map input data to its self
# # token_len = 128
# # embedding_dim = 756
# # batch_len = 10

# # att_blk = Attention(128, 756, 64)

# # #generate a bunch pf (token_len, embedding_dim) tensors
# # batch_training_data = [ ]

# # for k in range(batch_len) :
# #     input_ = lala.Tensor(data=[[i*e*k for i in range(embedding_dim)] for e in range(token_len)])
# #     target = input_.view() #this creates a Tensor pointing to the same storage with the same shape ad dtype (saves memory)

# #     batch_training_data.append((input_, target))

# # att_blk.bach_train(batch_training_data)

# # """




# class MLP:
#     def __init__(self, input_features: int, layers: List[int], output_features: int):
#         self.layers = [(lala.zeros(input_features, layers[0], requires_grad=True), lala.zeros(input_features, layers[0], requires_grad=True))]
#         for i in range(1, len(layers)):
#             self.layers.append((lala.zeros(layers[i-1], layers[i], requires_grad=True), lala.zeros(layers[i], requires_grad=True)))
        
#     def forward(self, x: lala.Tensor):
#         for w, b in self.layers:
#             x = x @ w + b
#         return x

#     def train(self, batch_data: List[Tuple[lala.Tensor, lala.Tensor]]):
#         for input_, target in batch_data:
#             logits = self.forward(input_)
#             loss = (logits - target).spow(2).mean()
#             loss.backward()
#             print(loss.get_item())

#         for i in range(len(self.layers)):
#             self.layers[i] = (
#                 self.layers[i][0] - self.layers[i][0].grad,
#                 self.layers[i][1] - self.layers[i][1].grad,
#             )

#         loss.visualize()



# input_features = 100
# output_features = 10
# layers = [8, 8]

# # mlp = MLP(100, layers, 10)

# import numpy as np

# data = []
# batch_len = 10
# for i in range(batch_len):
#     input_ = lala.tensor(data=np.random.rand(1,input_features))
#     target = input_.smul(2)
#     data.append((input_, target))



# layer = mlp.layers[0][0]
# mlp.train(data)


import lala
import os
import numpy as np

print(f"PID: {os.getpid()}")

l = lala.tensor([[[r for i in range(5)] for j in range(5)] for r in range(5)], dtype=lala.float32, requires_grad=True)
l2 = lala.tensor([[j for i in range(5)] for j in range(5)])


l3 = l @ l2.expand(5, 5, 5)

l4 = l3[::2, 2:5, 2:]
print(l4.tolist())
print(l4.stride())
print(hex(l4.data_ptr()))
print(l3.storage)

import torch
print(torch.tensor(l3.tolist(), dtype=torch.float32))
