from typing import Tuple, List, Callable, Optional
import lala
from lala.functional import relu
"""
#inplement a simple Attention block 
#as of the 2015 (Attention is all you need paper from Google)
#Single Head Attention

def mean_square_loss(logits: lala.Tensor, target: lala.Tensor):
    mse  = (logits - target).spow(2).mean()
    return mse

class Attention:
    def __init__(self, token_len: int, embeding_dim: int, attention_score_dim, loss_fn: Optional[Callable[[lala.Tensor, lala.Tensor], lala.Tensor]]=None):
        self.input_dim = (token_len, embeding_dim)
        self.wq = lala.empty(embeding_dim, attention_score_dim, requires_grad=True)
        self.wk = lala.empty(embeding_dim, attention_score_dim, requires_grad=True)
        
        self.wv = lala.empty()
        self.loss_fn = loss_fn if loss_fn is not None else mean_square_loss

    def forward(self, x: lala.Tensor):
        assert x.shape == self.input_dim, f"Input must be of shape {self.input_dim}"
        Q = x @ self.wq
        K = x @ self.wk
        V = x @ self.wv

        ap = Q @ K.T #Attention patter/Attention Score

        out = relu(ap @ V)
        return out
    
    def bach_train(self, bach_data: List[Tuple[lala.Tensor, lala.Tensor]]):
        batch_loss = lala.Tensor(0)
        for  data in bach_data:
            print(data)
            input_, target = data

            logits = self.forward(input_)
            loss = self.loss_fn(logits, target)
            batch_loss += loss
            loss.detach()

            bach_data

        batch_loss.backward()
        return batch_loss


#Train an attention block to map input data to its self
token_len = 128
embedding_dim = 756
batch_len = 10

att_blk = Attention(128, 756, 64)

#generate a bunch pf (token_len, embedding_dim) tensors
batch_training_data = [ ]

for k in range(batch_len) :
    input_ = lala.Tensor(data=[[i*e*k for i in range(embedding_dim)] for e in range(token_len)])
    target = input_.view() #this creates a Tensor pointing to the same storage with the same shape ad dtype (saves memory)

    batch_training_data.append((input_, target))

att_blk.bach_train(batch_training_data)

"""
