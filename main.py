import lala


#Weights and biases
weights = lala.zeros(2, 2, dtype=lala.float32, requires_grad=True)
bias =    weights.clone()

#input and target
input_ = lala.zeros(2, 2, dtype=lala.float32)
target = lala.zeros(2, 2, dtype=lala.float32)

#model pred and loss
logits = input_ @ weights + bias
loss = (logits - target).spow(2).mean()

#go backward and calculate gradients
loss.backward()

weights -= weights.grad
