import lala

"""
This code implements a Single Layer Fully Connected neural net 
The network is trained to map 

Assuming correct lalagrad installation you should see an output of decreasing numbers
which is the loss of the model going down as the net trains over several loops
"""
input_features, neurons = 2, 3

#Weights and biases
weights = lala.rand(input_features, neurons,  requires_grad=True)
bias = lala.rand(1, neurons, requires_grad=True)

for _ in range(100):
    loss_avg = 0.0
    for i in range(10):
        #input and target
        input_ = lala.rand(1, input_features)
        #use the input as a target (teaching the nn to map input to itself)
        target = input_ * 2

        #model pred and loss
        logits = input_ @ weights + bias
        loss = (logits - target).spow(2).mean()

        #go backward and calculate gradients
        loss.backward()
        #update weights and biases
        weights -= weights.grad
        bias -= bias.grad
        weights.detach()
        bias.detach()

        #remove grad for next run
        weights.grad = None; bias.grad = None
        loss_avg += loss.get_item()
    print("100 loops avg loss:", loss_avg/100)

loss.visualize()


