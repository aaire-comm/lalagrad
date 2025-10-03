import lala


input_features, neurons = 2, 2


#model Weights and biases
weights = lala.rand(input_features, neurons,  requires_grad=True)
bias = lala.rand(1, neurons, requires_grad=True)

epoch, batch = 25, 5

losses = []
for epoch in range(epoch):
    batch_loss = 0.0
    for i in range(batch):
        #input and target
        input_ = lala.rand(1, input_features)
        #use the input as a target (teaching the nn to map input to itself)
        target = input_.smul(3)

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
        batch_loss += loss.get_item()
    losses.append(batch_loss)
    print(f"epoch {epoch} batch avg loss:", batch_loss/batch)

loss.visualize()

print("""
This code implements a Single Layer Fully Connected neural net 
The network is trained to map 

Assuming correct lalagrad installation you should see an output of decreasing numbers
which is the loss of the model going down as the net trains multiply an input_ by 2 
      
it also generates graph.html open it with a browser to see your computation graph
""")

#plot the loss over time
import matplotlib.pyplot as plt
plt.plot([x for x in range(epoch+1)], losses, label="lalagrad")

plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss over epoch")
plt.legend()

plt.show()