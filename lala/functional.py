from .matrice import Matrice, Relu

def relu(m: Matrice):
    relu = Relu(m)
    return Matrice(relu.forward(), grad_fn=relu, requires_grad=m.requires_grad)