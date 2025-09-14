from .tensor import Matrice, Relu
from .c.lib_loader import lib

def relu(m: Matrice): return Relu(m)()