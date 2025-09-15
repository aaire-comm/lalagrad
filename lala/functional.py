from .tensor import Matrice, Relu
from ._C.lib_loader import lib

def relu(m: Matrice): return Relu(m)()