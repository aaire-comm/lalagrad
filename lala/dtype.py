import ctypes 
from typing import Union


class Dtype:
    def __init__(self, name: str, bytes: int, base: Union[ctypes.c_int, ctypes.c_float], python_eq,  strength=0):
        self.name = name
        self.bytes = bytes
        self.base = base
        self.ptr_t = base + "*"
        self.strength = strength
        self.python_eq = python_eq

    def __lt__(self, other: "Dtype"): return self.strength < other.strength
    def __gt__(self, other: "Dtype"): return self.strength > other.strength

        

char = Dtype("char", 1, "char", None, 0)
int32  = Dtype("int32", 4, "int", int, 1)
float32  = Dtype("float32", 4, "float", float,  2)
