import ctypes 
from typing import Union

class Dtype:
    """
    name: Name your dtype
    bytes: How much memory does it take
    base: the C/C++ equivalent use with cffi interfacing with the _C backend
    strength: used to determine the result's dtype in case of an op between two Tensors for different dtype
    """

    def __init__(self, name: str, bytes: int, base: str, python_eq,  strength=0):
        self.name = name
        self.bytes = bytes
        self.base = base #the C/C++ equivalent
        self.ptr_t = base + "*"
        self.strength = strength
        self.python_eq = python_eq

    def __lt__(self, other: "Dtype"): return self.strength < other.strength
    def __gt__(self, other: "Dtype"): return self.strength > other.strength

        

#this only exists for comparison (replaces none dtype objects where dtype is required)
Null = int8 = Dtype("NULL", 1, "NULL", None, 0) 


char = int8 = Dtype("char", 1, "char", None, 1)
int16  = Dtype("int16", 2, "int16_t", int, 2)
int32  = Dtype("int32", 4, "int", int, 3)
float32  = Dtype("float32", 4, "float", float,  4)
