import ctypes 
from typing import Union

class Dtype:
    def __init__(self, name: str, bytes: int, base: Union[ctypes.c_int, ctypes.c_float]):
        self.name = name
        self.bytes = bytes
        self.base = base
        

int32  = Dtype("int32", 4, ctypes.c_int)
float32  = Dtype("float32", 4, ctypes.c_float)
