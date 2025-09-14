from typing import Optional, List, Any
from .dtype import Dtype, float32
from .c.lib_loader import lib, libc
from .utils import _to_list
import ctypes

class Blob:
    def __init__(self, ptr: Optional[ctypes.POINTER]=None, dtype: Dtype=float32, nbytes: Optional[int]=None, zero_init=True):
        base = dtype.base
        self.nbytes = nbytes
        if ptr is None:
            if zero_init:
                ptr = (base * nbytes)()
            else:
                ptr = libc.malloc(dtype.bytes*nbytes)

        self.ptr = ctypes.cast(ptr, ctypes.POINTER(base))        

    
    @classmethod
    def from_list(cls, _from: List[List[Any]], dtype=float32):
        # np_dtype, base = (np.float32, ctypes.c_float) if dtype is float32 else (np.int32, ctypes.c_int)
        # arr = np.array(_from, dtype=np_dtype, order="C")
        # ptr = arr.ctypes.data_as(ctypes.POINTER(base))
        # len_ = len(arr)
        # del arr
        return
        # return cls(ptr, nbytes=len_, dtype=dtype)
    
    def __getitem__(self, index):
        return self.ptr[index]
    
    def fill(self, value):
        lib.fill_int(self.ptr, value, self.nbytes)
        return self
    
    def tolist(self, shape):
        return _to_list(self.ptr, shape)