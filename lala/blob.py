from typing import Optional
from .utils import _to_python_list
import ctypes
from lala.dtype import Dtype, float32, char
from lala._C import ffi, lib as libc


class Blob:
    def __init__(self, nbytes: int, ptr: Optional[ctypes.POINTER]=None, fill=None, zero_init=False):
        assert nbytes is not None or ptr is not None, "Blob requires nbytes or ptr"
        self.nbytes = nbytes
        self.zero_init = zero_init

        self.__ptr =  ffi.cast("void*", ptr if ptr is not None else libc.malloc(nbytes))

        if zero_init:
            libc.memset(self.__ptr, 0, self.nbytes)
        elif fill is not None:
            libc.memset(self.__ptr, fill, self.nbytes)

    def __repr__(self):
        return f"Blob({self._get_pointer("void*")} <nbytes={self.nbytes}>)"
    def _get_pointer(self, dtype: str):
        if dtype is None:
            return self.__ptr
        return ffi.cast(dtype, self.__ptr)
    
    def copy(self, other: "Blob"):
        size = min(self.nbytes, other.nbytes)
        libc.memcpy(self._get_pointer(), other._get_pointer(), size)
    
    def _release(self):
        libc.free(self._get_pointer(float32))
        return True
    
    def _to_python_list(self, shape, dtype: str):
        return _to_python_list(self._get_pointer(dtype), shape)

    def _get_item(self, offset: int, dtype: str):
        return self._get_pointer(dtype)[offset]

    def _copy(self, other):
        libc.memcpy(self._get_pointer(float32), other._get_pointer(float32), self.nbytes)
        return True