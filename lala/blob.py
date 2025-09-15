from typing import Optional
from .utils import _to_python_list
import ctypes
from lala.dtype import Dtype, float32, char
from lala._C import ffi, lib as libc


class Blob:
    def __init__(self, ptr: Optional[ctypes.POINTER]=None, nbytes: Optional[int]=None, fill=None, zero_init=True):
        assert nbytes is not None or ptr is not None, "Blob requires nbytes or ptr"
        self.nbytes = nbytes
        self.zero_init = zero_init

        self.__ptr =  ffi.cast("int*", ptr if ptr is not None else libc.malloc(nbytes))

        if zero_init:
            libc.memset(self.__ptr, 0, self.nbytes)
        elif fill is not None:
            libc.memset(self.__ptr, fill, self.nbytes)

    
    def _get_pointer(self, dtype: Optional[Dtype]=None):
        if dtype is None:
            return self.__ptr
        return ffi.cast(dtype.ptr_t, self.__ptr)
    
    def _release(self):
        libc.free(self._get_pointer(float32))
        return True
    
    def _to_python_list(self, shape):
        return _to_python_list(self._get_pointer(float32), shape)

    def __getitem__(self, key):
        return self._get_pointer(char)[key]

    def _clone(self, other):
        libc.memcpy(self._get_pointer(float32), other._get_pointer(float32), self.nbytes)
        return True