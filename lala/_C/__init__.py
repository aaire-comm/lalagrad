#Equevalent to pytorches Aten dispatcher
#just a bunch of python function that call to their equivalent C cffi lib  functions

from .shared._c_lib_tensor import lib, ffi
from typing import TYPE_CHECKING

class Blob: ...


class Int32Ops: 
    
    @staticmethod
    def add_t(rhs: Blob, lhs: Blob, res: Blob):
        lib.add_int(rhs._get_pointer("int*"), lhs._get_pointer("int*"), res._get_pointer("int*"), int(rhs.nbytes/4))

    @staticmethod
    def add_s(rhs: Blob, scalar: int, res: Blob):
        lib.add_scalar_int(rhs._get_pointer("int*"), scalar, res._get_pointer("int*"), int(rhs.nbytes/4))
    
    @staticmethod
    def sub_t(rhs: Blob, lhs: Blob, res: Blob):
        lib.sub_int(rhs._get_pointer("int*"), lhs._get_pointer("int*"), res._get_pointer("int*"), int(rhs.nbytes/4))

    @staticmethod
    def sub_s(rhs: Blob, lhs: Blob, res: Blob):
        lib.sub_scalar_int(rhs._get_pointer("int*"), lhs._get_pointer("int*"), res._get_pointer("int*"), int(rhs.nbytes/4))

    @staticmethod
    def mul_t(rhs: Blob, lhs: Blob, res: Blob):
        lib.mul_int(rhs._get_pointer("int*"), lhs._get_pointer("int*"), res._get_pointer("int*"), int(rhs.nbytes/4))

    @staticmethod
    def mul_s(rhs: Blob, scalar: int, res: Blob):
        lib.mul_scalar_int(rhs._get_pointer("int*"), scalar, res._get_pointer("int*"), int(rhs.nbytes/4))

    @staticmethod
    def div_t(rhs: Blob, lhs: Blob, res: Blob):
        lib.div_int(rhs._get_pointer("int*"), lhs._get_pointer("int*"), res._get_pointer("int*"), int(rhs.nbytes/4))
    
    @staticmethod
    def div_s(rhs: Blob, scalar: int, res: Blob):
        lib.div_scalar_int(rhs._get_pointer("int*"), scalar, res._get_pointer("int*"), int(rhs.nbytes/4))

    @staticmethod
    def sum_t(t: Blob, res: Blob):
        lib.sum_int(t._get_pointer("int*"), res._get_pointer("int*"), int(t.nbytes/4))
    
    def fill(t: Blob, value):
        lib.fill_float(t._get_pointer("int*"), value)



class Float32ops:

    @staticmethod
    def add_t(rhs: Blob, lhs: Blob, res: Blob):
        lib.add_float(rhs._get_pointer("float*"), lhs._get_pointer("float*"), res._get_pointer("float*"), int(rhs.nbytes/4))

    @staticmethod
    def mean_t(t: Blob, res: Blob):
        lib.mean_(t._get_pointer("float*"), res._get_pointer("float*"), int(t.nbytes/4))

    @staticmethod
    def sum_t(t: Blob, res: Blob):
        lib.sum_float(t._get_pointer("float*"), res._get_pointer("float*"), int(t.nbytes/4))

    def fill(t: Blob, value):
        lib.fill_float(t._get_pointer("float*"), value, int(t.nbytes/4))

    
ops = {
    "int32": Int32Ops,
    "float32": Float32ops
}