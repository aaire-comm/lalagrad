#Equevalent to pytorches Aten dispatcher
#just a bunch of python function that call to their equivalent C cffi lib  functions

from .shared._c_lib_tensor import lib, ffi
from typing import Tuple

class Blob: ...

#TODO: We want to change this to a much faster


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
    
    def mean_t(t: Blob, res: Blob):
        lib.mean_int(t._get_pointer("float*"), res._get_pointer("float*"), int(t.nbytes/4))

    @staticmethod
    def fill(t: Blob, value):
        lib.fill_float(t._get_pointer("int*"), value)

    @staticmethod
    def matmul(t1: Blob, t2: Blob, s0: Tuple[int], s1: Tuple[int]):
        s0_ptr = ffi.new(f"int[{len(s0)}]", s0)
        s1_ptr = ffi.new(f"int[{len(s1)}]", s1)
        lib.matmul_int(t1._get_pointer("int*"), t2._get_pointer("int*"), s0_ptr, s1_ptr)



class Float32Ops:

    @staticmethod
    def add_t(rhs: Blob, lhs: Blob, res: Blob):
        lib.add_float(rhs._get_pointer("float*"), lhs._get_pointer("float*"), res._get_pointer("float*"), int(rhs.nbytes/4))

    @staticmethod
    def mean_t(t: Blob, res: Blob):
        lib.mean_float(t._get_pointer("float*"), res._get_pointer("float*"), int(t.nbytes/4))

    @staticmethod
    def sum_t(t: Blob, res: Blob):
        lib.sum_float(t._get_pointer("float*"), res._get_pointer("float*"), int(t.nbytes/4))

    @staticmethod
    def fill(t: Blob, value):
        lib.fill_float(t._get_pointer("float*"), value, int(t.nbytes/4))
    
    @staticmethod
    def matmul(t1: Blob, t2: Blob, ret: Blob, lhs_rows: int, lhs_cols: int, rhs_cols: int):
        lib.matmul_float(lhs_rows, lhs_cols, rhs_cols, t1._get_pointer("float*"), t2._get_pointer("float*"), ret._get_pointer("float*"))

    @staticmethod
    def batch_matmul(t1: Blob, t2: Blob, lhs_shape: Tuple[int], lhs_stride: Tuple[int], rhs_shape: Tuple[int], rhs_stride: Tuple[int], dims: int):
        pointer = lib.batch_matmul_float(t1._get_pointer("float*"), t2._get_pointer("float*"), lhs_shape, lhs_stride, rhs_shape, rhs_stride, dims)
        return pointer

    @staticmethod
    def transpose(t: Blob, res: Blob, dim0: int, dim1: int, strides: Tuple[int]):
        s = ffi.new(f"int[{len(strides)}]", strides)
        lib.transpose_float(t._get_pointer("float*"), res._get_pointer("float*"), dim0, dim1)

    def cast(t: Blob, t1: Blob, size: int):
        lib.cast_int_float(t._get_pointer("int*"), t1._get_pointer("float*"), int(t.nbytes/4))

    @staticmethod
    def mul_t(rhs: Blob, lhs: Blob, res: Blob):
        lib.mul_float(rhs._get_pointer("float*"), lhs._get_pointer("float*"), res._get_pointer("float*"), int(rhs.nbytes/4))

    @staticmethod
    def mul_s(rhs: Blob, s: Blob, res: Blob):
        lib.mul_scalar_float(rhs._get_pointer("float*"), s, res._get_pointer("float*"), int(rhs.nbytes/4))

    @staticmethod
    def sub_t(lhs: Blob, rhs: Blob, res: Blob):
        lib.sub_float(lhs._get_pointer("float*"), rhs._get_pointer("float*"), res._get_pointer("float*"), int(lhs.nbytes/4))
    
    @staticmethod
    def exp(lhs: Blob, exp: float, res: Blob):
        lib.power_scalar_float(lhs._get_pointer("float*"), exp, res._get_pointer("float*"), int(lhs.nbytes/4))

ops = {
    "int32": Int32Ops,
    "float32": Float32Ops
}