import numpy as np
import sys
import ctypes

rrows = rcols = lrows = lcols = int(sys.argv[1])

RowR = ctypes.c_int * rcols
RowL = ctypes.c_int * lcols
RowRes = ctypes.c_int * lcols

MatriceR = RowR * rrows
MatriceL = RowL * rcols
MatriceRes = RowRes * lrows

lib = ctypes.CDLL("./libtensor.so")

lib.matmul.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),  # flat int *
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]

lib.mul.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]

# lib.add.argtypes = [
#     ctypes.c_int,
#     ctypes.c_int,
#     ctypes.POINTER(ctypes.c_int),
#     ctypes.POINTER(ctypes.c_int),
#     ctypes.POINTER(ctypes.c_int),
# ]

# Allocate NumPy arrays
rhs_np = np.ones((rrows, rcols), dtype=np.int32)
lhs_np = np.ones((rcols, lcols), dtype=np.int32)
res_np = np.zeros((rrows, lcols), dtype=np.int32)

# Get raw int * pointers
rhs = rhs_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
lhs = lhs_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
res = res_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

import time

start = time.perf_counter()
lib.mul(rrows, rcols, rhs, lhs, res)
lib.matmul(rrows, rcols, lcols, rhs, lhs, res)
# lib.add(rrows, rcols, rhs, lhs, res)
# lib.add(rrows, rcols, res, res, res)
end = time.perf_counter()
print(end - start)

# After call, res_np contains the result
print(res_np, res_np.nbytes)
