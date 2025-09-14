from lala.matrice import Matrice
from lala.functional import relu
from lala.matrice import float32, int32



a = Matrice.fill(2000, 300, value=-1, dtype=int32)
b = Matrice.rand(300, 400, dtype=float32)

# c = a @ b
# d = relu(c)

print(a.shape, b.shape)
print(list(a._data.ptr[i] for i in range(10)))
print(list(b._data.ptr[i] for i in range(10)))
# print(list(c._data.ptr[i] for i in range(10)))
# print(list(d._data.ptr[i] for i in range(10)))



