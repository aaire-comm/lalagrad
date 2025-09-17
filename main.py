import lala

lt = lala.Tensor(data=[[i*j for i in range(10)] for j in range(10)], dtype=lala.float32, requires_grad=True)
lt2 = lala.Tensor(data=[[i*j for i in range(10)] for j in range(10)], dtype=lala.float32)

lt3 = lala.ones(2, 3, dtype=lala.int32)
d = lala.Tensor.dummy(1)
print(lt3.tolist())
print(d.tolist())
print(d.shape)
print(d.storage)
# l2 = lt + lt2
# l3 = l2.smul(2)
# l4 = l3.mean()

# l4.backward()
