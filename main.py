import lala

lt = lala.Tensor(data=[[i*j for i in range(10)] for j in range(10)], dtype=lala.float32, requires_grad=True)
lt2 = lala.Tensor(data=[[i*j for i in range(70)] for j in range(10)], dtype=lala.float32)

lt4 = lt.mean()
lt.backward(lt)

print(lt.tolist())
# lt4.backward()

lt4.visualize()