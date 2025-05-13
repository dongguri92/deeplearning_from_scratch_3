## 이차미분

import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True)
print(x.grad)

# 두번째 역전파 시행
gx = x.grad
gx.backward()
print(x.grad) # 미분값이 남아 있는 상태에서 새로운 역전파 수행하여 새로운 미분값이 '더해진' 상태

x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True)
print(x.grad)

gx = x.grad
x.cleargrad() # 미분값 재설정
gx.backward()
print(x.grad)

# 미분 자동 수행
x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad 

    x.data -= gx.data / gx2.data
