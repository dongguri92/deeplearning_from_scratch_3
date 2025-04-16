## 2차이상의 고차 미분준비편
## Variable과 Function 클래스 보기

import numpy as np
from dezero import Variable, Function

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx
    

def sin(x):
    return Sin()(x)

x = Variable(np.array(1.0))
y = sin(x)
y.backward(retain_grad=True)
