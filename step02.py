import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data

class Function_: # 수정전 function class
    def __call__(self, input):
        x = input.data # 데이터를 꺼낸다
        y = x ** 2
        output = Variable(y) # Variable 형태로 되돌린다
        return output
    
x = Variable(np.array(10))
f = Function_()
y = f(x)

print(type(y))
print(y.data)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 구체적인 계산은 forward 메서드에서 한다.
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)