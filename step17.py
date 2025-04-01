class obj:
    pass

def f(x):
    print(x)

a = obj() # 변수 대입: 참조 카운트 1
f(a) # 함수에 전달: 함수 안에서는 참조 카운트 2
# 함수 완료 : 빠져나오면 참조 카운트 1
a = None # 대입 해제: 참조 카운트 0

a = obj()
b = obj()
c = obj()

a.b = b # a가 b를 참조
b.c = c # b가 c를 참조

# a : 1, b : 2, c : 2

a = b = c = None

# a : 0, b : 1, c : 1  --> a 삭제, b와c는 연달아 삭제

a = obj()
b = obj()
c = obj()

a.b = b # a가 b를 참조
b.c = c # b가 c를 참조
c.a = a # c가 a를 참조

# a : 2, b : 2, c : 2 --> 순환참조

a = b = c = None

# a : 1, b : 1, c : 1  --> 삭제되지 않음

# weakref 모듈을 사용하면 순환참조 문제를 해결할 수 있음

import weakref
import numpy as np

a = np.array([1, 2, 3])
b = weakref.ref(a)

print(b)
print(b())

a = None
print(b)

import weakref
import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output은 약한 참조
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs] # weakref사용 --> 출력 변수를 약하게 참조
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


for i in range(10):
    x = Variable(np.random.randn(10000))  # big data
    y = square(square(square(x)))