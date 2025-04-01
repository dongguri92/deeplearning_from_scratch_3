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