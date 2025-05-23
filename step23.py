import numpy as np
from dezero.core_simple import Variable

x = Variable(np.array(1.0))
print(x)



if '__file__' in globals(): # 전역 변수가 정의되어 있는지 확인 --> "지금 코드가 .py 파일로 실행되고 있는가?"
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..")) # 현재 디렉토리에서 상위 디렉토리 모듈 경로 모두 추가, github보면 step이라는 폴더안에 위치해서 상위 디렉토리를 추가한 것으로 생각함

import numpy as np
from dezero import Variable
x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)