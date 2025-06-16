import numpy as np
from dezero import test_mode
import dezero.functions as F

x = np.ones(5)
print(x)

# When training
y = F.dropout(x)
print(y)

# when testing (predicting)
with test_mode():
    y = F.dropout(x)
    print(y)