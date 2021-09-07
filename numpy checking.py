import  numpy as np


x = np.arange(5*5*5*5, dtype=np.uint8)
print(x)
x = x.reshape(5,5,5,5)
print(x)