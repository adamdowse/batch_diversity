import numpy as np
a = np.array([0,2,4],dtype=int)

preds = [0,0,0,0,0]
b = [10,11,12]
for count, i in enumerate(a):
    preds[i] = b[count]
print(np.zeros((2,5)))


