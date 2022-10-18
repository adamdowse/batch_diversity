import numpy as np
a = np.array([1,2,3,4,5,6,7,8,9,10])
b = np.array([1,3,5])

c = np.ones(a.size,dtype=bool)
c[b] = False

print(a[b])
print(a[c])