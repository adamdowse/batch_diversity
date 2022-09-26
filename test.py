import numpy as np
a = np.array([1,2,3,4,5,6,7])

b = np.concatenate(([10],a[:-1]))

print(b)