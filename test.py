import numpy as np


vals = [1,2,3,4,5,6,7,1,2,3,4,5,4,3,2]
vals = np.array(vals)

box = np.zeros((5,5))

idx = np.tril_indices(5)
print(idx)

box[idx] = vals
print(box)






