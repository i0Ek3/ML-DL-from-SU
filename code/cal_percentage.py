#!/usr/bin/env python3

import numpy as np

# create 3x4 array
m = np.array([[1,2,3,4], [6,4,3,5], [2,4,6,1]])
percentage = 100 * m / (m.sum(axis=0)).reshape(1,4)
print(percentage)
