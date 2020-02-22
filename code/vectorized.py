#!/usr/bin/env python3

import time
import numpy as np

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
tac = time.time()
print("Vectorized: " + str(1000*(tac-tic)) + "ms")

c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] * b[i]
tac = time.time()
print("Non-vectorized: " + str(1000*(tac-tic)) + "ms")

