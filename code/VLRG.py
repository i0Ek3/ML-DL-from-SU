#!/usr/bin/env python3
# VLRG --> Vectorizing Logisitic Regression's Gradient

import numpy as np

Z = np.dot(w.T, X) + b
A = np.delta(Z)
dZ = A - Y
dw = 1/m * X * np.dot(d, z.T)
db = 1/m * np.num(dZ)
w := w - a * dw
b := b - a * db
