import math
import numpy as np
from numpy import exp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mask = [
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]

transformed = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
]

K = M = 3
L = N = 3

for k in range(K):
    for l in range(L):
        accum = 0
        for m in range(M):
            for n in range(N):
                accum += mask[m][n] * exp(-2j * math.pi * k * m / K) * exp(-2j * math.pi * l * n / L)
        complexAccum = accum / 16
        transformed[k][l] = accum / 16

mask_dft = np.fft.fft2(mask)
print(mask_dft)

Axes3D.plot_surface(transformed)
plt.show()

