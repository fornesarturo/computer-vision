import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

vx = np.array([np.exp(-16/2), np.exp(-9/2), np.exp(-4/2), np.exp(-1/2), 1, np.exp(-1/2), np.exp(-4/2), np.exp(-9/2), np.exp(-16/2)])
vy = np.array([[x] for x in vx])
mask = 1 / (2*np.pi) * vx * vy

x, y = np.mgrid[-9:9:9*1j, -9:9:9*1j]
freqMag = np.absolute(np.fft.fft2(mask))

fig = plt.figure(figsize=plt.figaspect(0.5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax1.plot_surface(x, y, mask)
ax2.plot_surface(x, y, freqMag)
plt.show()