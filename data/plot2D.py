import numpy as np
import matplotlib.pyplot as plt 
import sys

def find_sus(fid, delta):
    return -2 * np.log(fid) / delta**2

file = sys.argv[1]

sample_no = int(sys.argv[2])

x = np.linspace(-5,5,sample_no)

F = np.load(file)

start = x[0]
end = x[-1]

fig, ax = plt.subplots()
ax.set_title(file)
ax.set_ylabel(r"$U_j$")
ax.set_xlabel(r"$U_i$")
# pc = ax.pcolormesh(x, x, F, vmin=0, vmax=1, cmap='seismic')
im = ax.imshow(F, interpolation = 'spline16', cmap = 'seismic', extent = [start, end, start, end], origin = 'lower')
fig.colorbar(im)

plt.show()
