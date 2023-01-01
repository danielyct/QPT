import time
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plot fidelity graph
import long_range_ssh as lrssh
import sys

size = 4
sample_no = 20
eta = 0.6

def find_fid(size, delta, **kwargs):
    h_list = np.linspace(-5, 5, sample_no)
    F = np.zeros((sample_no,sample_no), float)
    degen = 0
    for i,h in enumerate(h_list):   # y-axis
        for j,k in enumerate(h_list):   # x-axis
            E_1, s0_1 = lrssh.states(L = size, U = k, V = h, **kwargs)
            E_2, s0_2 = lrssh.states(L = size, U = k+delta, V = h, **kwargs)
            F[i,j] = abs(np.dot(s0_1,s0_2))
            print("size: " + str(size) + "  sample no.: ", str(i), str(j))
    return F, h_list

F, x = find_fid(size = size, delta = 0.1, eta = 0.6, c = 0, d = 0)

start = x[0]
end = x[-1]

fig, ax = plt.subplots()
ax.set_title("lrssh Fidelity")
ax.set_ylabel("V")
ax.set_xlabel("U")
im = ax.imshow(F, interpolation = 'spline16', cmap = 'seismic', extent = [start, end, start, end])
fig.colorbar(im)

plt.show()
