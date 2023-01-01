import os
import time
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plot fidelity graph
import ssh_model as ssh
import long_range_ssh as lrssh
import sys

start = time.perf_counter()

size = int(sys.argv[1])
sample_no = int(sys.argv[2])
eta = float(sys.argv[3])

# find fidelity, return fidelity, susceptibility, x-axis, degeneracy
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

F, x = find_fid(size = size, delta = 0.1, eta = eta, c = 1, d = 0)

file = 'data/M_eta'+str(eta)+'_size'+ str(size) + '_' + str(sample_no) +'lrssh_cd_'

np.save(file + 'F.npy', F)
np.save(file + 'x.npy', x)

end = time.perf_counter()
print(f"finished in {end - start:0.4f} seconds")
