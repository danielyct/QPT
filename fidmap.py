import os
import time
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plot fidelity graph
import ssh_model as ssh
import long_range_ssh as lrssh
import sys

os.environ['OMP_NUM_THREADS'] = '8' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS'] = '8' # set number of MKL threads to run in parallel
os.environ['MKL_DEBUG_CPU_TYPE'] = '5' # utilize AVX2

start = time.perf_counter()

size = int(sys.argv[5])
V = int(sys.argv[1])
sample_no = 50
eta = float(sys.argv[2])
c = float(sys.argv[3])
d = float(sys.argv[4])
print(V)
print(eta)
print(c)
print(d)

# find fidelity, return fidelity, susceptibility, x-axis, degeneracy
def find_fid(size, **kwargs):
    h_list = np.linspace(-5, 5, sample_no)
    F = np.zeros((sample_no,sample_no), float)
    degen = 0
    for i,h in enumerate(h_list):   # y-axis
        for j,k in enumerate(h_list):   # x-axis
            E_1, s0_1 = lrssh.states(L = size, U = h, **kwargs)
            E_2, s0_2 = lrssh.states(L = size, U = k, **kwargs)
            F[i,j] = abs(np.dot(s0_1,s0_2))
            print("size: " + str(size) + "  sample no.: ", str(i), str(j))
    return F, h_list

F, x = find_fid(size = size, eta = eta, c = c, d = d, V = V)

file = 'data/fid_map/eta' + str(eta) + 'V' + str(V) + 'size' + str(size) + 'c' + str(c) + 'd' + str(d) + 'n' + str(sample_no)

np.save(file + 'F.npy', F)

end = time.perf_counter()
print(f"finished in {end - start:0.4f} seconds")
