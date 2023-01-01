from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import os

os.environ['OMP_NUM_THREADS'] = '8' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS'] = '8' # set number of MKL threads to run in parallel
os.environ['MKL_DEBUG_CPU_TYPE'] = '5' # utilize AVX2

# h:(0.25,2)

def states(L,h):

    J = 1

    J_x = [[-J, i, (i+1)%L] for i in range(L)]
    h_z = [[-h, i] for i in range(L)]

    static = [["xx",J_x],["z",h_z]]
    dynamic = []

    basis = spin_basis_1d(L = L)

    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
    H = hamiltonian(static, dynamic, dtype=np.float64, basis = basis, **no_checks)

    E,V = H.eigsh(k = 1, which = 'SA')

    return E, V[:,0]
