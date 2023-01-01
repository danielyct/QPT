from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import os
from scipy.sparse.linalg import eigsh

os.environ['OMP_NUM_THREADS'] = '12' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS'] = '12' # set number of MKL threads to run in parallel
os.environ['MKL_DEBUG_CPU_TYPE'] = '5' # utilize AVX2

def states(L,eta = 0, c = 1, d = 1, U = 0, V = 0):
    
    L = 2 * L
    basis=spinless_fermion_basis_1d(L, Nf=L//2)
    a = -(1+eta)
    b = -(1-eta)
    
    I1 = [[U,i,(i+1)%L] for i in range(0,L,2)]
    I2 = [[V,(i+1)%L,(i+2)%L] for i in range(0,L,2)]

    tA = [[+a,i,(i+1)%L] for i in range(0,L,2)]
    ta = [[-a,i,(i+1)%L] for i in range(0,L,2)]

    tB = [[+b,(i+1)%L,(i+2)%L] for i in range(0,L,2)]
    tb = [[-b,(i+1)%L,(i+2)%L] for i in range(0,L,2)]
    
    tC = [[+c,i,(i+3)%L] for i in range(0,L,2)]
    tc = [[-c,i,(i+3)%L] for i in range(0,L,2)]
    
    tD = [[+d,(i+1)%L,(i+4)%L] for i in range(0,L,2)]
    td = [[-d,(i+1)%L,(i+4)%L] for i in range(0,L,2)]

    static=[["nn",I1],["nn",I2],["+-",tA],["-+",ta],["+-",tB],["-+",tb],["+-",tC],["-+",tc],["+-",tD],["-+",td]]
    dynamic=[]

    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
    H=hamiltonian(static, dynamic, basis=basis, dtype=np.float64,**no_checks)

    t0 = 0
    E,V = eigsh(H.aslinearoperator(time=t0),k=1,which="SA")

    return E, V[:,0]
