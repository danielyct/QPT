from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions

def states(L,h):
    
    L = 2 * L
    basis=spinless_fermion_basis_1d(L,Nf = L//2)
    eta = h
    t = 1.0
    mu = 3
    nsint = t * (1 + eta)
    nuint = t * (1 - eta)

    n1 = [[-mu,i] for i in range(0,L,2)]
    n2 = [[-mu,(i+1)%L] for i in range(0,L,2)]
    d1 = [[-nsint,i,(i+1)%L] for i in range(0,L,2)]
    d2 = [[-nsint,(i+1)%L,i] for i in range(0,L,2)]
    d3 = [[-nuint,(i+1)%L,(i+2)%L] for i in range(0,L,2)]
    d4 = [[-nuint,(i+2)%L,(i+1)%L] for i in range(0,L,2)]

    static=[["n",n1],["n",n2],["-+",d1],["-+",d2],["-+",d3],["-+",d4]]
    dynamic=[]

    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
    H=hamiltonian(static, dynamic, basis=basis, dtype=np.float64,**no_checks)

    E,V = H.eigsh(k = 1, which = 'SA')

    return E, V[:,0]
