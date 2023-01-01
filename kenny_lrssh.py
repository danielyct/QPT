from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d # Hilbert space spin basis
import numpy as np

def psi(tc,td,L,U,V,n):
    basis = spinless_fermion_basis_1d( L=L,Nf=L//2 ) 
    a = [ [(-1-n),i,(i+1)%L] for i in range(0,L,2)]
    aa = [ [(1+n),i,(i+1)%L] for i in range(0,L,2)]
    b = [ [(-1+n),(i+1)%L,(i+2)%L] for i in range(0,L,2)]
    bb = [ [(1-n),(i+1)%L,(i+2)%L] for i in range(0,L,2)]
    c = [ [tc,i,(i+3)%L] for i in range(0,L,2)]
    cc = [ [-tc,i,(i+3)%L] for i in range(0,L,2)]
    d = [ [td,(i+1)%L,(i+4)%L] for i in range(0,L,2)]
    dd = [ [-td,(i+1)%L,(i+4)%L] for i in range(0,L,2)]
    U_field = [ [U,i,(i+1)%L] for i in range(0,L,2)]
    V_field = [ [V,(i+1)%L,(i+2)%L] for i in range(0,L,2)]
    static = [ ["+-",a],["-+",aa],["+-",b],["-+",bb],["+-",c],["-+",cc],["+-",d],["-+",dd],["nn",U_field],["nn",V_field] ]
    dynamic = []
    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
    H = hamiltonian(static,dynamic,dtype=np.float64,basis=basis,**no_checks)
    value, vector = H.eigsh(k=1,which='SA')
    return value, vector[:,0]