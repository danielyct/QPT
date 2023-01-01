# Creator: Daniel Yu
# Reference: Kenny Hui and example from Quspin

import os
import time
import sys
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plot fidelity graph
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import ising_model as ising
import ssh_model as ssh
import long_range_ssh as lrssh

os.environ['OMP_NUM_THREADS'] = '8' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS'] = '8' # set number of MKL threads to run in parallel
os.environ['MKL_DEBUG_CPU_TYPE'] = '5' # utilize AVX2

# start timer
start = time.perf_counter()

# initial variables
model = lrssh
step = 2
start_size = 4
end_size = 8
end_size += 1
sizes = range(start_size,end_size,step)
sample_no = 100
delta = 0.1
J = 1.0

column = sample_no
row = len(sizes)
fid = []
sus = []

ln_size = []
recip_size = []

min_fid = []
min_fid_pos = []

max_sus =  []
max_sus_pos = []
max_sus_per_site = []
ln_max_sus = []

def find_sus(fid, delta):
    return -2 * np.log(fid) / delta**2

# find fidelity, return fidelity, susceptibility, x-axis
def find_fid(size = 6, delta = 0.01, **kwargs):
    h_list = np.linspace(-7, 5, sample_no)
    F = []
    sus = []
    for i,h in enumerate(h_list):
        E_1, s_1 = model.states(size, U = h, **kwargs)
        E_2, s_2 = model.states(size, U = h+delta, **kwargs)
        # E_1, s_1 = model.states(size,h)
        # E_2, s_2 = model.states(size,h+delta)
        fidelity = abs(np.dot(s_1,s_2))
        F.append(fidelity)
        sus.append(find_sus(fidelity,delta))
        print("size: " + str(size) + "  sample no.: ", str(i))
    
    file = 'data/main_' + str(size) 

    np.save(file + 'F.npy', F)
    np.save(file + 'x.npy', h_list)

    return F, sus, h_list

# regression function, return regression result, intercept, coefficient of determination
def reg(x, y, degree):
    moded_x = PolynomialFeatures(degree = degree, include_bias=False).fit_transform(x)
    model = LinearRegression().fit(moded_x, y)
    err = model.score(moded_x,y)
    intercept = model.intercept_
    result = model.predict(moded_x)
    return result, intercept, err

# looping for different size of lattice
for size in sizes:
    F, S, x = find_fid(size = size, delta = delta, eta = 0.6, V = -2, c = 0, d = 0)
    # F, S, x = find_fid(size = size, delta = delta)
    recip_size.append(1/size)
    ln_size.append(np.log(size))

    fid.append(F)
    sus.append(S)

    min_f = np.amin(F)
    min_fid.append(min_f)
    print(F)
    index_maxima = int(np.where(F == min_f)[0])
    min_fid_pos.append(x[index_maxima])

    max_s = np.amax(S)
    max_sus.append(max_s)
    max_sus_pos.append(x[index_maxima])
    max_sus_per_site.append(max_s/size)
    ln_max_sus.append(np.log(max_s))
    
# modifing data
np.reshape(fid, (row, column))
np.reshape(sus, (row, column))
recip_size = np.array(recip_size).reshape(-1, 1)
min_fid_pos = np.array(min_fid_pos)

# pluging results into regression with different degree
reg1, inter1, det1 = reg(recip_size, min_fid_pos, 1)
reg2, inter2, det2 = reg(recip_size, min_fid_pos, 2)
reg3, inter3, det3 = reg(recip_size, min_fid_pos, 3)
reg4, inter4, det4 = reg(recip_size, min_fid_pos, 4)

a_reg1, a_inter1, a_det1 = reg(recip_size, max_sus_per_site, 1)
a_reg2, a_inter2, a_det2 = reg(recip_size, max_sus_per_site, 2)
a_reg3, a_inter3, a_det3 = reg(recip_size, max_sus_per_site, 3)

# # First graph
# # plot setting
# fig, ax = plt.subplots()
# ax.set_xlabel('h')
# ax.set_title("Fidelity and susceptibility ")

# # plot fidelity
# ax.set_ylabel('fidelity', color = 'tab:red')
# ax.plot(x, F, label = 'Fidelity', color = 'tab:red')
# ax.tick_params(axis = 'y', labelcolor = 'tab:red')

# # plot susceptibility
# ax1 = ax.twinx()
# ax1.set_ylabel('susceptibility', color = 'tab:blue')
# ax1.plot(x, S, label = 'Susceptibility', color = 'tab:blue')
# ax1.tick_params(axis = 'y', labelcolor = 'tab:blue')

# # Second graph
# # plot field strength of max susceptibility 
# sfig2, ax2 = plt.subplots()
# ax2.set_title("critical point vs 1/N")
# ax2.set_xlabel("1/N")
# ax2.set_ylabel("h at criticial point")
# ax2.plot(recip_size, min_fid_pos, '-x', label = "true")
# ax2.plot(recip_size, reg1, label = "1st degree regression")
# ax2.plot(recip_size, reg2, label = "2nd degree regression")
# ax2.plot(recip_size, reg3, label = "3rd degree regression")
# # ax2.plot(recip_size, reg4, label = "4th degree correction")
# ax2.legend()

# # Third graph
# # ln max susceptibility vs ln systme size
# fig3, ax3 = plt.subplots()
# ax3.set_title(r"$ln \chi_F vs ln N$")
# ax3.set_xlabel("ln N")
# ax3.set_ylabel(r"$ln\chi_F$")
# ax3.plot(ln_size, ln_max_sus, '-x')

# # Forth graph
# # max susceptibility per site vs 1/size
# fig4, ax4 = plt.subplots()
# ax4.set_title("h at critical point vs 1/size")
# ax4.set_ylabel("h at critical point")
# ax4.set_xlabel("1/N")
# ax4.plot(sizes, max_sus_pos, '-x')    # max_sus_list / N

# Fifth graph
# All fidelity graph
fig5, ax5 = plt.subplots()
ax5.set_title("All fidelity")
ax5.set_ylabel("Fidelity")
ax5.set_xlabel('h')
for i, size in enumerate(sizes):
    ax5.plot(x, fid[i], label = "unit cell = " + str(size))
ax5.legend()

# # Sixth graph
# # All susceptibility graph
# fig6, ax6 = plt.subplots()
# ax6.set_title("All susceptibility")
# ax6.set_ylabel("susceptibility")
# ax6.set_xlabel("h")
# for j, size in enumerate(sizes):
#     ax6.plot(x, sus[j], label = "size = " + str(size))
# ax6.legend()

# # Seventh graph
# fig7, ax7 = plt.subplots()
# ax7.set_title(r"$\chi_F vs 1/N$")
# ax7.set_ylabel(r"$\chi_F / N$")
# ax7.set_xlabel("1/N")
# ax7.plot(recip_size, max_sus_per_site, '-x', label = "true")
# ax7.plot(recip_size, a_reg1, label = "1st degree regression")
# ax7.plot(recip_size, a_reg2, label = "2nd degree regression")
# ax7.plot(recip_size, a_reg3, label = "3rd degree regression")
# ax7.legend()

# # print out important data
# print("coefficient of determination(1): " + str(det1) + " intercept" + str(inter1))
# print("coefficient of determination(2): " + str(det2) + " intercept" + str(inter2))
# print("coefficient of determination(3): " + str(det3) + " intercept" + str(inter3))
# print("coefficient of determination(4): " + str(det4) + " intercept" + str(inter4))

# print(a_inter1)
# print(a_inter2)
# print(a_inter3)

# stop timer and display execution time
end = time.perf_counter()
print(f"finished in {end - start:0.4f} seconds")

# show graphs, must be after stop timer or it time of showing graphs will be included
plt.show()
