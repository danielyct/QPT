import numpy as np # generic math functions
import matplotlib.pyplot as plt # plot fidelity graph
size = 10
delta = 0.2

fid = []
x_list = np.linspace(0,2)
x = []
F = []
sus = []
recip_size = []
min_fid_h_pos = []
ln_size = []
ln_max_sus = []

reg1 = reg2 = reg3 = reg4 = []


# plot setting
fig, ax = plt.subplots(3)
ax[0].set_xlabel('magnetic strength')
ax[0].set_title("size = " + str(size) + ", delta = " + str(delta))

# first graph
# plot fidelity
ax[0].set_ylabel('fidelity', color = 'tab:red')
ax[0].plot(x,F,label = 'Fidelity', color = 'tab:red')
ax[0].tick_params(axis = 'y', labelcolor = 'tab:red')

# plot susceptibility
axm = ax[0].twinx()
axm.set_ylabel('susceptibility', color = 'tab:blue')
axm.plot(x,sus, label = 'Susceptibility', color = 'tab:blue')
axm.tick_params(axis = 'y', labelcolor = 'tab:blue')

# second graph
# plot field strength of max susceptibility 

ax[1].set_xlabel("1/N")
ax[1].set_ylabel("fidelity")
ax[1].set_title("min_fid_h_pos")
ax[1].plot(recip_size, min_fid_h_pos, label = "true")
ax[1].plot(recip_size, reg1, label = "1st degree correction")
ax[1].plot(recip_size, reg2, label = "2nd degree correction")
ax[1].plot(recip_size, reg3, label = "3rd degree correction")
ax[1].plot(recip_size, reg4, label = "4th degree correction")
ax[1].legend()

# third graph
ax[2].set_title("max susceptibility")
ax[2].set_xlabel("ln N")
ax[2].set_ylabel(r"$ln\chi_F$")
ax[2].plot(ln_size, ln_max_sus)

sizes = range(6,13)

i = 0
color_list = color_list = ['#0000FF', # blue
              '#00B050', # green
              '#A81551', # burgundy
              '#FFC000', # orange yellow
             ]
fig5, ax5 = plt.subplots()
ax5.set_title("All fidelity")
ax5.set_ylabel("Fidelity")
ax5.set_xlabel("magnetic strength")
for size in range(len(sizes)):
    if size % 2 == 1:
        continue
    i = i + 1
    ax5.plot(x, fid[size], label = "size = " + str(size))
 

plt.show()