import numpy as np
import matplotlib.pyplot as plt 
import sys

file = sys.argv[1]

F_filename = file + "F.npy"
x_filename = file + "x.npy"

F = np.load(F_filename)
x = np.load(x_filename)

fig, ax = plt.subplots()
ax.set_title("Fidelity")
ax.set_xlabel("dp")
ax.set_ylabel("fidelity")
ax.plot(x,F)

plt.show()