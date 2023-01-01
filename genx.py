import numpy as np # generic math functions
import sys

file = sys.argv[1]
lowest = float(sys.argv[2])
highest = float(sys.argv[3])
sample_no = int(sys.argv[4])

x = np.linspace(lowest, highest, sample_no)

np.save(file + 'x.npy', x)
