import matplotlib.pyplot as plt
import numpy as np
import os 
os.chdir('/home/sdoshi/github/lrm')
print(os.getcwd())

total_counts = np.load('/home/sdoshi/github/lrm/data/dark_count_sweep2.npy')

# NOTE - sweep 1
# gain = [2.0, 4.0, 6.0, 8.0]
# exposure_us = list(10**np.array([2,3,4,5,6])) # 100-1,000,000us (100us-1s) exposure times
# threshold = 0.0
# N_points = 5

# NOTE - sweep 2 - lens off
gain = [1.0, 2.0, 3.0, 4.0]
exposure_us = list(10**np.array([2,3,4,5,6])) # 100-1,000,000us (100us-1s) exposure times
threshold = 0.0
N_points = 5

# Compute mean and standard deviation, min and max
total_counts_mean = np.mean(total_counts, axis=2)
total_counts_std = np.std(total_counts, axis=2)

# error_bars = np.zeros((2, len(total_counts_min)))
# error_bars[1, :] = total_counts_min
# error_bars[2,: ] = total_counts_max



fig = plt.figure()
for i, g in enumerate(gain):
    plt.errorbar(exposure_us, total_counts_mean[i], yerr=total_counts_std[i], label='g=%.1f' % g)

ax = fig.axes[0]
ax.set_xscale('log')
plt.title("Dark count - lens off")
plt.ylabel("Total Camera Counts")
plt.xlabel("Exposure Time (us)")
plt.grid(visible=True, which='both', axis='x')
plt.legend()
plt.show()