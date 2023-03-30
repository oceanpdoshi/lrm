import os 
os.chdir('/home/sdoshi/github/lrm')
print(os.getcwd())

from lrm import LRM
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm
from pprint import *
from util import hdf5_to_dict, new_timestamp

import IPython

'''
dark counts (np.sum(img))) for a 1s exposure at 1x analog gain - 03-30-2023
Room Lights on
1985084
1895626
1935937

Room lights on 
1727156
1721377
1729981

~10% fewer dark couts with room lights on


'''

lrm = LRM()
numberImages = 100
gain = 1.0
exposure_us = int(1e6)
threshold=0.0
dataDir = '/home/sdoshi/github/lrm/data'
filePrefix = new_timestamp() + '_' + 'camerav3noir_1s_100imgseries'
exp_dict = lrm.capture_beta_series(numberImages, gain, exposure_us, threshold=threshold, dataDir=dataDir, filePrefix=filePrefix, save=False)

img_list = []
total_counts_list = []

for k,v in exp_dict.items():
    img, metadata = v
    img_list.append(img)
    total_counts_list.append(np.sum(img))

# Compute fixed pattern noise correction
img_stack = np.stack(img_list, axis=2)
print(img_stack.shape)
img_mean = np.mean(img_stack, axis=2)
print(img_mean.shape)
img_std = np.std(img_stack, axis=2)
print(img_std.shape)

# Compute total counts statistics
total_counts_list = np.array(total_counts_list)
total_mean = np.mean(total_counts_list)
total_std = np.std(total_counts_list)

savepath = '/home/sdoshi/github/lrm/fpn_correction_1s_lightson.npy'
np.save(savepath, img_mean)

print("Number of Images: " + str(numberImages))
print("gain=%.2f, "%gain + "exposure=%d[us], "%exposure_us + "threshold=%.1f"%threshold)
print("Average total image counts: " + str(total_mean))
print("Standard deviation of total image counts: " + str(total_std))

plt.figure(1)
plt.title('Averaged FPN')
plt.imshow(img_mean, cmap='inferno', norm=Normalize()) # this should be the same as the above 2 (commented out) lines of code
plt.colorbar()

plt.figure(2)
plt.title('FPN variation')
plt.imshow(img_std, cmap='inferno', norm=Normalize()) # this should be the same as the above 2 (commented out) lines of code
plt.colorbar()

plt.show()
