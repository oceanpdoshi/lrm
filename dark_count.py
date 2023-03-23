import numpy as np
import os 
os.chdir('/home/sdoshi/github/lrm')
print(os.getcwd())

from lrm import LRM
import matplotlib.pyplot as plt
from util import hdf5_to_dict

import IPython

lrm = LRM()

numberImages = 1
gain = [1.0, 2.0, 3.0, 4.0]
exposure_us = list(10**np.array([2,3,4,5,6])) # 100-1,000,000us (100us-1s) exposure times
threshold = 0.0
N_points = 5
total_counts = np.zeros((len(gain), len(exposure_us), N_points))


dataDir = '/home/sdoshi/github/lrm/data/'
filePrefix = 'dark_count_sweep2'

for i,g in enumerate(gain):
    for j,e_us in enumerate(exposure_us):
        for k in range(N_points):
            exp_dict = lrm.capture_beta(numberImages, g, e_us, threshold=threshold, dataDir=dataDir, filePrefix=filePrefix, save=False)
            img, metadata = exp_dict['img']
            total_counts[i][j][k] = np.sum(img)

savepath = dataDir + filePrefix + '.npy'
np.save(savepath, total_counts)

IPython.embed()
