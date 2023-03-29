import numpy as np
import os 
os.chdir('/home/sdoshi/github/lrm')
print(os.getcwd())

from lrm import LRM
import matplotlib.pyplot as plt
from util import hdf5_to_dict, new_timestamp

import IPython

lrm = LRM()

numberImages = 1
gain = [1.0, 2.0, 3.0]
exposure_us = [200, 1000, 10000, 100000, 1000000] # 100-1,000,000us (100us-1s) exposure times (camera limits are 26 to 1722331)
threshold = 0.0
N_points = 5
total_counts = np.zeros((len(gain), len(exposure_us), N_points))

dataDir = '/home/sdoshi/github/lrm/data/'
filePrefix = 'dark_count_sweep_' + new_timestamp()

# TODO - for future dark count characterization (if needed) can use tqdm to track progress
iteration = 1
for i,g in enumerate(gain):
    for j,e_us in enumerate(exposure_us):
        for k in range(N_points):
            exp_dict = lrm.capture_beta(numberImages, g, e_us, threshold=threshold, dataDir=dataDir, filePrefix=filePrefix, save=False)
            img, metadata = exp_dict['img']
            total_counts[i][j][k] = np.sum(img)
            print('iteration ' + str(iteration) + '/' + str(len(gain)*len(exposure_us)*N_points))
            iteration += 1

savepath = dataDir + filePrefix + '.npy'

try:
    np.save(savepath, total_counts)
except:
    pass

IPython.embed()
