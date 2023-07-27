import os 
os.chdir('/home/sdoshi/github/lrm')
print(os.getcwd())

from lrm import LRM
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm
from util import hdf5_to_dict, new_timestamp

import IPython

lrm = LRM()

# NOTE - going for 60s exposure here - start off by checking that 1s exposure doesn't saturate the camera
numberImages = 1
# gain, exposure_us = lrm.get_brightfield_autoexposure_controls()
gain = 1.0
exposure_us = int(1e6)
threshold=0.0
fpn_correct = '/home/sdoshi/github/lrm/fpn_correction_1s_lightson.npy'
dataDir = '/home/sdoshi/github/lrm/data'
filePrefix = new_timestamp() + '_' + 'camerav3noir_1s_testshield'
exp_dict = lrm.capture_beta(numberImages, gain, exposure_us, threshold=threshold, fpn_correct=fpn_correct, dataDir=dataDir, filePrefix=filePrefix, save=False)
# exp_dict = lrm.capture_brightfield(numberImages, gain, exposure_us, dataDir=dataDir, filePrefix=filePrefix, save=True)

img, metadata = exp_dict['img']
print(metadata['ExposureTime'])
print(exposure_us)
print(gain)
print(metadata['AnalogueGain'])
print(np.sum(img))

# lrm.plot_brightfield_arr(img)
lrm.plot_beta_arr(img)
# lrm.plot_beta_arr_log(img+1) # add 1 to avoid log(0) issue
# IPython.embed()

# NOTE - when you load from an hdf5 file (using read_direct), it will return arrays of type np.float64 (correct numerical value, but not ints)
# exp_dict = hdf5_to_dict('/home/sdoshi/github/lrm/data/test_img2023-03-20-15-24-10.hdf5')
