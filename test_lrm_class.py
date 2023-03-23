import os 
os.chdir('/home/sdoshi/github/lrm')
print(os.getcwd())

from lrm import LRM
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from util import hdf5_to_dict

import IPython

lrm = LRM()

numberImages = 60
# gain, exposure_us = lrm.get_brightfield_autoexposure_controls()
gain = 2.0
exposure_us = int(1e6)
threshold=0.0
dataDir = '/home/sdoshi/github/lrm/data'
filePrefix = 'beta_waters_exp60s_'
exp_dict = lrm.capture_beta(numberImages, gain, exposure_us, threshold=threshold, dataDir=dataDir, filePrefix=filePrefix, save=True)
# exp_dict = lrm.capture_brightfield(numberImages, gain, exposure_us, dataDir=dataDir, filePrefix=filePrefix, save=True)

img, metadata = exp_dict['img']
print(metadata['ExposureTime'])
print(exposure_us)
print(gain)
print(metadata['AnalogueGain'])

# lrm.plot_brightfield_arr(img)
lrm.plot_beta_arr(img)
lrm.plot_beta_arr_log(img)
# IPython.embed()

# NOTE - when you load from an hdf5 file (using read_direct), it will return arrays of type np.float64 (correct numerical value, but not ints)
# exp_dict = hdf5_to_dict('/home/sdoshi/github/lrm/data/test_img2023-03-20-15-24-10.hdf5')
