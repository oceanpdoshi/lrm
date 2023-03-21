import os 
os.chdir('/home/sdoshi/github/lrm')
print(os.getcwd())

from lrm import LRM
import matplotlib.pyplot as plt
from util import hdf5_to_dict

import IPython

lrm = LRM()

numberImages = 2
gain, exposure_us = lrm.get_brightfield_autoexposure_controls()
dataDir = '/home/sdoshi/github/lrm/data'
filePrefix = 'test_beta'
# exp_dict = lrm.capture_beta(numberImages, gain, exposure_us, threshold=0.0, dataDir=dataDir, filePrefix=filePrefix, save=True)
exp_dict = lrm.capture_brightfield(numberImages, gain, exposure_us, dataDir=dataDir, filePrefix=filePrefix, save=True)

img, metadata = exp_dict['img1']
print(metadata['ExposureTime'])
print(exposure_us)
print(gain)
print(metadata['AnalogueGain'])

lrm.plot_brightfield_arr(img)

# IPython.embed()

# NOTE - when you load from an hdf5 file (using read_direct), it will return arrays of type np.float64 (correct numerical value, but not ints)
# exp_dict = hdf5_to_dict('/home/sdoshi/github/lrm/data/test_img2023-03-20-15-24-10.hdf5')
