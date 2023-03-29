import os 
os.chdir('/home/sdoshi/github/lrm')
print(os.getcwd())

from lrm import LRM
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from util import hdf5_to_dict, new_timestamp

import IPython

lrm = LRM()

# NOTE - going for 60s exposure here - start off by checking that 1s exposure doesn't saturate the camera
numberImages = 60
# gain, exposure_us = lrm.get_brightfield_autoexposure_controls()
gain = 1.0
exposure_us = int(1e6)
threshold=10.0
dataDir = '/home/sdoshi/github/lrm/data'
filePrefix = 'camerav3noir_c14urea_beta_60s' + '_' + new_timestamp()
exp_dict = lrm.capture_beta(numberImages, gain, exposure_us, threshold=threshold, dataDir=dataDir, filePrefix=filePrefix, save=True)
# exp_dict = lrm.capture_brightfield(numberImages, gain, exposure_us, dataDir=dataDir, filePrefix=filePrefix, save=True)
'''
Dark count looks 10x higher when room lights are on
43766703
426591870
416784043
245119707
18606329

For a 1s exposure, ignoring "hot" pixels, it looks like about ~1-15 dark counts per px - will use flatfield to correct for hot pixels
Gonna use a threshold of 10px for now, and will ignore the two hotspots that I see
'''

img, metadata = exp_dict['img']
print(metadata['ExposureTime'])
print(exposure_us)
print(gain)
print(metadata['AnalogueGain'])

# lrm.plot_brightfield_arr(img)
lrm.plot_beta_arr(img)
lrm.plot_beta_arr_log(img+1) # add 1 to avoid log(0) issue
# IPython.embed()

# NOTE - when you load from an hdf5 file (using read_direct), it will return arrays of type np.float64 (correct numerical value, but not ints)
# exp_dict = hdf5_to_dict('/home/sdoshi/github/lrm/data/test_img2023-03-20-15-24-10.hdf5')
