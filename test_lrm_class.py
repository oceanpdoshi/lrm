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
filePrefix = 'test_img'
exp_dict = lrm.capture_brightfield(numberImages, gain, exposure_us, dataDir, filePrefix, save=True)

img, metadata = exp_dict['img1']
print(metadata['ExposureTime'])
print(exposure_us)
print(gain)
print(metadata['AnalogueGain'])

plt.imshow(img)
plt.show()

exp_dict = hdf5_to_dict('/home/sdoshi/github/lrm/data/test_img2023-03-20-15-24-10.hdf5')

IPython.embed()