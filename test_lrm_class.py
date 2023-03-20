import os 
os.chdir('/home/sdoshi/github/lrm')
print(os.getcwd())

from lrm import LRM

import IPython

lrm = LRM()

numberImages = 2
gain, exposure_us = lrm.get_brightfield_autoexposure_controls()
dataDir = '/home/sdoshi/github/lrm/data'
filePrefix = 'test_img'
lrm.capture_brightfield(numberImages, gain, exposure_us, dataDir, filePrefix, save=True)

IPython.embed()