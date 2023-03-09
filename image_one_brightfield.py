'''image_one_brightfield.py
  
Protocol to:
	capture 1 brightfield image
'''
from lrm import LRM
import time


# File-related Settings
baseDir = './data/'
experimentDir='one-brightfield'
experimentDataPath = baseDir + experimentDir
bfFileName = 'bf-0'

# Initialize LRM class
LRM = LRM()
  
# Get exposure settings for brightfield images and lock them in
bfGain,bfShutter = LRM.get_brightfield_exposure()

# Override brightfield shutter duration if so desired
bfShutter=10000

# Announce
print("Capturing single brightfield image. Shutter = " + str(bfShutter) + ' ms')

# Capture brightfield 
LRM.capture_brightfield(experimentDataPath,bfFileName, 1, bfGain, bfShutter)
LRM.preview_last(True)
