from lrm import LRM
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from util import hdf5_to_dict

import IPython

path = '/home/sdoshi/github/lrm/data/bf_air22023-03-23-17-32-51.hdf5'

exp_dict = hdf5_to_dict(path)
img, metadata = exp_dict['img1']
print(metadata['ExposureTime'])
print(metadata['AnalogueGain'])

lrm = LRM()
lrm.plot_brightfield_arr(img)
# lrm.plot_beta_arr(img)
# lrm.plot_beta_arr_log(img)