from lrm import LRM
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import numpy as np
from util import hdf5_to_dict

import IPython

path = '/home/sdoshi/github/lrm/data/camerav3noir_c14urea_beta_60s_2023-03-28-20-04-472023-03-28-20-07-52.hdf5'

exp_dict = hdf5_to_dict(path)
img, metadata = exp_dict['img']
print(metadata['ExposureTime'])
print(metadata['AnalogueGain'])

print(np.sum(img))
'''
173842.0
179806.0
'''

lrm = LRM()
# lrm.plot_brightfield_arr(img)
lrm.plot_beta_arr(img)
lrm.plot_beta_arr_log(img+1)