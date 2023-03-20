'''
Utility functions for saving and loading experiment data
'''

import os
import glob
import h5py
import numpy as np
from datetime import datetime

from pint import UnitRegistry
u = UnitRegistry()
Q_ = u.Quantity

TIMESTAMP_FORMAT = '%Y-%m-%d-%H-%M-%S'  # (YYYY-MM-DD-HH-MM-SS)

def new_timestamp():
    '''Returns current timestamp (str) in %Y-%m-%d-%H-%M-%S format (YYYY-MM-DD-HH-MM-SS)'''
    return datetime.strftime(datetime.now(),TIMESTAMP_FORMAT)

def is_within_percent(a, b, percent):
    """ Return true if a is percent within b """
    diff = abs(a - b)

    if a == 0:
        p = (diff) * 100

    else:
        p = (diff / a) * 100.

    within = p < percent
    return within

def append_slash(path):
    """ Append a slash to path if it doesn't already end with one """
    if not (path[-1] == '/'):
        path += '/'

    return path

def _get_parent_directory(path):
    """ Get the parent directory of a given filepath. If path is a dir, just return path, but with a / at the end"""
    # Check that file extension is ".xxx" and that path doesn't end with '/'
    isFilePath = path[-4] == '.' and not (path[-1] == '/')

    if isFilePath:
        directory = path[: path.rfind('/') + 1]
        return directory
    else:
        return append_slash(path)

def check_or_make_directory(path):
    """ Check for or make path to data sub directory """

    path = _get_parent_directory(path)
    directory = os.path.dirname(path)
    dirExists = os.path.exists(directory)

    if not dirExists:
        os.makedirs(directory)

def newest_subdir(data_dir,filter="*"):
    '''
    Returns the most recently created subdirectory within data_dir that matches UNIX pattern filter
        Parameters
            data_dir (str) : parent directory to search within for subdirs
            filter (str)   : UNIX pattern matching expression to search for subdirs
            
        Returns
            subdir (str)   : most recently created subdirectory that matches filter criteria
    '''
    subdirs = [dd for dd in glob.glob(os.path.join(data_dir,filter)) if os.path.isdir(dd)]
    if subdirs:
        return max(subdirs, key=os.path.getctime)
    else:
        return None

def newest_file(data_dir,filter="*"):
    '''
    Returns the most recently created file within data_dir that matches UNIX pattern filter
        Parameters
            data_dir (str) : parent directory to search within for files
            filter (str)   : UNIX pattern matching expression to search for files
            
        Returns
            file (str)   : most recently created file that matches filter criteria
    '''
    files = [ff for ff in glob.glob(os.path.join(data_dir,filter)) if os.path.isfile(ff)]
    if files:
        return max(files, key=os.path.getctime)
    else:
        return None

def new_path(data_dir, name, extension='',timestamp=True):
    '''
    Creates path string by joining in order [data_dir, ds_type, name, timestamp_string]
        Parameters
            data_dir (str)   : parent directory to store file
            name (str)       : filename
            extension (str)  : file extension ('.h5')
            timestamp (bool) : whether or not to include timestamp
            
        Returns
            full_path (str)  : filepath created by joining [data_dir, ds_type, name, timestamp_string]
    '''

    if timestamp:
        timestamp_string = new_timestamp()
    else:
        timestamp_string = None
    if extension and extension[0]!='.':
        extension = '.' + extension
    name_parts = [name, timestamp_string]
    full_name = '_'.join([part for part in name_parts if part is not None]) + extension
    full_path = os.path.normpath(os.path.join(data_dir,full_name))
    return full_path


# HDF5 utilities for images and their metadata
# leaving in commented out code in case future io code needs to be generalized past multiple images
def dump_to_hdf5(exp_dict, fpath, open_mode='a'):
    '''
    Creates hdf5 file at fpath from given exp_dict.
    images <-> dsets, metadata <-> attributes
        Parameters
            exp_dict (dict)  : dictionary to save as an hdf5 file
            fpath (str)      : file path
            open_mode (str)  : standard Python File I/O modes ('r', 'w', 'a', etc.)
            
        Returns
            None
            
    '''
    with h5py.File(fpath, open_mode) as f:

        for k,v in exp_dict.items():
            data, metadata = v
            dset = f.create_dataset(k,
                            data.shape,
                            dtype=data.dtype,
                            data=data,
                            compression='gzip')
            for km, vm in metadata.items():
                dset.attrs[km] = vm
        
        f.flush()

    return

def hdf5_to_dict(fpath):
    '''
    Recurisvely iterate through groups of hdf5 file.
    Groups <-> dictionaries, datasets <-> arrays/constants, attributes <-> units
    '''

    print("loading file: " + fpath)
    exp_dict = {}
    with h5py.File(fpath, "r") as f:
        # Note that file object "f" is also the root group object 
        for k in f.keys():
            if isinstance(f[k], h5py.Dataset):
                exp_dict[k] = _load_dataset(f[k])
            elif isinstance(f[k], h5py.Group):
                exp_dict[k] = _hdf5_to_dict(f[k])
        f.close()

    return exp_dict

def _load_dataset(dset):
    '''
    Subfunction to handle attributes <-> units
    '''
    # scalars
    # if dset.shape == ():
    #     dset_scalar = dset[()]
    #     if list(dset.attrs) != []:
    #         dset_scalar = u.Quantity(dset_scalar, dset.attrs['units'])
    #     return dset_scalar
    
    # numpy.ndarray
    dset_arr = np.zeros(dset.shape)
    dset.read_direct(dset_arr)
    # Check if there are units
    # if list(dset.attrs) != []:
    #     dset_arr = u.Quantity(dset_arr, dset.attrs['units'])
    metadata = {}
    for k,v in dset.attrs.items():
        metadata[k] = v
    return (dset_arr, metadata)

def _hdf5_to_dict(grp):
    '''
    Recursive inner function (doesn't have file io)
    '''
    exp_dict = {}
    for k in grp.keys():
        if isinstance(grp[k], h5py.Dataset):
            exp_dict[k] = _load_dataset(grp[k])
        elif isinstance(grp[k], h5py.Group):
            exp_dict[k] = _hdf5_to_dict(grp[k])
    return exp_dict 