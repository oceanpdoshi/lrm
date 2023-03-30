''' LRM.py
Defines LRM class that contains all beta microscope functionality
'''

from libcamera import controls as libcontrols # for enums
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import numpy as np
from picamera2 import Picamera2
from PIL import Image
from pprint import *
import time
from gpiozero import LED

from util import *

# TODO - update all the docstrings and delete the deprecated code
# TODO - make class functions to print these
# Useful stuff to print before modifying any configurations/controls
# pprint(picam2.sensor_modes)
# pprint(picam2.camera_controls)
# pprint(picam2.camera_properties)

# Constants
LED_PIN = 21

'''V2 resolutions (3280,2464) (1920,1080) (1640,1232) (1640,922) (1280,720) (640,480)'''
'''V3 resolutions (4608, 2592) (2304, 1296) (1536, 864)'''
# Main stream configuration
DEFAULT_RESOLUTION = (4608, 2592)
DEFAULT_STREAM_FORMAT = 'BGR888' # see manual

# Raw stream configuration
DEFAULT_SENSOR_MODE = 2 # 10 bit, (4608, 2592), SRGGB10_CSI2P format, (26, 1722331)us  exposure limits, 14.35fps

# Default ColourGain Setting (camera controls) - disables awb when set
DEFAULT_AWB_GAINS = (0.0/8.0*32.0, 0.0/8.0*32.0)

# From the paper - default gain was 2.5 
DEFAULT_BETA_GAIN = 2.5
DEFAULT_BETA_EXPOSURE = int(5e6) # this might have to be an integer (picam2.camera_controls[ExposureTime] limits are ints)

# Percent differece for ExposureTime and AnalogueGain for __check_camera()
EXPOSURE_TIME_THRESHOLD = 10
GAIN_THRESHOLD = 25
SETTLED_PERCENT = 2.5


class LRM:
    """Class containing all beta microscope functionality - see fuction docstrings."""

    def __init__(self):

        # Most recent analog gain setting and exposure setting (us)
        self._gainSet = None
        self._exposureSet = None

        self.info = {}

        self.led = LED(LED_PIN)
        self.led.off()

        # Configure (Chapter 4, Appendix B) and start() the camera
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(main={"size" : DEFAULT_RESOLUTION, "format" : DEFAULT_STREAM_FORMAT})
        self.picam2.configure(config)
        self.picam2.start() # Uses Preview.NULL by default
        time.sleep(2)

    def __setup_beta_controls(self, gain, exposure_us, ColourGains=DEFAULT_AWB_GAINS):
        """Set the camera controls to those optimized for beta imaging.
        gain: analog gain
        exposure_us : exposure time in us
        """

        # Copy provided values into LRM gain and shutter setpoint variables
        self._gainSet = gain
        self._exposureSet = exposure_us

        # any paramter not commented is just set to default value
        # all af (autofocus values are set to 0 - most importatly AfMode 0=manual, so no autofocusing)
        controls = {
            'AeConstraintMode': 0,
            'AeEnable': False, # don't want Autoexposure
            'AeExposureMode': 0,
            'AeMeteringMode': 0,
            'AnalogueGain': gain,
            'AwbEnable': False, # Don't want autowhite balances, this is also automatically disabled by setting ColourGains
            'AwbMode': 0,
            'Brightness': 0.0,
            # 'ColourCorrectionMatrix': (-16.0, 16.0, None), read-only, see manual (Appendix C)
            'ColourGains': ColourGains, # See manual (Appendix C)
            'Contrast': 1.0,
            'ExposureTime': exposure_us,
            'ExposureValue': 0.0,
            # 'FrameDurationLimits': (47183, 11767556, None), set by camera_config 
            'NoiseReductionMode': 0, # no noise reduction wanted
            'Saturation': 1.0,
            'ScalerCrop': (0, 0, 3280, 2464),
            'Sharpness': 1.0
        }
        self.picam2.set_controls(controls)

        # wait for automatic gain/shutter adjustment
        time.sleep(3)

    def __setup_brightfield_controls(self, gain, exposure_us):
        """Start up the camera with settings optimized for brightfield imaging. (Same as beta, but ColourGains 1,1 -> 1,2).
        gain: analog gain
        exposure_us : exposure time in us
        """

        ColourGains = DEFAULT_AWB_GAINS

        self.__setup_beta_controls(gain, exposure_us, ColourGains=ColourGains)

    def __check_camera(self):
        """ Check that the gain and exposure settings are at the desired setpoint using camera metadata."""
        metadata = self.picam2.capture_metadata()
        gain = metadata["AnalogueGain"]
        exposure_us = metadata["ExposureTime"]

        exposure_correct = is_within_percent(self._exposureSet, exposure_us, EXPOSURE_TIME_THRESHOLD)
        gain_correct = is_within_percent(self._gainSet, gain, GAIN_THRESHOLD)

        return exposure_correct and gain_correct

    def __reboot(self):
        """ Reboots the camera """

        self.picam2.stop()
        print("rebooting")
        config = self.picam2.create_still_configuration(main={"size" : DEFAULT_RESOLUTION, "format" : DEFAULT_STREAM_FORMAT})
        self.picam2.configure(config)
        self.picam2.start() # Uses Preview.NULL by default
        time.sleep(3)

    def __snap_beta(self, threshold=0.0):
        """
        Grabs an RGB frame (np.ndarrary) from camera (function assumes you are using beta imaging controls)
        Sums across the RGB channels and zeros out pixels less than threshold (default threshold=0 does no thresholding).
        returns: np.ndarray camera image, image metadata+capture time
        """

        while not self.__check_camera():
            metadata = self.picam2.capture_metadata()
            print("Gain was: " + str(metadata["AnalogueGain"]) + ", wanted " + str(self._gainSet))
            print("ExposureTime: " + str(metadata["ExposureTime"]) + " [us]" + ", wanted " + str(self._exposureSet) + " [us]")
            self.__reboot()

        # Capture image (np.ndarray) from the camera + record total capture time
        captureStartTime = time.time()
        request = self.picam2.capture_request("main")
        array, metadata = request.make_array("main"), request.get_metadata()
        request.release()
        captureTimeSeconds = time.time() - captureStartTime

        self.info = metadata
        self.info['capture_time_s'] = captureTimeSeconds
        
        # Sum across RGB Channels to generate one channel array
        # NOTE - Doing this instead of a weighted sum because we are treating values as "camera counts"
        array = array.astype(np.uint64)
        array = array.sum(axis=2) # this does automatic cast from uint8 to uint64 when numbers exceed 255, above line is to be explicit

        # Apply threshold
        array[array < threshold] = 0

        # Compute some image metrics and store in info
        self.info['image_sum'] = array.sum()
        self.info['image_std'] = np.std(array)
        self.info['image_max'] = array.max()
        self.info['image_min'] = array.min()
        self.info['image_mean'] = array.mean()

        return array, self.info

    def __snap_brightfield(self):
        """
        Grabs an RGB frame (np.ndarrary) from camera with brightfield imaging controls
        returns: (3D np.ndarray) camera image, (dict) image metadata+capturetime
        """

        if not self.__check_camera():
            metadata = self.picam2.capture_metadata()
            print("Gain was: " + str(metadata["AnalogueGain"]) + ", wanted " + str(self._gainSet))
            print("ExposureTime: " + str(metadata["ExposureTime"]) + " [us]" + ", wanted " + str(self._exposureSet) + " [us]")
            raise Exception("Camera gain or exposure time was not set correctly.")

        # Capture image (np.ndarray) from the camera + record total capture time
        captureStartTime = time.time()
        request = self.picam2.capture_request("main")
        array, metadata = request.make_array("main"), request.get_metadata()
        request.release()
        captureTimeSeconds = time.time() - captureStartTime

        self.info = metadata
        self.info['capture_time_s'] = captureTimeSeconds

        return array, self.info

    def capture_beta(self, numberImages, gain, exposure_us, threshold=0.0, fpn_correct='', dataDir=None, filePrefix=None, save=False):
        """Integrate numberImages, and save resulting summed image to hdf5 file (with metadata from last image)
        dataDir: save dir
        filePrefix: prefix of saved file
        numberImages: number of frames to integrate over
        gain: analog gain setting
        exposure_us: exposure time
        Returns: exp_dict = {"img" : (summed, metadata)}
        """

        # Make sure the LED is off just in case
        self.led.off()
        time.sleep(0.5)

        # Make destination directory if it doesn't exist
        check_or_make_directory(dataDir)

        # Set up camera for beta acquisition
        self.__setup_beta_controls(gain, exposure_us)

        # Announce
        metadata = self.picam2.capture_metadata()
        print(f"Capturing beta image(s): path = {dataDir} prefix = {filePrefix} # {numberImages}\n")
        print("Gain = " + str(metadata["AnalogueGain"]))
        print("Exposure (us) = " + str({metadata["ExposureTime"]}))

        if fpn_correct != '':
            fpn_correct_arr = np.load(fpn_correct)
            fpn_correct = True

        # NOTE - for now just saving last image's metadata
        infoList = []

        for i in range(numberImages):

            array, info = self.__snap_beta(threshold)
            if fpn_correct:
                array = array - fpn_correct
                array = np.multiply((array >= 0), array) # don't store negative array values
            infoList.append(info)

            # Sum image
            if i == 0:
                summed = array
            else:
                summed = summed + array

            print("Captured " + str(i+1) + "/" + str(numberImages) + " Duration = " + str(
                info['capture_time_s']) + " Sum = " + str(array.sum()) + " Total sum = " + str(summed.sum()))
                
        # Save data using h5py
        outFileName = append_slash(dataDir) + filePrefix + new_timestamp() + '.hdf5'
        exp_dict = {"img" : (summed, infoList[-1])}
        if save:
            dump_to_hdf5(exp_dict, outFileName)

        return exp_dict

    def capture_beta_series(self, numberImages, gain, exposure_us, threshold=0.0, dataDir=None, filePrefix=None, save=False):
        """Capture numberImages of beta images and save them to hdf5 file (with metadata from last image)
        dataDir: save dir
        filePrefix: prefix of saved file
        numberImages: number of frames to integrate over
        gain: analog gain setting
        exposure_us: exposure time
        Returns: exp_dict = {"img" : (summed, metadata)}
        """

        # Make sure the LED is off just in case
        self.led.off()
        time.sleep(0.5)

        # Make destination directory if it doesn't exist
        check_or_make_directory(dataDir)

        # Set up camera for beta acquisition
        self.__setup_beta_controls(gain, exposure_us)

        # Announce
        metadata = self.picam2.capture_metadata()
        print(f"Capturing beta image(s): path = {dataDir} prefix = {filePrefix} # {numberImages}\n")
        print("Gain = " + str(metadata["AnalogueGain"]))
        print("Exposure (us) = " + str({metadata["ExposureTime"]}))

        exp_dict = {}

        for i in range(numberImages):

            array, info = self.__snap_beta(threshold)
            exp_dict["img" + str(i+1)] = (array, info)

            print("Captured " + str(i+1) + "/" + str(numberImages) + " Duration = " + str(
                info['capture_time_s']) + " Sum = " + str(array.sum()))
                
        # Save data using h5py
        outFileName = append_slash(dataDir) + filePrefix + new_timestamp() + '.hdf5'
        if save:
            dump_to_hdf5(exp_dict, outFileName)

        return exp_dict

    def capture_brightfield(self, numberImages, gain, exposure_us, dataDir=None, filePrefix=None, save=False):
        """Capture one or more brightfield images and write to disk in an hdf5 file 
        dataDir: path to save brightfield images
        filePrefix: prefix of saved images 
        numberImages: number of frames to capture        
        gain: fixed analog gain setting; 
        exposure_us: exposure Time in us
        Returns: exp_dict = {"img1" : (img, metadata), "img2" : (img, metadata), ...}
        """

        self.led.on()
        time.sleep(0.5)

        # Make destination directory if it doesn't exist
        check_or_make_directory(dataDir)

        self.__setup_brightfield_controls(gain, exposure_us)
        
        # Announce
        metadata = self.picam2.capture_metadata()
        print(f"Capturing brightfield image(s): path = {dataDir} prefix = {filePrefix} # {numberImages}\n")
        print("Gain = " + str(metadata["AnalogueGain"]))
        print("Exposure (us) = " + str({metadata["ExposureTime"]}))

        exp_dict = {}
        outFileName = append_slash(dataDir) + filePrefix + new_timestamp() + '.hdf5'

        for i in range(numberImages):
            # Capture an image
            array, info = self.__snap_brightfield()
            print("Captured " + str(i+1) + "/" + str(numberImages) + " Duration = " + str(info['capture_time_s']))

            # For now save RGB images, I don't know why they did this...
            # array = array.astype(np.uint8)
            # imageU16 = array.sum(axis=2)
            # imageU16 = imageU16.astype(np.uint16)

            exp_dict["img" + str(i+1)] = (array, info)

        if save:
            dump_to_hdf5(exp_dict, outFileName)

        self.led.off()
        time.sleep(0.5)

        return exp_dict

    def __is_camera_settled(self):
        """Check if the camera autoexposure mode has settled on a gain and exposure time setting
        
        returns: gain, exposure_us, True if camera is settled
        """

        # Get current gain / exposure values
        metadata = self.picam2.capture_metadata()
        lastGain = metadata['AnalogueGain']
        lastExposureTime = metadata['ExposureTime']

        # Give the camera some time to settle, based on current exposure value
        settleWaitTime = lastExposureTime / 1e6
        time.sleep(1 + (settleWaitTime * 5))

        # Get current gain / exposure values
        metadata = self.picam2.capture_metadata()
        gain = metadata['AnalogueGain']
        ExposureTime = metadata['ExposureTime']

        # Check if values have change much during wait period
        settled = (is_within_percent(gain, lastGain, SETTLED_PERCENT) and
                   is_within_percent(ExposureTime, lastExposureTime, SETTLED_PERCENT))

        return gain, ExposureTime, settled

    def wait_for_camera_settled(self):
        """ Wait for camera autoexposure mode to settle on analog gain and exposure time settings
        returns: settled gain and exposure time values
        """

        settled = False

        # Put the camera in autoexposure mode
        controls = {
            'AeConstraintMode': libcontrols.AeConstraintModeEnum.Normal,
            'AeEnable': True,
            'AeExposureMode': libcontrols.AeExposureModeEnum.Normal,
            'AeMeteringMode': libcontrols.AeMeteringModeEnum.CentreWeighted,
            # 'AnalogueGain': gain,
            'AwbEnable': False, # Don't want autowhite balances, this is also automatically disabled by setting ColourGains
            'AwbMode': 0,
            'Brightness': 0.0,
            # 'ColourCorrectionMatrix': (-16.0, 16.0, None), read-only, see manual (Appendix C)
            'ColourGains': DEFAULT_AWB_GAINS, # See manual (Appendix C)
            'Contrast': 1.0,
            # 'ExposureTime': exposure_us,
            # 'ExposureValue': 0.0,
            # 'FrameDurationLimits': (47183, 11767556, None), set by camera_config 
            'NoiseReductionMode': 0, # no noise reduction wanted
            'Saturation': 1.0,
            'ScalerCrop': (0, 0, 3280, 2464),
            'Sharpness': 1.0
        }
        self.picam2.set_controls(controls)
        time.sleep(3)

        # Loop until the camera has settled
        while not settled:
            lastGain, lastExposureTime, settled = self.__is_camera_settled()
            time.sleep(0.1)

        return lastGain, lastExposureTime

    def get_brightfield_autoexposure_controls(self):
        """Run the camera in autoexposure mode and report gain and exposure time once stabilized. 
        The returned paramters can then be used to set the gain/exposure settings with __setup_brightfield_controls()
        
        returns:
            gain: analog gain of camera
            exposure_us: shutter speed in microseconds
        """

        # Turn on LED
        self.led.on()
        time.sleep(0.5)

        # Turn on autoexposure mode

        # Wait for camera to stabilize and collect stabilized values
        lastGain, lastExposureTime = self.wait_for_camera_settled()

        # Turn off LED and return to de
        self.led.off()
        time.sleep(0.5)

        return lastGain, lastExposureTime

    def plot_brightfield_arr(self, img_arr):
        "Plot a single brightfield (RGB) image as a BW image"
        img_arr = img_arr / np.max(img_arr) # Each channel will either be ints from 0-255 or floats from 0.0-255.0 (loaded from hdf5 file)
        img_arr = np.dot(img_arr[...,:3], [0.2989, 0.5870, 0.1140])
        plt.imshow(img_arr, cmap='gray', norm=Normalize())
        plt.colorbar()
        plt.show()
    
    def plot_beta_arr(self, img_arr):
        "Plot a single beta (BW) image"
        # Each pixel will either be ints from 0-3*255*n_images or floats from 0.0-3*255.0*n_images (loaded from hdf5 file) -> map to [0.0, 1.0]
        # img_arr = img_arr / np.max(img_arr) # normalize image to be bewteen 0 and 1
        # plt.imshow(img_arr, cmap='gray', vmin=0.0, vmax=1.0)
        plt.imshow(img_arr, cmap='inferno', norm=Normalize()) # this should be the same as the above 2 (commented out) lines of code
        plt.colorbar()
        plt.show()

    def plot_beta_arr_log(self, img_arr):
        "Plot a single beta (BW) image on log scale"
        plt.imshow(img_arr, cmap='inferno', norm=LogNorm())
        plt.colorbar()
        plt.show()
