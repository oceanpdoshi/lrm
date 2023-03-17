''' LRM.py
Defines LRM class that contains all beta microscope functionality
'''

# added imports
from picamera2 import Picamera2
from libcamera import controls as libcontrols
from pprint import *

import io
from PIL import Image
import numpy as np
import time
import datetime
import os
import glob
from gpiozero import LED
import pickle
import lz4.frame
from lrmimage import LrmImage

# mpl packages
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Useful stuff to print before modifying any configurations/controls
# pprint(picam2.sensor_modes)
# pprint(picam2.camera_controls)
# pprint(picam2.camera_properties)

# Constants
LED_PIN = 21

'''V2 resolutions (3280,2464) (1920,1080) (1640,1232) (1640,922) (1280,720) (640,480)'''
# Main stream configuration
DEFAULT_RESOLUTION = (3280, 2464)
DEFAULT_STREAM_FORMAT = 'XBGR8888'

# Raw stream configuration
DEFAULT_SENSOR_MODE = 3 # 10 bit, (3280, 2464), SRGGB10_CSI2P format, (75us, 11766829us) exposure limits, 21.19fp

# Default ColourGain Setting (camera controls) - disables awb when set
DEFAULT_AWB_GAINS = (1.0/8.0*32.0, 11.0/8.0*32.0)

# Percent differece threshold for ExposureTime and AnalogueGain for __check_camera()
EXPOSURE_TIME_THRESHOLD = 10
GAIN_THRESHOLD = 25

# Util functions
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
    """ Append a slash a path if one doesn't already exist """
    if not (path[-1] == '/'):
        path += '/'

    return path


def _get_parent_directory(path):
    """Get the parent directory of a given filepath. If path is to a dir, just return the dirpath but with a / at the end"""
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

# TODO - update all the docstrings and delete the old/deprecated code
# TODO - Don't open a Preview.QTGL window upon class initialization - write a user function to do that
# TODO - look for time.sleep() calls and make changes accordingly

class LRM:
    """Class containing all beta microscope functionality

    Example use case:
    
    Capture brightfield images as separate files:
        
        gain, shutter = LRM.get_brightfield_exposure()
        
        lrm = LRM()    
        
        lrm.snap_brightfield('./data/bf/','bf', 1, gain, shutter)

    Capture multiple beta images as separate files:
        lrm.snap_betas('./data/beta/','bm_5g_10000ms',1000,2.5,10)

    Capture and sum multiple beta image into a single file single file:
        lrm.snap_beta('./data/beta/','bm_5g_10000ms_1000fr',1000,2.5,10)
    
    """

    def __init__(self):

        # Most recent analog gain setting and exposure setting (us)
        self._gainSet = None
        self._exposureSet = None # TODO - this was changed from shutterSET - old repo used shutter speed, here we use exposures https://github.com/raspberrypi/picamera2/issues/145

        self.info = {}

        self.led = LED(LED_PIN)
        self.led.off()

        # Configure (Chapter 4, Appendix B) and start() the camera
        self.picam2 = Picamera2()
        # TODO - decide whether or not we need raw stream
        # config = self.picam2.create_still_configuration(main={"size" : DEFAULT_RESOLUTION, "format" : DEFAULT_STREAM_FORMAT}, raw={self.picam2.sensor_modes[DEFAULT_SENSOR_MODE]})
        config = self.picam2.create_still_configuration(main={"size" : DEFAULT_RESOLUTION, "format" : DEFAULT_STREAM_FORMAT})
        self.picam2.configure(config)
        self.picam2.start() # Uses Preview.NULL by default
        time.sleep(2)

    # TODO - rename __setup_beta() and __setup_brightfield() to __setup_beta_controls() and __setup_brightfield_controls()
    def __setup_beta(self, gain=None, exposure_us=None, ColourGains=DEFAULT_AWB_GAINS):
        """Start up the camera with settings optimized for beta imaging
        gain: analog gain
        exposure_us : exposure time in us
        """

        # Gain and shutter speed must be provided
        if gain is None:
            raise Exception("No gain provided!")
        elif exposure_us is None:
            raise Exception("No exposure_us provided!")

        # Copy provided values into LRM gain and shutter setpoint variables
        self._gainSet = gain
        self._exposureSet = exposure_us

        # any paramter not commented is just set to default alue
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
        time.sleep(2)


    # TODO - figure out why this function exists - it's the same as _setup_beta, but (1,1) -> (1,2) in terms of old awb gains
    def __setup_brightfield(self, gain=None, exposure_us=None):
        """Start up the camera with settings optimized for beta imaging
        gain: analog gain
        exposure_us : exposure time in us
        """

        self.__setup_beta(gain, exposure_us, ColourGains=(1.0/8.0 * 32.0, 2.0/8.0*32.0))


    def __check_camera(self):
        """ Check both the gain and exposure settings using metadata functionality"""
        metadata = self.picam2.capture_metadata()
        gain = metadata["AnalogueGain"]
        exposure_us = metadata["ExposureTime"]

        exposure_correct = is_within_percent(self.exposureSet, exposure_us, EXPOSURE_TIME_THRESHOLD)
        gain_correct = is_within_percent(self.gainSet, gain, GAIN_THRESHOLD)

        return exposure_correct and gain_correct

    # TODO - see the control loop that uses this function and reboot - maybe don't even need this function
    def __reboot(self):
        """ Reboots the camera """

        self.picam2.stop()
        config = self.picam2.create_still_configuration(main={"size" : DEFAULT_RESOLUTION, "format" : DEFAULT_STREAM_FORMAT})
        self.picam2.configure(config)
        self.picam2.start() # Uses Preview.NULL by default
        time.sleep(3)


    # TODO - reinstate reboot loop under if not self.__check_camera() ???
    def __snap_beta(self, threshold=0):
        """
        Grabs an RGB frame (np.ndarrary) from camera using beta imaging controls
        Sums across the RGB channels and zeros out pixels less than threshold (default 0 does no thresholding).
        returns: np.ndarray camera image, image metadata+additional info
        """

        # Setup camera controls, and check you've waited long enough
        # self.__setup_beta(self._gainSet, self._exposureSet)
        if not self.__check_camera():
            metadata = self.picam2.capture_metadata()
            print("Gain was: " + str(metadata["AnalogueGain"]) + ", wanted " + str(self._gainSet))
            print("ExposureTime: " + str(metadata["ExposureTime"]) + " [us]" + ", wanted " + str(self._exposureSet) + " [us]")
            raise Exception("Camera gain or exposure time was not set correctly.")

        # Capture image (np.ndarray) from the camera + record total capture time
        # NOTE - use the naming convention array to differentiate from PIL image objects - consistent with picamera2 manual
        captureStartTime = time.time()
        request = self.picam2.capture_request("main")
        array, metadata = request.make_array(), request.get_metadata()
        captureTimeSeconds = time.time() - captureStartTime

        self.info = self.__get_info()
        self.info['capture_time_s'] = captureTimeSeconds
        
        # Sum across RGB Channels to generate one channel array
        array = array.sum(axis=2, dtype=np.uint16)

        # Apply threshold
        array[array < threshold] = 0

        # Compute some image metrics and store in info
        self.info['image_sum'] = array.sum()
        self.info['image_std'] = np.std(array)
        self.info['image_max'] = array.max()
        self.info['image_std'] = array.min()
        self.info['image_mean'] = array.mean()

        return array, self.info

    def __snap_brightfield(self):
        """
        Grabs an RGB frame (np.ndarrary) from camera with brightfield imaging controls
        returns: (np.ndarray) camera image, (dict) image metadata+additonal info
        """

        # Setup camera controls, and check you've waited long enough
        # self.__setup_brightfield(self._gainSet, self._exposureSet)
        if not self.__check_camera():
            metadata = self.picam2.capture_metadata()
            print("Gain was: " + str(metadata["AnalogueGain"]) + ", wanted " + str(self._gainSet))
            print("ExposureTime: " + str(metadata["ExposureTime"]) + " [us]" + ", wanted " + str(self._exposureSet) + " [us]")
            raise Exception("Camera gain or exposure time was not set correctly.")

        # Capture image (np.ndarray) from the camera + record total capture time
        # NOTE - use the naming convention array to differentiate from PIL image objects - consistent with picamera2 manual
        captureStartTime = time.time()
        request = self.picam2.capture_request("main")
        array, metadata = request.make_array(), request.get_metadata()
        captureTimeSeconds = time.time() - captureStartTime

        self.info = self.__get_info()
        self.info['capture_time_s'] = captureTimeSeconds

        return array, self.info

    def capture_beta(self, dataDir, filePrefix, numberImages, analogGain, exposureDurationUs, threshold=0):
        """Integrate desired number of frames from camera object and write to disk (save threaded version - no idea what this means)
        dataDir: save dir
        filePrefix: prefix of saved file
        numberImages: number of frames to capture
        analogGain: fixed analog gain setting; automatic if None
        exposureDurationUs: fixed shutter speed; automatic if None
        Returns: nothing
        """

        # Make sure the LED is off just in case
        self.led.off()

        # Make destination directory if it doesn't exist
        check_or_make_directory(dataDir)

        # Set up camera for beta acquisition
        self.__setup_beta(analogGain, exposureDurationUs)

        # Announce
        metadata = self.picam2.capture_metadata()
        print(f"Capturing beta image(s): path = {dataDir} prefix = {filePrefix} # {numberImages}\n")
        print("Gain = " + str(metadata["AnalogueGain"]))
        print("Exposure (us) = " + str({metadata["ExposureTime"]}))

        infoList = []

        # TODO - they tossed the first image for some reason - figure out why
        for i in range(numberImages):

            # Capture one image

            array, info = self.__snap_beta(threshold)
            infoList.append(info)

            # Sum image
            if i == 0:
                summed = array
            else:
                summed = summed + array

            print("Captured " + str(i) + "/" + str(numberImages) + " Duration = " + str(
                info['capture_time_s']) + " Sum = " + str(array.sum()) + " Total sum = " + str(summed.sum()))
                
        # TODO - Write image to file - just work with numpy arrays and pickle for now. They seem to be knowledgable about dtype as well
        # bizarre to work with infoList given eacch iteration checks for same exposure/gain
        outFileName = append_slash(dataDir) + filePrefix
        np.save(outFileName + ".npy")
        
        with open('saved_dictionary.pkl', 'wb') as f:
            pickle.dump(infoList[-1], f)
        
        # TODO - How to open pickle dict
        # with open('saved_dictionary.pkl', 'rb') as f:
        #     loaded_dict = pickle.load(f)


        # TODO - rewrite LrmImage class 
        # betaImage = LrmImage(summed, infoList)
        # betaImage.save(outFileName)

    def capture_brightfield(self, dataDir, filePrefix, numberImages, analogGain=None, exposureDurationUs=None):
        """Capture one or more brightfield images and write to disk as .png
        dataDir: path to save brightfield images
        filePrefix: prefix of saved images
        numberImages: number of frames to capture        
        analogGain: fixed analog gain setting; automatiIf the file is a .npy file, then a single array is returned.

c if None
        exposureDurationUs: fixed shutter speed; automatic if None
        Returns: nothing
        """

        # Make sure LED is on
        self.led.on()

        # Make destination directory if it doesn't exist
        check_or_make_directory(dataDir)

        self.__setup_brightfield(analogGain, exposureDurationUs)
        
        # Announce
        metadata = self.picam2.capture_metadata()
        print(f"Capturing brightfield image(s): path = {dataDir} prefix = {filePrefix} # {numberImages}\n")
        print("Gain = " + str(metadata["AnalogueGain"]))
        print("Exposure (us) = " + str({metadata["ExposureTime"]}))

        for i in range(numberImages):
            # Capture one image
            array, info = self.__snap_brightfield()
            print("Captured " + str(i) + "/" + str(numberImages) + " Duration = " + str(info['capture_time_s']))

            # Extract data from camera stream

            bfOutFileFullPath = append_slash(dataDir) + filePrefix + str(i) + ".j2k"

            # TODO - check if this is buggy... - looks like it might be easier to directly convert numpy -> PIL -> save png?
            # Sum along color channel axis and enforce uint16 output
            array = array.astype(np.uint8)
            imageU16 = array.sum(axis=2)
            imageU16 = imageU16.astype(np.uint16)

            # Save to file
            Image.fromarray(imageU16).save(bfOutFileFullPath)

        # Turn LED off
        self.led.off()

    # TODO - refactor the next three functions to take a brightfield autoexposure
    def is_camera_settled(self, camera):
        """Check if the camera autoexposure mode has settled on a gain and shutter setting
        
        returns: most recent gain, shutter speed, and True if camera is settled
        """
        settledPercent = 2.5

        # Get current gain / exposure values
        lastGain = camera.analog_gain
        lastShutterUs = camera.exposure_speed

        # Give the camera some time to settle, based on current exposure value
        settleWaitTime = lastShutterUs / 1000000.
        time.sleep(1 + (settleWaitTime * 5))

        # Get current gain / exposure values
        gain = camera.analog_gain
        shutter = camera.exposure_speed

        # Check if values have change much during wait period
        settled = (is_within_percent(gain, lastGain, settledPercent) and
                   is_within_percent(shutter, lastShutterUs, settledPercent))

        return gain, shutter, settled

    def wait_for_camera_settled(self, camera):
        """ Wait for camera autoexposure mode to settle on analog gain and shutter settings
        returns: settled gain and shutter speed values
        """

        settled = False

        # Loop until the camera has settled
        while not settled:
            lastGain, lastShutterUs, settled = self.is_camera_settled(camera)
            time.sleep(0.1)

        return lastGain, lastShutterUs

    def get_brightfield_exposure(self):
        """Run the camera in autoexposure mode and report gain and shutter speed once stabilized
        
        returns:
            gain: analog gain of camera
            shutterUs: shutter speed in microseconds
        """

        # Turn on LED
        self.led.on()

        # Start up camera 
        camera = picamera.PiCamera()
        camera.iso = 100
        camera.framerate_range = (30, 30)
        camera.resolution = (3280, 2464)
        camera.sensor_mode = 0
        camera.awb_mode = 'auto'
        camera.shutter_speed = 0
        camera.exposure_mode = 'auto'

        # Wait for camera to stabilize and collect stabilized values
        lastGain, lastShutterUs = self.wait_for_camera_settled(camera)

        # Turn off LED
        self.led.off()

        # Close the camera
        camera.close()

        return lastGain, lastShutterUs

    # TODO - replace this with a plotting/image display function
    def preview_last(self, hold=False):

        def prepare_image(image, colorMap='gray'):
            displayResolution = (616, 820)
            im = Image.fromarray(image[1140:2140, 882:1882])
            im = im.convert('L')
            # im.thumbnail((820,616))
            im = np.array(im, dtype=np.uint8)
            cmap = cm.get_cmap(colorMap)
            im = cmap(im)
            im = im * 255
            im = im.astype(np.uint8)

            return im

        bfYes = self.lastBfImage is not None
        betaYes = self.lastBetaImage is not None

        if hold == False:
            plt.figure("Preview")
            plt.ion()
            plt.show()
        else:
            plt.figure("Preview")

        if bfYes:
            bf = prepare_image(self.lastBfImage, 'gray')

        if betaYes:
            beta = prepare_image(self.lastBetaImage, 'inferno')

        if bfYes and betaYes:
            merged = bf + beta
            plt.imshow(merged)

        else:
            if bfYes:
                plt.figure()
                plt.imshow(bf)

            else:

                if betaYes:
                    plt.figure()
                    plt.clim(0, 10)
                    plt.imshow(beta)

        if hold == False:
            plt.draw()
            plt.pause(0.01)
        else:
            plt.show()
