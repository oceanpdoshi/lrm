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
        self.lastSaveFileFullPath = None
        self.lastCaptureDurationSeconds = None
        self.logFileFullPath = None
        self.lastBfImage = None
        self.lastBetaImage = None

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


    # TODO - see the control loop that uses this function and reboot - maybe don't even need these functions..
    def __check_camera(self):
        """ Check both the gain and exposure settings using metadata functionality"""
        metadata = self.picam2.capture_metadata()
        gain = metadata["AnalogueGain"]
        exposure_us = metadata["ExposureTime"]

        exposure_correct = is_within_percent(self.exposureSet, exposure_us, EXPOSURE_TIME_THRESHOLD)
        gain_correct = is_within_percent(self.gainSet, gain, GAIN_THRESHOLD)

        return exposure_correct and gain_correct

    
    def __reboot(self):
        """ Reboots the camera """

        self.picam2.stop()
        config = self.picam2.create_still_configuration(main={"size" : DEFAULT_RESOLUTION, "format" : DEFAULT_STREAM_FORMAT})
        self.picam2.configure(config)
        self.picam2.start() # Uses Preview.NULL by default
        time.sleep(3)

    def __stream_to_np(self, stream):
        """Convert a supplied stream into an numpy array
        returns image from stream as numpy array
        """

        stream.seek(0)

        # Copy locally
        streamBuffer = stream.getvalue()

        # Convert to numpy array
        image = np.array(Image.open(io.BytesIO(streamBuffer)))

        # Convert to B/W by summing color channels into one 16-bit channel
        if len(image.shape) == 3:
            image = image.sum(axis=2, dtype=np.uint16)

        return image

    def __stream_to_RGB_np(self, stream):
        """Convert a supplied stream into an numpy array
        """

        stream.seek(0)

        # Copy locally
        streamBuffer = stream.getvalue()

        # Convert to numpy array
        image = np.array(Image.open(io.BytesIO(streamBuffer)))
        # image = image.astype(np.uint8)

        return image

    def __get_info(self, camera):
        """ Grab settings from the camera and return as dictionary
        """

        info = {'analog_gain': float(camera.analog_gain),
                'framerate_range': (float(camera.framerate_range[0]), float(camera.framerate_range[1])),
                'sharpness': camera.sharpness,
                'brightness': camera.brightness,
                'saturation': camera.saturation,
                'video_stabilization': camera.video_stabilization,
                'exposure_compensation': camera.exposure_compensation,
                'meter_mode': camera.meter_mode,
                'image_effect': camera.image_effect,
                'image_denoise': camera.image_denoise,
                'color_effects': camera.color_effects,
                'drc_strength': camera.drc_strength,
                'awb_gains': (float(camera.awb_gains[0]), float(camera.awb_gains[1])),
                'iso': camera.iso,
                'shutter_speed': camera.shutter_speed,
                'exposure_speed': camera.exposure_speed,
                'awb_mode': camera.awb_mode,
                'exposure_mode': camera.exposure_mode,
                'sensor_mode': camera.sensor_mode,
                'resolution': (camera.resolution[0], camera.resolution[1]),
                'datetime': datetime.datetime.now}

        return info

    def __snap_beta(self, camera, threshold=0):
        """Grab frame from camera and return stream as numpy array
        rgb: returns image as 3d numpy array with RGB channels
        returns: numpy array of camera image
        """

        # Check, reboot, and setup until camera settings are correct
        while not self.__check_camera(camera):
            print("Check camera failed, rebooting ...")
            print(f"Gain setpoint = {self._gainSet} Current = {camera.analog_gain}")
            print(f"Shutter setpoint = {self._shutterSet} Current = {camera.shutter_speed}")

            camera = self.__reboot(camera)
            self.__setup_beta(camera, self._gainSet, self._shutterSet)

        # Capture from the camera
        stream = io.BytesIO()
        captureStartTime = time.time()
        next(camera.capture_continuous(stream, 'jpeg', use_video_port=True, bayer=False))
        captureTimeSeconds = time.time() - captureStartTime

        # Grab image from camera (returned as stream)
        self.info = self.__get_info(camera)
        self.info['capture_time_s'] = captureTimeSeconds

        if stream is not None:
            # Copy covert stream to np array and copy into image variable
            image = self.__stream_to_np(stream)

            # Apply threshold
            image[image < threshold] = 0

            # Compute some image metrics and store in info
            self.info['image_sum'] = image.sum()
            self.info['image_std'] = np.std(image)
            self.info['image_max'] = image.max()
            self.info['image_std'] = image.min()
            self.info['image_mean'] = image.mean()

            return image, self.info

    def __snap_brightfield(self, camera):
        """Grab frame from camera and return stream as numpy array
        rgb: returns image as 3d numpy array with RGB channels
        returns: numpy array of camera image
        """

        # Check, reboot, and setup until camera settings are correct
        while not self.__check_camera(camera):
            print("Check camera failed, rebooting ...")
            print("Gain setpoint = " + str(self._gainSet) + " Current = " + str(camera.analog_gain))
            print("Shutter setpoint = " + str(self._shutterSet) + " Current = " + str(camera.shutter_speed))

            camera = self.__reboot(camera)
            self.__setup_brightfield(camera, self._gainSet, self._shutterSet)

        # Capture from the camera
        stream = io.BytesIO()
        captureStartTime = time.time()
        next(camera.capture_continuous(stream, 'jpeg', use_video_port=True, bayer=False))
        captureTimeSeconds = time.time() - captureStartTime

        # Grab image from camera (returned as stream)
        self.info = self.__get_info(camera)
        self.info['capture_time_s'] = captureTimeSeconds

        if stream is not None:
            image = self.__stream_to_RGB_np(stream)

            return image, self.info

    def capture_betas(self, fullPath, filePrefix, numberImages, analogGain, exposureDurationUs):
        """Capture desired number of frames from camera object and write to disk (save threaded version)
        fullPath: save path
        filePrefix: prefix of saved file
        numberImages: number of frames to capture
        analogGain: fixed analog gain setting; automatic if None
        exposureDurationUs: fixed shutter speed; automatic if None
        Returns: nothing
        """

        # Make sure the LED is off just in case+
        self.led.off()

        # Make destination directory if it doesn't exist
        check_or_make_directory(fullPath)

        # Capture
        camera = picamera.PiCamera()

        # Set up camera for beta imaging mode
        self.__setup_beta(camera, analogGain, exposureDurationUs)

        # Announce
        print(f"Capturing beta image(s): path = {fullPath} prefix = {filePrefix} # {numberImages}\n"
              f"Gain = {camera.analog_gain} Exposure (s) = {camera.shutter_speed / 1000000.}")

        for i in range(numberImages + 1):

            # Capture one image
            image, meta = self.__snap_beta(camera)

            # Write image to file, toss first image
            if i > 0:
                betaImage = LrmImage(image, meta)
                outFileName = f"{append_slash(fullPath)}{filePrefix}_{i}"
                betaImage.save(outFileName)

            print(f"Captured {i} of {numberImages} Duration = {meta['capture_time_s']}")

            # Store for display
            self.lastBetaImage = image

        # Framerate needs to be set to 1 in order to let the camera close at long shutter speeds
        camera.framerate = 1
        camera.close()

    def capture_beta(self, fullPath, filePrefix, numberImages, analogGain, exposureDurationUs, threshold=0):
        """Integrate desired number of frames from camera object and write to disk (save threaded version)
        fullPath: save path
        filePrefix: prefix of saved file
        numberImages: number of frames to capture
        analogGain: fixed analog gain setting; automatic if None
        exposureDurationUs: fixed shutter speed; automatic if None
        Returns: nothing
        """

        # Make sure the LED is off just in case
        self.led.off()

        # Make destination directory if it doesn't exist
        check_or_make_directory(fullPath)

        # Capture
        camera = picamera.PiCamera()

        # Set up camera for beta acquisition
        self.__setup_beta(camera, analogGain, exposureDurationUs)

        # Announce
        # Announce
        print(f"Capturing beta image(s): path = {fullPath} prefix = {filePrefix} # {numberImages}\n"
              f"Gain = {camera.analog_gain} Exposure (s) = {camera.shutter_speed / 1000000.}")


        infoList = []

        for i in range(numberImages + 1):

            # Capture one image

            image, info = self.__snap_beta(camera, threshold)

            # Write image to file, toss first image
            if i > 0:

                infoList.append(info)

                # Sum image
                if i == 1:
                    summed = image
                else:
                    summed = summed + image

                print("Captured " + str(i) + "/" + str(numberImages) + " Duration = " + str(
                    info['capture_time_s']) + " Sum = " + str(image.sum()) + " Total sum = " + str(summed.sum()))

        # Close the camera as soon as we can
        camera.framerate = 1
        camera.close()

        # Write image to file
        betaImage = LrmImage(summed, infoList)
        outFileName = append_slash(fullPath) + filePrefix
        betaImage.save(outFileName)

        # Store for display
        self.lastBetaImage = summed

    def capture_brightfield(self, fullPath, filePrefix, numberImages, analogGain=None, exposureDurationUs=None):
        """Capture one or more brightfield images and write to disk as .png
        fullPath: path to save brightfield images
        filePrefix: prefix of saved images
        numberImages: number of frames to capture        
        analogGain: fixed analog gain setting; automatic if None
        exposureDurationUs: fixed shutter speed; automatic if None
        Returns: nothing
        """

        # Make sure LED is on
        self.led.on()

        # Make destination directory if it doesn't exist
        check_or_make_directory(fullPath)

        # Capture
        camera = picamera.PiCamera()

        self.__setup_brightfield(camera, analogGain, exposureDurationUs)

        # Announce
        print(
            "Capturing brightfield image(s): path = " + fullPath + " prefix = " + filePrefix + " # " + str(numberImages)
            + " Gain = " + str(round(camera.analog_gain, 2))
            + " Exposure (s) = " + str(round(camera.shutter_speed / 1000000., 2)))

        for i in range(numberImages):
            # Capture one image
            image, info = self.__snap_brightfield(camera)
            print("Captured " + str(i) + "/" + str(numberImages) + " Duration = " + str(info['capture_time_s']))

            # Store for display
            self.lastBfImage = image

            # Extract data from camera stream
            outFileFullPath = append_slash(fullPath) + filePrefix + str(i)

            # bfImage = BetaImage(image,info)

            # Close camera as soon as we can
            camera.framerate = 1
            camera.close()

            # Save file as lz4-compressed data
            # disable for now for space savings 5/23/2019 and speed
            # bfImage.save(outFileFullPath)

            # ... and also as .jp2 for quick preview
            bfOutFileFullPath = append_slash(fullPath) + filePrefix + str(i) + ".j2k"

            # Sum along color channel axis and enforce uint16 output
            image = image.astype(np.uint8)
            imageU16 = image.sum(axis=2)
            imageU16 = imageU16.astype(np.uint16)

            # Save to file
            Image.fromarray(imageU16).save(bfOutFileFullPath)

        # Turn LED off
        self.led.off()

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
