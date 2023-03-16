import os
import time

from picamera2 import Picamera2, Preview
from pprint import *

print(os.getcwd())

picam2 = Picamera2()
# camera_config = picam2.create_preview_configuration()
# picam2.configure(camera_config)
# picam2.start_preview(Preview.QTGL)
# picam2.start()
# time.sleep(2)
# picam2.capture_file('test.jpg')

# Useful stuff to print before modifying any configurations/controls
pprint(picam2.sensor_modes)
pprint(picam2.camera_controls)
pprint(picam2.camera_properties)
