from time import sleep
from picamera import PiCamera

# NOTE - picamera no longer works for Raspberry Pi 64-bit os 
# TODO - edit this code stack and migrate it to picamera2 (see manual)

camera = PiCamera()
camera.resolution = (1024, 768)
camera.start_preview()
# Camera warm-up time
sleep(2)
camera.capture('foo.jpg')