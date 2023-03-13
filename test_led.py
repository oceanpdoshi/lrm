from gpiozero import LED
from time import sleep


print("Hello world")
LED_PIN = 21
led = LED(LED_PIN)

# Turn LED on for 2 seconds before turning off

led.on()
sleep(2.0)
led.off()
