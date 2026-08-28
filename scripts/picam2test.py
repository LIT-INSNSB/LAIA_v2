from picamera2 import Picamera2
import time

camera = Picamera2()
camera.configure(
    camera.create_preview_configuration(
        main={"size": (640, 480)}
    )
)

camera.start()
time.sleep(2)

frame = camera.capture_array()

print(type(frame))
print(frame.shape)

camera.stop()
