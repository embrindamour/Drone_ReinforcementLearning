from djitellopy import Tello
import cv2
import time
from datetime import datetime


drone = Tello()
drone.connect()
print("Battery:", drone.get_battery())

# Start camera
drone.streamon()
reader = drone.get_frame_read()
time.sleep(1)  # let camera warm up

# Take off
drone.takeoff()
time.sleep(2)  

# Hover and display camera feed for 5 seconds
print("Hovering... press Q in the camera window to land early")
start = time.time()
while time.time() - start < 5:
    frame = reader.frame
    cv2.imshow("Tello Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Take picture at max height
print("Taking picture...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
frame = reader.frame
cv2.imwrite(f"photo_{timestamp}.jpg", frame)

# Land
drone.land()
drone.streamoff()
cv2.destroyAllWindows()
print("Done! Check max_height_photo.jpg in your project folder.")