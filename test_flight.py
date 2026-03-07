from djitellopy import Tello
import time

drone = Tello()
drone.connect()
print("Battery:", drone.get_battery())

time.sleep(2)
drone.takeoff()
time.sleep(3)
drone.land()