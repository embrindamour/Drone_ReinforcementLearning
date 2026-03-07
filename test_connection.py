from djitellopy import Tello

drone = Tello()
drone.connect()
print("Battery:", drone.get_battery())