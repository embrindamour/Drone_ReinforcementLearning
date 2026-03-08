from djitellopy import Tello
import cv2

drone = Tello()
drone.connect()
drone.streamon()
reader = drone.get_frame_read()

while True:
    frame = reader.frame
    # from command below: a window should pop up showing the live camera feed from the drone
    cv2.imshow("Tello Camera", frame)
    # from command below: image actually gets rendered
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

drone.streamoff()
cv2.destroyAllWindows()