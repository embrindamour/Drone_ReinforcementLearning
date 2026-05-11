# =============================================================================
# green_detect.py
# Perception pipeline for lime green ball detection on the DJI Tello.
# Milestone 1 deliverable — run standalone to test and calibrate before flying.
#
# As a module:  from green_detect import detect_ball
#               (cx, cy), area, display = detect_ball(frame)
#
# Standalone:   python green_detect.py
#               Streams from Tello (no takeoff), shows live feed with:
#               - 3x3 grid overlay
#               - Bounding box and centroid when ball detected
#               - Grid cell, area, and HSV printed in frame
#               - Click anywhere on frame to print HSV at that pixel (calibration)
#               Press Q to quit.
# =============================================================================

# =============================================================================
# HYPERPARAMETERS — calibrate these before flying
# Tip: run standalone, click on the ball in the frame, read the printed HSV,
# then set lower/upper bounds ~15 hue units and ~40 sat/val units around it.
# =============================================================================

# Lime green HSV range — single range, no wrap-around needed (unlike red)
GREEN_LOWER = (35, 100, 100)
GREEN_UPPER = (85, 255, 255)

# Morphological kernel size — larger removes more noise but may shrink blob
MORPH_KERNEL_SIZE = 5

# Minimum contour area to count as a valid detection (filters distant specks)
MIN_BLOB_AREA = 500

# Camera frame dimensions (Tello default)
#FRAME_W = 960
#FRAME_H = 720

# =============================================================================
# END HYPERPARAMETERS
# =============================================================================

import cv2
import numpy as np


def detect_ball(frame):
    """
    Detect the lime green ball in a single BGR frame.

    Returns:
        (cx, cy)  — centroid pixel coords, or None if not detected
        area      — contour area in pixels, or None if not detected
        display   — annotated copy of frame with bounding box, centroid,
                    grid overlay, and detection info drawn on it
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Single-range green mask — lime green sits cleanly mid-spectrum,
    # no hue wrap-around arithmetic needed (unlike red)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    # Morphological opening (erosion → dilation) removes small noise specks
    # without shrinking the main blob significantly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Dilation fills any gaps or holes inside the ball blob
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours in the cleaned mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    display = frame.copy()

    # Always draw the 3x3 grid so zone boundaries are visible even with no detection
    _draw_grid(display)

    if not contours:
        # No contours found — draw LOST indicator
        cv2.putText(display, "LOST — no ball detected",
                    (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return None, None, display

    # Use the largest contour only — filters out small green background objects
    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)

    if area < MIN_BLOB_AREA:
        # Largest contour is too small to be the ball — treat as no detection
        cv2.putText(display, f"LOST — largest blob area={int(area)} < {MIN_BLOB_AREA}",
                    (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        return None, None, display

    # Compute centroid from image moments
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None, display

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # NEW
    FRAME_W = frame.shape[1]
    FRAME_H = frame.shape[0]

    # Compute which grid cell the centroid falls in
    col      = min(int(cx // (FRAME_W / 3)), 2)
    row      = min(int(cy // (FRAME_H / 3)), 2)
    state    = row * 3 + col
    cell_str = f"cell ({row},{col})  s={state}"

    # Draw bounding box in green
    x, y, w, h = cv2.boundingRect(largest)
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw centroid dot in yellow
    cv2.circle(display, (cx, cy), 6, (0, 255, 255), -1)

    # Highlight the active grid cell with a colored border
    x1, y1 = col * FRAME_W // 3, row * FRAME_H // 3
    x2, y2 = (col + 1) * FRAME_W // 3, (row + 1) * FRAME_H // 3
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Overlay detection info: area, cell, centroid coords
    cv2.putText(display, f"area={int(area)}",
                (x, max(y - 22, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    cv2.putText(display, cell_str,
                (x, max(y - 6, 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
    cv2.putText(display, f"centroid ({cx},{cy})",
                (8, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

    return (cx, cy), area, display


def _draw_grid(frame):
    """
    Draw the 3x3 zone grid on the frame.
    Each zone corresponds to one grid state in the Q-learning state space.
    """
    h, w = frame.shape[:2]
    for i in range(1, 3):
        # Vertical dividers
        cv2.line(frame, (w * i // 3, 0), (w * i // 3, h), (80, 80, 80), 1)
        # Horizontal dividers
        cv2.line(frame, (0, h * i // 3), (w, h * i // 3), (80, 80, 80), 1)

    # Label each zone with its state index — useful for verifying discretization
    labels = ["s=0", "s=1", "s=2",
            "s=3", "s=4\n(ctr)", "s=5",
            "s=6", "s=7", "s=8"]
    for idx, label in enumerate(labels):
        r, c = divmod(idx, 3)
        tx = c * w // 3 + 6
        ty = r * h // 3 + 18
        cv2.putText(frame, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)


# =============================================================================
# Standalone test — streams from Tello, no takeoff required
# Click on any pixel to print its HSV value (use this to calibrate thresholds)
# =============================================================================

# Global for mouse callback
_last_click_hsv = None

def _mouse_callback(event, x, y, flags, param):
    """On left click, print HSV at that pixel for threshold calibration."""
    global _last_click_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = param
        h_val = hsv_frame[y, x, 0]
        s_val = hsv_frame[y, x, 1]
        v_val = hsv_frame[y, x, 2]
        _last_click_hsv = (h_val, s_val, v_val)
        print(f"  Clicked pixel ({x},{y}) → HSV = ({h_val}, {s_val}, {v_val})")
        print(f"  Suggested range: "
            f"lower=({max(0,h_val-15)}, {max(0,s_val-40)}, {max(0,v_val-40)})  "
            f"upper=({min(180,h_val+15)}, 255, 255)")


if __name__ == "__main__":
    print("Opening laptop webcam — no drone needed.")
    print("Hold the lime green ball in front of your camera.")
    print("Click on the ball to print HSV values for calibration.")
    print("Press Q to quit.\n")

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("green_detect — webcam test")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Webcam read failed.")
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.setMouseCallback("green_detect — webcam test",
                                 _mouse_callback, hsv_frame)

            centroid, area, display = detect_ball(frame)

            if centroid is not None:
                cx, cy = centroid
                col = min(int(cx // (frame.shape[1] / 3)), 2)
                row = min(int(cy // (frame.shape[0] / 3)), 2)
                print(f"\r  Detected: centroid=({cx},{cy})  "
                      f"cell=({row},{col})  area={int(area):<7}", end="")

            cv2.imshow("green_detect — webcam test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nDone.")