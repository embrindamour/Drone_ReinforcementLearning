# =============================================================================
# drone_qlearn.py
# Deploys the SARSA-trained Q-table on the DJI Tello for live lime green ball tracking.
# Loads q_table_sarsa.npy and runs the greedy policy — no online learning.
#
# =============================================================================
# QUICK SETUP CHECKLIST
# =============================================================================
#   [ ] Connect laptop to Tello WiFi (TELLO-XXXXXX)
#   [ ] Place q_table_sarsa.npy in the same folder as this file
#   [ ] Place green_detect.py and sim_env.py in the same folder
#   [ ] Charge drone to >50% battery
#   [ ] Clear a 2m × 2m indoor flight area, no overhead obstacles
#   [ ] Have the lime green ball ready
#
# =============================================================================
# HOW TO TEST — THREE MODES (run from terminal)
# =============================================================================
#
#   MODE 1 — Perception only (NO drone needed, use laptop webcam):
#       python drone_qlearn.py --test perception
#       ► Tests the color detector and grid state mapping in real time.
#         Aim the webcam at the lime green ball and verify:
#         - green bounding box appears around the ball
#         - correct grid cell highlights in the overlay
#         - state printed in terminal matches visual position
#         - LOST printed when ball leaves frame or is occluded
#       ► Press Q to quit.
#
#   MODE 2 — Command dry run (drone CONNECTED, NOT flying):
#       python drone_qlearn.py --test commands
#       ► Connects to drone, streams video, runs the full greedy policy loop
#         but NEVER calls takeoff() or send_rc_control().
#         Every step prints exactly what command WOULD be sent.
#         Use this to verify:
#         - drone video stream connects
#         - state detection works on the drone's camera
#         - Q-table loads and produces sensible actions
#         - timing loop runs without errors
#       ► Press Q to quit.
#
#   MODE 3 — Full flight (default):
#       python drone_qlearn.py
#       ► Runs a live tracked episode.
#         - Press T in the video window to take off and begin
#         - Press L at any time for emergency land
#         - Flight log and video saved to flight_data/
#
# =============================================================================

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# Q-table
Q_TABLE_PATH  = "q_table_sarsa.npy"      # SARSA table from pretrain.py

# --- Step timing -------------------------------------------------------------
# Each step has two phases: MOVE (drone translates) then STABILIZE (drone stops).
# Hover skips the MOVE phase and just waits STEP_DURATION.
#
# Total time per step = MOVE_DURATION + STABILIZE_DURATION = 1.0s
# At 50 max steps → 50s max per episode
# At ~45 avg steps (from eval) → ~45s avg episode
#
# To shorten: reduce MOVE_DURATION (less physical displacement per step)
# To lengthen: increase STABILIZE_DURATION (more time for drone to stop)
MOVE_DURATION       = 0.3   # seconds RC command is active
STABILIZE_DURATION  = 0.7   # seconds to stabilize after stopping
STEP_DURATION       = MOVE_DURATION + STABILIZE_DURATION   # = 1.0s

# --- Action speeds (send_rc_control values, range 0-100) --------------------
# Keep low — the goal is small corrective movements, not large translations.
# Increase if ball position barely changes between steps.
# Decrease if drone overshoots the target zone.
LR_SPEED  = 35   # left/right translation speed
UD_SPEED  = 35   # up/down translation speed

# --- Episode structure -------------------------------------------------------
MAX_STEPS_PER_EP = 50       # matches sim MAX_STEPS
TAKEOFF_WAIT     = 3.0      # seconds to stabilize after takeoff
EPISODE_WAIT     = 3.0      # seconds between episodes for repositioning

# --- Perception --------------------------------------------------------------
MIN_BLOB_AREA = 500         # minimum contour area for valid ball detection
FRAME_W       = 960         # Tello stream width
FRAME_H       = 720         # Tello stream height

# --- Logging -----------------------------------------------------------------
DATA_DIR = "flight_data"    # created automatically if it does not exist

# --- Reward values — mirror sim_env exactly ----------------------------------
R_CENTER   = +1.0
R_ADJACENT = +0.3
R_CORNER   = -0.1
R_LOST     = -1.0

# =============================================================================
# END HYPERPARAMETERS
# =============================================================================

import argparse
import csv
import datetime
import os
import sys
import threading
import time

import cv2
import numpy as np

from sim_env import LOST, NUM_STATES, NUM_ACTIONS, ACTION_NAMES, _decode

# Lime green ball detector — returns (centroid_or_None, area, annotated_frame)
from green_detect import detect_ball


# =============================================================================
# State helpers
# =============================================================================

_CELL_TYPE = {
    (0,0):"corner",   (0,1):"adjacent", (0,2):"corner",
    (1,0):"adjacent", (1,1):"center",   (1,2):"adjacent",
    (2,0):"corner",   (2,1):"adjacent", (2,2):"corner",
}

_REWARD_MAP = {"center": R_CENTER, "adjacent": R_ADJACENT, "corner": R_CORNER}


def discretize(cx, cy, fw=FRAME_W, fh=FRAME_H):
    """
    Map pixel centroid (cx, cy) → grid state integer 0-8.
    fw/fh are the actual stream dimensions — always pass these from
    the live frame so zone boundaries scale correctly regardless of
    whether the Tello streams at 960x720, 400x300, or anything else.
    """
    col = min(int(cx // (fw / 3)), 2)
    row = min(int(cy // (fh / 3)), 2)
    return row * 3 + col


def get_reward(state):
    if state == LOST:
        return R_LOST
    row, col = _decode(state)
    return _REWARD_MAP[_CELL_TYPE[(row, col)]]


def state_label(state):
    if state == LOST:
        return "LOST"
    r, c = _decode(state)
    return f"s={state} ({_CELL_TYPE[(r,c)]})"


# =============================================================================
# Action execution
# =============================================================================

# RC control values per action.
# send_rc_control(left_right, forward_backward, up_down, yaw)
# Sign convention:
#   left_right  : positive = right,  negative = left
#   up_down     : positive = up,     negative = down
# Drone moves LEFT  → camera pans left  → ball shifts RIGHT in frame  (action 0)
# Drone moves RIGHT → camera pans right → ball shifts LEFT  in frame  (action 1)
# Drone moves UP    → camera rises      → ball shifts DOWN  in frame  (action 2)
# Drone moves DOWN  → camera falls      → ball shifts UP    in frame  (action 3)
# Hover             → no translation                                   (action 4)
_RC_COMMANDS = {
    0: (-LR_SPEED, 0, 0,         0),   # move_left
    1: (+LR_SPEED, 0, 0,         0),   # move_right
    2: (0,         0, +UD_SPEED, 0),   # move_up
    3: (0,         0, -UD_SPEED, 0),   # move_down
    4: (0,         0, 0,         0),   # hover
}


def execute_action(drone, action, dry_run=False):
    """
    Execute one action on the drone and wait for the step to complete.

    Step timing:
        Hover  → wait STEP_DURATION (no movement phase)
        Others → send RC command for MOVE_DURATION
                 stop RC command
                 wait STABILIZE_DURATION

    dry_run=True prints the command but does not call send_rc_control.
    """
    lr, fb, ud, yaw = _RC_COMMANDS[action]
    name            = ACTION_NAMES[action]

    if action == 4:   # hover
        print(f"    [HOVER]  holding position  ({STEP_DURATION:.1f}s)")
        if not dry_run:
            drone.send_rc_control(0, 0, 0, 0)
        time.sleep(STEP_DURATION)

    else:
        print(f"    [MOVE ]  send_rc_control({lr:>4},{fb:>2},{ud:>4},{yaw:>2})"
              f"  → {name}  ({MOVE_DURATION:.1f}s)")
        if not dry_run:
            drone.send_rc_control(lr, fb, ud, yaw)
        time.sleep(MOVE_DURATION)

        print(f"    [STOP ]  send_rc_control(0,0,0,0)  stabilizing ({STABILIZE_DURATION:.1f}s)")
        if not dry_run:
            drone.send_rc_control(0, 0, 0, 0)
        time.sleep(STABILIZE_DURATION)


# =============================================================================
# HUD overlay
# =============================================================================

def draw_overlay(frame, state, action, reward, step, Q):
    h, w = frame.shape[:2]

    # Dim top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Grid lines
    for i in range(1, 3):
        cv2.line(frame, (w*i//3, 0),   (w*i//3, h),   (80, 80, 80), 1)
        cv2.line(frame, (0, h*i//3),   (w, h*i//3),   (80, 80, 80), 1)

    # Highlight active cell
    if state != LOST:
        r, c = _decode(state)
        x1, y1 = c*w//3,     r*h//3
        x2, y2 = (c+1)*w//3, (r+1)*h//3
        color = (0, 255, 0) if state == 4 else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Text
    a_str = ACTION_NAMES[action] if action is not None else "---"
    r_str = f"{reward:+.1f}" if reward is not None else "--"
    cv2.putText(frame, f"Step:{step:>3}",
                (8, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    cv2.putText(frame, f"State: {state_label(state)}",
                (8, 42),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
    cv2.putText(frame, f"Action: {a_str:<13}  Reward: {r_str}",
                (8, 64),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,100), 1)

    # Q-value bars for current state
    if state < NUM_STATES:
        q_row  = Q[state]
        q_min, q_max  = q_row.min(), q_row.max()
        q_range = max(q_max - q_min, 1e-6)
        bar_w, bar_h  = 55, 12
        bar_y         = h - 28
        for a in range(NUM_ACTIONS):
            fill  = int((q_row[a] - q_min) / q_range * bar_w)
            bx    = 8 + a * (bar_w + 6)
            color = (0, 255, 80) if a == int(np.argmax(q_row)) else (130,130,130)
            cv2.rectangle(frame, (bx, bar_y), (bx+fill, bar_y+bar_h), color, -1)
            cv2.rectangle(frame, (bx, bar_y), (bx+bar_w, bar_y+bar_h), (180,180,180), 1)
            label = ACTION_NAMES[a][:3].upper()
            cv2.putText(frame, label, (bx+1, bar_y-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220,220,220), 1)
    return frame


# =============================================================================
# One step: observe → decide → execute → observe result
# =============================================================================

def run_step(drone, Q, frame_read, step, fw=FRAME_W, fh=FRAME_H, dry_run=False):
    """
    Runs one full MDP step. Returns (action, new_state, reward, done, annotated_frame).

    Sequence:
        1. Capture frame → discretize → current state
        2. Greedy action from Q-table
        3. Execute action (timed)
        4. Capture frame → discretize → next state
        5. Compute reward
    """
    # --- Observe current state -----------------------------------------------
    frame    = frame_read.frame
    centroid, _, display = detect_ball(frame)
    state    = discretize(*centroid, fw=fw, fh=fh) if centroid is not None else LOST

    # --- Select greedy action ------------------------------------------------
    action   = int(np.argmax(Q[state]))

    print(f"\n  ┌─ Step {step:>2} ──────────────────────────────────────────")
    print(f"  │  Observed : {state_label(state)}")
    print(f"  │  Action   : {ACTION_NAMES[action]}")

    # --- Execute action ------------------------------------------------------
    execute_action(drone, action, dry_run=dry_run)

    # --- Observe result (retry up to 5 frames before declaring LOST) ---------
    # A single missed detection frame should not end the episode — the ball
    # may be briefly out of view during the movement phase.
    centroid = None
    display  = None
    for attempt in range(5):
        frame    = frame_read.frame
        centroid, _, display = detect_ball(frame)
        if centroid is not None:
            break
        time.sleep(0.1)

    new_state = discretize(*centroid, fw=fw, fh=fh) if centroid is not None else LOST
    reward    = get_reward(new_state)
    done      = (new_state == LOST) or (step >= MAX_STEPS_PER_EP)

    print(f"  │  Result   : {state_label(new_state)}  reward={reward:+.1f}"
          + ("  [LOST — episode ends]" if new_state == LOST else ""))
    print(f"  └──────────────────────────────────────────────────────")

    display = draw_overlay(display, new_state, action, reward, step, Q)
    return action, new_state, reward, done, display


# =============================================================================
# TEST MODE 1 — Perception only (no drone)
# =============================================================================

def test_perception():
    """
    Opens laptop webcam (or press 0 to use a video file).
    Shows the color detector + 3x3 grid overlay + state in terminal.
    No drone connection required.
    """
    print("\n=== PERCEPTION TEST ===")
    print("Using laptop webcam (index 0).")
    print("Hold the lime green ball in view. Press Q to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check index.")
        return

    # Override frame dimensions for webcam
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {w}x{h}")
    print("Note: discretize() uses FRAME_W/FRAME_H from this file.")
    print("For accurate zone mapping the webcam resolution should match")
    print(f"or update FRAME_W={FRAME_W}, FRAME_H={FRAME_H} at the top.\n")

    prev_state = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        centroid, area, display = detect_ball(frame)

        if centroid is not None:
            cx, cy = centroid
            # Use actual frame size for this test
            col   = min(int(cx // (w / 3)), 2)
            row   = min(int(cy // (h / 3)), 2)
            state = row * 3 + col
        else:
            state = LOST

        # Draw grid overlay
        for i in range(1, 3):
            cv2.line(display, (w*i//3, 0), (w*i//3, h), (60,60,60), 1)
            cv2.line(display, (0, h*i//3), (w, h*i//3), (60,60,60), 1)

        # Highlight active cell
        if state != LOST:
            r, c  = _decode(state)
            x1,y1 = c*w//3,     r*h//3
            x2,y2 = (c+1)*w//3, (r+1)*h//3
            cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,255), 2)

        # State label
        label = state_label(state)
        cv2.putText(display, label, (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0) if state==4 else (0,255,255), 2)

        cv2.imshow("Perception Test — press Q to quit", display)

        if state != prev_state:
            print(f"  State: {label}" + ("  ← ball centered!" if state==4 else ""))
            prev_state = state

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Perception test done.")


# =============================================================================
# TEST MODE 2 — Command dry run (drone connected, no flight)
# =============================================================================

def test_commands(Q):
    """
    Connects to drone and streams video. Runs the full greedy policy loop
    but DRY RUN — never calls takeoff() or send_rc_control().
    Prints exactly what each step would do.
    Press Q to quit.
    """
    from djitellopy import Tello

    print("\n=== COMMAND DRY RUN ===")
    print("Connecting to drone (no flight — commands printed only).")
    print("Press Q in the video window to quit.\n")

    drone = Tello()
    drone.connect()
    print(f"Battery: {drone.get_battery()}%  ✓")
    drone.streamon()
    frame_read = drone.get_frame_read()
    time.sleep(1)

    # Capture actual stream dimensions for correct discretization
    while True:
        frame = frame_read.frame
        if frame is not None and frame.shape[0] > 0:
            break
    actual_h, actual_w = frame.shape[:2]
    print(f"Stream: {actual_w}x{actual_h}  "
          f"(zones: col at {actual_w//3}px, {2*actual_w//3}px  "
          f"row at {actual_h//3}px, {2*actual_h//3}px)")

    step = 0
    try:
        while True:
            step += 1
            action, new_state, reward, done, display = run_step(
                drone, Q, frame_read, step, fw=actual_w, fh=actual_h, dry_run=True
            )
            cv2.imshow("Dry Run — press Q to quit", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if done:
                print("\n  [Episode would end here — resetting step counter]")
                step = 0
    finally:
        drone.streamoff()
        cv2.destroyAllWindows()
        print("Dry run complete.")


# =============================================================================
# MAIN FLIGHT LOOP
# =============================================================================

def fly(Q):
    """
    Full deployment flight.

    State machine:
        WAITING  → T key pressed  → TAKEOFF
        TAKEOFF  → stabilised     → EPISODE
        EPISODE  → LOST or steps  → LAND
        LAND     → always         → SAVE + EXIT
    """
    from djitellopy import Tello

    # --- Setup data directory ------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path   = os.path.join(DATA_DIR, f"flight_log_{timestamp}.csv")
    video_path = os.path.join(DATA_DIR, f"flight_video_{timestamp}.mp4")

    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step", "state", "action", "action_name",
        "new_state", "reward", "cumulative_reward"
    ])

    recorded_frames = []
    ep_rewards      = []
    ep_lengths      = []

    # --- Connect -------------------------------------------------------------
    drone = Tello()
    drone.connect()
    batt = drone.get_battery()
    print(f"\nBattery: {batt}%")
    if batt < 20:
        print("WARNING: Battery below 20%. Recommend charging before flight.")

    # Request 720p — prevents 400x300 fallback that breaks zone boundaries
    try:
        drone.set_video_resolution(Tello.RESOLUTION_720P)
        print("Video resolution set to 720p")
    except Exception as e:
        print(f"Note: could not set resolution ({e}) — using actual stream dimensions")

    drone.streamon()
    frame_read = drone.get_frame_read()

    # Wait for valid frame
    print("Waiting for video stream...")
    while True:
        frame = frame_read.frame
        if frame is not None and frame.shape[0] > 0:
            break
    actual_h, actual_w = frame.shape[:2]
    print(f"Stream ready: {actual_w}x{actual_h}")
    print(f"\nLogging  → {csv_path}")
    print(f"Video    → {video_path}")

    print("\n" + "="*56)
    print("  Press T in the video window to take off.")
    print("  Press L at any time for emergency land.")
    print("="*56 + "\n")

    taken_off = False
    landed    = False

    try:
        # --- Wait for T keypress ---------------------------------------------
        # [STATE: WAITING]
        while not taken_off:
            frame = frame_read.frame
            cv2.putText(frame, "Press T to take off  |  L to cancel",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("Tello — waiting", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('t'):
                taken_off = True
            elif key == ord('l'):
                print("Cancelled.")
                return

        # --- Takeoff ---------------------------------------------------------
        # [STATE: TAKEOFF]
        # drone.takeoff() blocks the main thread waiting for a response (up to
        # 20s on first command after WiFi connect), which freezes the cv2 window
        # and causes the macOS beach ball. Running it in a background thread
        # keeps the window alive while the drone takes off.
        print(f"\n[TAKEOFF] Taking off — window may update slowly, this is normal...")
        takeoff_error = []

        def _do_takeoff():
            try:
                drone.takeoff()
            except Exception as e:
                takeoff_error.append(str(e))

        t = threading.Thread(target=_do_takeoff, daemon=True)
        t.start()

        # Keep cv2 window alive while takeoff completes
        while t.is_alive():
            frame = frame_read.frame
            if frame is not None:
                cv2.putText(frame.copy(), "Taking off...",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 165, 255), 2)
                cv2.imshow("Tello — waiting", frame)
            cv2.waitKey(100)

        t.join()
        if takeoff_error:
            print(f"[TAKEOFF] Warning: {takeoff_error[0]}")
        else:
            print("[TAKEOFF] Airborne.")

        print(f"[TAKEOFF] Stabilising ({TAKEOFF_WAIT:.0f}s)...")
        time.sleep(TAKEOFF_WAIT)
        print("[TAKEOFF] Ready.\n")

        # --- ARMED: wait for ball before episode begins ---------------------
        # [STATE: ARMED]
        # Episode does not start until the ball appears in frame.
        # Gives time to position the ball after takeoff stabilisation
        # without burning episode steps. Press L to abort.
        print("\n" + "="*56)
        print("  ARMED — hold ball in front of camera to begin")
        print("  Episode starts the moment the ball is first detected.")
        print("  Press L to abort and land.")
        print("="*56)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('l'):
                print("\n[ABORT] L pressed during armed stage — landing.")
                try:
                    drone.land()
                except Exception as e:
                    # 'error' response means drone already landed — safe to ignore
                    print(f"[ABORT] land response: {e} (drone may already be grounded)")
                landed = True
                return

            frame    = frame_read.frame
            centroid, _, display = detect_ball(frame)

            cv2.putText(display,
                        "ARMED — show ball to start  |  L = abort",
                        (10, actual_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.imshow("Tello — ARMED", display)

            if centroid is not None:
                init_state = discretize(*centroid, fw=actual_w, fh=actual_h)
                print(f"\n[GO] Ball detected at {state_label(init_state)} — episode starting!")
                break

        # --- Episode ---------------------------------------------------------
        # [STATE: EPISODE]
        print("\n" + "="*56)
        print("  EPISODE START")
        print(f"  Policy: SARSA greedy  |  MAX_STEPS={MAX_STEPS_PER_EP}")
        print(f"  Step duration: {STEP_DURATION:.1f}s  "
              f"(move {MOVE_DURATION:.1f}s + stabilise {STABILIZE_DURATION:.1f}s)")
        print(f"  Stream: {actual_w}x{actual_h}  "
              f"zones at {actual_w//3}px/{2*actual_w//3}px  "
              f"{actual_h//3}px/{2*actual_h//3}px")
        print("="*56)

        total_reward = 0.0

        for step in range(1, MAX_STEPS_PER_EP + 1):

            # Emergency land check (non-blocking)
            if cv2.waitKey(1) & 0xFF == ord('l'):
                print("\n[EMERGENCY] L pressed — landing.")
                drone.land()
                landed = True
                return

            action, new_state, reward, done, display = run_step(
                drone, Q, frame_read, step, fw=actual_w, fh=actual_h, dry_run=False
            )
            total_reward += reward

            # Log
            csv_writer.writerow([
                step, state_label(new_state), action,
                ACTION_NAMES[action], new_state,
                round(reward, 3), round(total_reward, 3)
            ])
            recorded_frames.append(display.copy())
            cv2.imshow("Tello — live", display)

            ep_lengths.append(step)

            if done:
                break

        ep_rewards.append(total_reward)

        print(f"\n{'='*56}")
        print(f"  EPISODE COMPLETE")
        print(f"  Steps: {step}  |  Total reward: {total_reward:+.2f}")
        lost_pct = sum(1 for r in ep_lengths if r < MAX_STEPS_PER_EP) / 1 * 100
        print(f"  Ended by: {'LOST' if new_state == LOST else 'timeout'}")
        print(f"{'='*56}\n")

        # --- Land ------------------------------------------------------------
        # [STATE: LAND]
        print("[LAND] Landing...")
        drone.land()
        landed = True
        print("[LAND] Done.")

    finally:
        # Land only if not already landed — prevents redundant land errors
        if not landed:
            try:
                drone.land()
            except Exception:
                pass

        # Save video
        if recorded_frames:
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                8,
                (actual_w, actual_h)
            )
            for f in recorded_frames:
                writer.write(f)
            writer.release()
            print(f"\nVideo saved  → {video_path}  ({len(recorded_frames)} frames)")

        csv_file.close()
        drone.streamoff()
        cv2.destroyAllWindows()

        print(f"Log saved    → {csv_path}")
        if ep_rewards:
            print(f"\nFlight summary:")
            print(f"  Total reward : {ep_rewards[0]:+.2f}")
            print(f"  Steps taken  : {step}")
            print(f"  Episode ended: {'LOST' if new_state == LOST else 'timeout (MAX_STEPS)'}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tello ball tracking — SARSA deployment")
    parser.add_argument(
        "--test",
        choices=["perception", "commands"],
        default=None,
        help="perception: test color detection (no drone)  |  "
             "commands: dry run policy loop (drone connected, no flight)"
    )
    args = parser.parse_args()

    if args.test == "perception":
        test_perception()
        sys.exit(0)

    # Load Q-table for all other modes
    if not os.path.exists(Q_TABLE_PATH):
        print(f"ERROR: Q-table not found at '{Q_TABLE_PATH}'.")
        print("Run pretrain.py first to generate q_table_sarsa.npy.")
        sys.exit(1)

    Q = np.load(Q_TABLE_PATH)
    print(f"Loaded Q-table: {Q_TABLE_PATH}  shape={Q.shape}  "
          f"(NUM_STATES={Q.shape[0]}, NUM_ACTIONS={Q.shape[1]})")

    if Q.shape != (NUM_STATES, NUM_ACTIONS):
        print(f"ERROR: Q-table shape {Q.shape} does not match "
              f"expected ({NUM_STATES}, {NUM_ACTIONS}).")
        print("Regenerate q_table_sarsa.npy using the current sim_env.py.")
        sys.exit(1)

    if args.test == "commands":
        test_commands(Q)
    else:
        fly(Q)