# =============================================================================
# drone_qlearn_withdepth.py
# Real-world tabular Q-learning on the DJI Tello for red ball tracking.
# Extends drone_qlearn.py with a distance dimension: blob area is used to
# discretize how far the ball is from the drone (FAR / IN_RANGE / CLOSE).
# Loads the pre-trained Q-table from pretrain_withdepth.py and fine-tunes
# it on hardware.
#
# Before running:
#   1. Run red_detect.py and hold the ball at your desired near/far distances
#      to measure real blob areas — fill in AREA_FAR_THRESHOLD and
#      AREA_CLOSE_THRESHOLD below before flying
#   2. Connect your laptop to the Tello WiFi network
#   3. Hold the red ball in front of the drone camera
#   4. Run: python drone_qlearn_withdepth.py
#   5. Press T in the OpenCV window to take off and begin
#   6. Press L at any time for emergency land
# =============================================================================

# =============================================================================
# HYPERPARAMETERS — tweak these before flying
# =============================================================================

# Q-learning
ALPHA               = 0.1       # learning rate — lower than sim; real transitions are noisier
GAMMA               = 0.95      # discount factor — same as sim
EPSILON_START       = 0.3       # start lower than sim since Q-table is pre-trained
EPSILON_END         = 0.05      # floor
EPSILON_DECAY       = 0.995     # slower decay than sim — real episodes are expensive

# Episode structure
MAX_STEPS_PER_EP    = 30        # max steps before ending episode and re-centering
MAX_EPISODES        = 200       # total real-world training episodes

# Action execution — how long each control burst lasts before hovering
ACTION_DURATION     = 0.4       # seconds the drone moves before stopping
HOVER_DURATION      = 0.3       # seconds to stabilize after each action

# Action speeds — sent to send_rc_control()
YAW_SPEED           = 30        # degrees/sec (0–100)
UP_DOWN_SPEED       = 25        # cm/s     (0–100)
FORWARD_SPEED       = 20        # cm/s     (0–100) — conservative for first flights

# Reward values — position component (mirror sim_env_withdepth.py exactly)
R_CENTER            = +1.0
R_ADJACENT          = +0.3
R_CORNER            = -0.1
R_LOST              = -1.0

# Reward values — distance component (added on top of position reward)
R_IN_RANGE          =  0.0      # distance is good — no bonus/penalty
R_TOO_FAR           = -0.2      # ball is shrinking — drone should move forward
R_TOO_CLOSE         = -0.2      # ball is very large — drone is too close

# Distance thresholds — FILL THESE IN after measuring with red_detect.py
# Hold the ball at your desired maximum follow distance → read printed area → AREA_FAR_THRESHOLD
# Hold the ball uncomfortably close to the drone       → read printed area → AREA_CLOSE_THRESHOLD
AREA_FAR_THRESHOLD   = None     # e.g. 4000  — area below this → FAR bin
AREA_CLOSE_THRESHOLD = None     # e.g. 30000 — area above this → CLOSE bin

# Minimum contour area to count as a valid ball detection (filters noise)
MIN_BLOB_AREA       = 500

# Camera frame dimensions (Tello default)
FRAME_W             = 960
FRAME_H             = 720

# How often to save the Q-table (in episodes)
SAVE_INTERVAL       = 10
SAVE_PATH           = "q_table_realworld_withdepth.npy"
PRETRAINED_PATH     = "q_table_pretrained_withdepth.npy"

# =============================================================================
# END HYPERPARAMETERS
# =============================================================================

import cv2
import numpy as np
import time
from djitellopy import Tello
from green_detect import detect_ball
from sim_env_withdepth import (
    LOST,
    NUM_STATES,
    NUM_ACTIONS,
    ACTION_NAMES,
    FAR,
    IN_RANGE,
    CLOSE,
    encode,
    decode,
    _decode,
    _CELL_TYPE,
)

# ---------------------------------------------------------------------------
# Startup check — catch missing thresholds before the drone connects
# ---------------------------------------------------------------------------
if AREA_FAR_THRESHOLD is None or AREA_CLOSE_THRESHOLD is None:
    raise ValueError(
        "\nAREA_FAR_THRESHOLD and AREA_CLOSE_THRESHOLD are not set.\n"
        "Run red_detect.py, hold the ball at your desired near/far distances,\n"
        "read the printed contour area values, and fill them in before flying."
    )

# ---------------------------------------------------------------------------
# Reward lookup
# ---------------------------------------------------------------------------

_POS_REWARD_MAP = {
    "center":   R_CENTER,
    "adjacent": R_ADJACENT,
    "corner":   R_CORNER,
}

_DIST_REWARD_MAP = {
    FAR:      R_TOO_FAR,
    IN_RANGE: R_IN_RANGE,
    CLOSE:    R_TOO_CLOSE,
}

_DIST_LABELS = {
    FAR:      "FAR",
    IN_RANGE: "IN_RANGE",
    CLOSE:    "CLOSE",
}


def get_reward(state):
    """
    Return reward for landing in a given state.
    Total reward = position component + distance component.
    Mirrors sim_env_withdepth reward structure so Q-values stay on the same scale.
    """
    if state == LOST:
        return R_LOST
    row, col, dist_bin = decode(state)
    cell_type    = _CELL_TYPE[(row, col)]
    pos_reward   = _POS_REWARD_MAP[cell_type]
    dist_reward  = _DIST_REWARD_MAP[dist_bin]
    return pos_reward + dist_reward


# ---------------------------------------------------------------------------
# Perception layer
# ---------------------------------------------------------------------------




def discretize_position(cx, cy):
    """
    Map pixel centroid (cx, cy) to grid indices (row, col).
    Divides the frame into a 3x3 grid — mirrors the sim state space exactly.
    col 0 = left third, col 1 = center third, col 2 = right third.
    row 0 = top third,  row 1 = center third, row 2 = bottom third.
    """
    col = int(cx // (FRAME_W / 3))
    row = int(cy // (FRAME_H / 3))
    col = min(col, 2)   # guard against cx == FRAME_W exactly
    row = min(row, 2)
    return row, col


def discretize_distance(area):
    """
    Map blob contour area to a distance bin (FAR / IN_RANGE / CLOSE).
    Thresholds are set in the hyperparameters block above — fill them in
    after measuring with red_detect.py before flying.
    FAR      → area < AREA_FAR_THRESHOLD   (ball is small, drone too far)
    IN_RANGE → AREA_FAR_THRESHOLD <= area <= AREA_CLOSE_THRESHOLD
    CLOSE    → area > AREA_CLOSE_THRESHOLD (ball is large, drone too close)
    """
    if area < AREA_FAR_THRESHOLD:
        return FAR
    elif area > AREA_CLOSE_THRESHOLD:
        return CLOSE
    else:
        return IN_RANGE


def get_state(cx, cy, area):
    """
    Combine position and distance into a single integer state index.
    Returns LOST if centroid or area is None (ball not detected).
    """
    if cx is None or area is None:
        return LOST
    row, col = discretize_position(cx, cy)
    dist_bin = discretize_distance(area)
    return encode(row, col, dist_bin)


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

def execute_action(drone, action):
    """
    Send a timed RC control burst for the chosen action, then hover.
    send_rc_control(left_right, forward_backward, up_down, yaw)
    All values: -100 to 100.
    """
    # Map action index to RC control values.
    # Mirrors ACTION_DELTAS sign convention from sim_env_withdepth.py:
    #   yaw_right    → ball moves left  in frame → positive yaw
    #   yaw_left     → ball moves right in frame → negative yaw
    #   move_up      → ball moves down  in frame → positive up_down
    #   move_down    → ball moves up    in frame → negative up_down
    #   move_forward → drone closes distance     → positive forward_backward
    rc_commands = {
        0: (0,              0, 0,             YAW_SPEED),    # yaw_right
        1: (0,              0, 0,            -YAW_SPEED),    # yaw_left
        2: (0,              0, UP_DOWN_SPEED,  0),            # move_up
        3: (0,              0,-UP_DOWN_SPEED,  0),            # move_down
        4: (0, FORWARD_SPEED, 0,               0),            # move_forward
    }

    lr, fb, ud, yaw = rc_commands[action]

    # Send movement command
    drone.send_rc_control(lr, fb, ud, yaw)
    time.sleep(ACTION_DURATION)

    # Stop and hover to let the drone stabilize before next observation
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(HOVER_DURATION)


# ---------------------------------------------------------------------------
# Display overlay
# ---------------------------------------------------------------------------

def draw_overlay(frame, state, action, reward, episode, step, epsilon, Q):
    """
    Draw a minimal HUD on the live frame so we can monitor the agent
    without looking at the terminal during flight.
    Shows: current grid state, distance bin, chosen action, last reward,
    episode/step, epsilon, and a mini Q-value bar for the current state.
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark bar at top for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw the 3x3 grid lines so it's clear which zone the ball is in
    for i in range(1, 3):
        cv2.line(frame, (w * i // 3, 0), (w * i // 3, h), (100, 100, 100), 1)
        cv2.line(frame, (0, h * i // 3), (w, h * i // 3), (100, 100, 100), 1)

    # Highlight the current cell in the grid
    if state != LOST:
        row, col, dist_bin = decode(state)
        x1, y1 = col * w // 3, row * h // 3
        x2, y2 = (col + 1) * w // 3, (row + 1) * h // 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        dist_label = _DIST_LABELS[dist_bin]
    else:
        dist_label = "—"

    # Text HUD
    action_str = ACTION_NAMES[action] if action is not None else "none"
    state_str  = (f"s={state} {_decode(state)}" if state != LOST else "s=LOST")
    reward_str = f"{reward:+.2f}" if reward is not None else "--"

    cv2.putText(frame, f"Ep:{episode}  Step:{step}  e:{epsilon:.3f}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"State: {state_str}   Dist: {dist_label}",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(frame, f"Action: {action_str}   Reward: {reward_str}",
                (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 1)

    # Mini Q-value bar for current state — shows relative preference across actions
    if state < NUM_STATES:
        bar_y = h - 30
        q_row = Q[state]
        q_min, q_max = q_row.min(), q_row.max()
        q_range = max(q_max - q_min, 1e-6)
        bar_w = 55
        for a in range(NUM_ACTIONS):
            fill  = int((q_row[a] - q_min) / q_range * bar_w)
            bx    = 10 + a * (bar_w + 8)
            color = (0, 255, 0) if a == int(np.argmax(q_row)) else (150, 150, 150)
            cv2.rectangle(frame, (bx, bar_y), (bx + fill, bar_y + 14), color, -1)
            cv2.rectangle(frame, (bx, bar_y), (bx + bar_w, bar_y + 14),
                          (200, 200, 200), 1)
            cv2.putText(frame, ACTION_NAMES[a][:3],
                        (bx, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (255, 255, 255), 1)

    return frame


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():

    # --- Load pre-trained Q-table -------------------------------------------
    # Warm-start from sim rather than zeros — the agent begins with a
    # directionally correct centering policy on its first real flight
    Q = np.load(PRETRAINED_PATH)
    print(f"Loaded pre-trained Q-table from {PRETRAINED_PATH}  shape={Q.shape}")
    print(f"Distance thresholds — FAR below: {AREA_FAR_THRESHOLD}  "
          f"CLOSE above: {AREA_CLOSE_THRESHOLD}")

    epsilon = EPSILON_START

    # Per-episode tracking for post-flight analysis
    ep_rewards = []
    ep_lengths = []

    # --- Connect to Tello ----------------------------------------------------
    drone = Tello()
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")

    drone.streamon()
    frame_read = drone.get_frame_read()

    print("\nPress T in the video window to take off and begin training.")
    print("Press L at any time to land immediately.\n")

    taken_off = False

    try:
        # --- Wait for takeoff keypress before starting training loop --------
        while not taken_off:
            frame = frame_read.frame
            cv2.imshow("Tello — press T to take off", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('t'):
                drone.takeoff()
                time.sleep(2)   # let the drone stabilize at hover altitude
                taken_off = True
            if key == ord('l'):
                print("Landing.")
                break

        if not taken_off:
            return

        # =====================================================================
        # Episode loop
        # =====================================================================
        for episode in range(1, MAX_EPISODES + 1):

            total_reward = 0.0
            step         = 0
            action       = None
            reward       = None

            print(f"\n--- Episode {episode} / {MAX_EPISODES}  ε={epsilon:.3f} ---")
            print("Position the red ball in the frame, then it will begin automatically.")
            time.sleep(1.5)   # brief pause for repositioning between episodes

            # Get initial state from first frame of this episode
            frame = frame_read.frame
            centroid, area, display = detect_ball(frame)
            cx, cy = centroid if centroid else (None, None)
            state  = get_state(cx, cy, area)

            # =================================================================
            # Step loop
            # =================================================================
            for step in range(1, MAX_STEPS_PER_EP + 1):

                # --- Epsilon-greedy action selection -------------------------
                # Use the pre-trained Q-table to pick the best known action,
                # or explore randomly with probability epsilon
                if np.random.random() < epsilon:
                    action = np.random.randint(0, NUM_ACTIONS)
                else:
                    action = int(np.argmax(Q[state]))

                # --- Execute action on drone ---------------------------------
                execute_action(drone, action)

                # --- Observe new state from camera ---------------------------
                frame = frame_read.frame
                centroid, area, display = detect_ball(frame)
                cx, cy     = centroid if centroid else (None, None)
                new_state  = get_state(cx, cy, area)

                # --- Compute reward ------------------------------------------
                reward = get_reward(new_state)
                total_reward += reward

                # --- Q-learning update ---------------------------------------
                # Standard Bellman update: same equation as pretrain_withdepth.py
                best_next        = np.max(Q[new_state])
                td_target        = reward + GAMMA * best_next
                td_error         = td_target - Q[state, action]
                Q[state, action] += ALPHA * td_error

                # --- Display annotated frame ---------------------------------
                display = draw_overlay(display, new_state, action, reward,
                                       episode, step, epsilon, Q)
                cv2.imshow("Tello Q-Learning (with depth)", display)

                # --- Log step -----------------------------------------------
                dist_str = (f"dist={_DIST_LABELS[decode(new_state)[2]]}"
                            if new_state != LOST else "LOST")
                print(f"  step {step:>3}  s={state}→{new_state}  {dist_str:<16}"
                      f"a={ACTION_NAMES[action]:<14}  r={reward:+.2f}  "
                      f"Q={Q[state].round(3)}")

                state = new_state

                # --- Check for emergency land keypress -----------------------
                key = cv2.waitKey(1) & 0xFF
                if key == ord('l'):
                    print("Emergency land triggered.")
                    drone.land()
                    return

                # --- End episode if ball is lost -----------------------------
                # LOST is the real-world equivalent of a crash in Anwar's system
                if new_state == LOST:
                    print(f"  Ball lost — ending episode.")
                    break

            # --- End of episode bookkeeping ----------------------------------
            ep_rewards.append(total_reward)
            ep_lengths.append(step)

            avg_r = (np.mean(ep_rewards[-10:]) if len(ep_rewards) >= 10
                     else np.mean(ep_rewards))
            print(f"Episode {episode} done  |  total_r={total_reward:+.2f}  "
                  f"length={step}  10-ep avg={avg_r:+.2f}")

            # Decay epsilon after each episode
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            # Periodically save the updated Q-table
            if episode % SAVE_INTERVAL == 0:
                np.save(SAVE_PATH, Q)
                print(f"  Q-table saved to {SAVE_PATH}")

        # --- Training complete — land and save final Q-table ----------------
        print("\nTraining complete. Landing.")
        drone.land()

    finally:
        # Always land safely — mirrors Anwar's try/finally crash recovery
        # This block runs even if an exception is thrown mid-flight
        print("Executing safe landing in finally block.")
        try:
            drone.land()
        except Exception:
            pass
        drone.streamoff()
        cv2.destroyAllWindows()
        np.save(SAVE_PATH, Q)
        print(f"Final Q-table saved to {SAVE_PATH}")
        print(f"\nFlight summary: {len(ep_rewards)} episodes  "
              f"avg reward={np.mean(ep_rewards):+.2f}  "
              f"avg length={np.mean(ep_lengths):.1f}")


if __name__ == "__main__":
    main()