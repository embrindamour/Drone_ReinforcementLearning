# =============================================================================
# sim_env.py
# Simulated MDP for red ball tracking (pre-training).
# State space: 3x3 grid position × 3 distance bins + LOST = 28 states
# Distance bins: FAR (ball too small), IN_RANGE (good follow distance), CLOSE
# =============================================================================

# =============================================================================
# HYPERPARAMETERS — tweak these before pre-training
# =============================================================================

# Probability that the executed action slips one extra step in a random direction
# Models imperfect drone response (WiFi lag, motor jitter)
P_ACTION_SLIP   = 0.10

# Probability that the ball drifts one cell in a random direction each step
# Models the person slowly moving the ball
P_BALL_DRIFT    = 0.20

# Probability that the ball distance bin shifts by one each step
# Models the person walking toward/away from the drone
P_DIST_DRIFT    = 0.15

# Reward values — position component
R_CENTER        = +1.0   # ball is in center cell (1,1)
R_ADJACENT      = +0.3   # ball in an edge-adjacent cell
R_CORNER        = -0.1   # ball in a corner cell
R_LOST          = -1.0   # ball not in frame

# Reward values — distance component (added on top of position reward)
R_IN_RANGE      =  0.0   # distance is good — no bonus/penalty
R_TOO_FAR       = -0.2   # ball is shrinking — drone should move forward
R_TOO_CLOSE     = -0.2   # ball is very large — drone is too close

# Max steps per episode before forced reset
MAX_STEPS       = 50

# When recovering from LOST: ball re-spawns on a random edge cell (not corners)
# Set to True to also allow corner re-spawns (harder recovery)
RECOVER_TO_CORNERS = False

# Action direction conventions — FLIP THESE if real-world behavior is inverted
# Position actions: (delta_row, delta_col) shift of the ball in the frame
# Distance actions: delta_dist_bin (negative = getting closer)
ACTION_DELTAS = {
    0: ( 0, -1,  0),   # yaw_right    → ball shifts left  in frame
    1: ( 0, +1,  0),   # yaw_left     → ball shifts right in frame
    2: (+1,  0,  0),   # move_up      → ball shifts down  in frame
    3: (-1,  0,  0),   # move_down    → ball shifts up    in frame
    4: ( 0,  0, -1),   # move_forward → distance bin decrements (getting closer)
}

ACTION_NAMES = {
    0: "yaw_right",
    1: "yaw_left",
    2: "move_up",
    3: "move_down",
    4: "move_forward",
}

# =============================================================================
# END HYPERPARAMETERS
# =============================================================================

import numpy as np

# ---- Distance bins ----------------------------------------------------------
# 0 = FAR    (ball too small, drone needs to move forward)
# 1 = IN_RANGE (good following distance)
# 2 = CLOSE  (ball too large, drone is too close)
FAR      = 0
IN_RANGE = 1
CLOSE    = 2
NUM_DIST_BINS = 3

_DIST_REWARD = {
    FAR:      R_TOO_FAR,
    IN_RANGE: R_IN_RANGE,
    CLOSE:    R_TOO_CLOSE,
}

# ---- State encoding ---------------------------------------------------------
# States 0–26: grid cell (row, col) × distance bin
#   state = (row * 3 + col) * NUM_DIST_BINS + dist_bin
# State 27: LOST
LOST       = 27
NUM_STATES = 28
NUM_ACTIONS = len(ACTION_DELTAS)

# Cell type labels (used for position reward lookup)
CENTER   = "center"
ADJACENT = "adjacent"
CORNER   = "corner"

_CELL_TYPE = {
    (0, 0): CORNER,   (0, 1): ADJACENT, (0, 2): CORNER,
    (1, 0): ADJACENT, (1, 1): CENTER,   (1, 2): ADJACENT,
    (2, 0): CORNER,   (2, 1): ADJACENT, (2, 2): CORNER,
}

_POS_REWARD = {
    CENTER:   R_CENTER,
    ADJACENT: R_ADJACENT,
    CORNER:   R_CORNER,
}

# Edge cells for LOST recovery spawns
_EDGE_CELLS     = [(r, c) for (r, c), t in _CELL_TYPE.items() if t == ADJACENT]
_ALL_CELLS      = list(_CELL_TYPE.keys())
_NONCENTER_CELLS = [(r, c) for (r, c) in _ALL_CELLS if (r, c) != (1, 1)]


def encode(row, col, dist_bin):
    """(row, col, dist_bin) → integer state index 0–26"""
    return (row * 3 + col) * NUM_DIST_BINS + dist_bin


def decode(state):
    """integer state index 0–26 → (row, col, dist_bin)"""
    grid_idx = state // NUM_DIST_BINS
    dist_bin = state  % NUM_DIST_BINS
    row, col = divmod(grid_idx, 3)
    return row, col, dist_bin


# Keep _decode as a position-only helper for display/logging compatibility
def _decode(state):
    """integer state → (row, col) ignoring distance bin. Used for display."""
    if state == LOST:
        return (None, None)
    row, col, _ = decode(state)
    return row, col


class BallTrackingEnv:
    """
    Tabular MDP for red ball tracking with distance awareness.

    Observation: integer in {0, ..., 27}
                 0–26 → encoded (row, col, dist_bin)
                 27   → LOST (ball not in frame)

    Action: integer in {0, 1, 2, 3, 4}
            see ACTION_NAMES / ACTION_DELTAS above

    Usage:
        env = BallTrackingEnv()
        state = env.reset()
        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            if done:
                break
    """

    def __init__(self, seed=None):
        self.rng   = np.random.default_rng(seed)
        self.state = None
        self.steps = 0

    # ------------------------------------------------------------------
    def reset(self):
        """
        Start episode with ball at a random non-center cell, IN_RANGE distance.
        Starting in-range means the agent must learn centering before
        it encounters distance correction — a mild curriculum effect.
        """
        r, c = self._random_noncenter_cell()
        self.state = encode(r, c, IN_RANGE)
        self.steps = 0
        return self.state

    # ------------------------------------------------------------------
    def step(self, action):
        """
        Returns (next_state, reward, done).
        done=True when ball is LOST or MAX_STEPS exceeded.
        """
        assert action in ACTION_DELTAS, f"Invalid action: {action}"
        self.steps += 1

        # ---- From LOST: attempt recovery (re-spawn on edge cell) --------
        if self.state == LOST:
            next_state = self._recovery_spawn()
            reward     = R_LOST
            done       = False
            self.state = next_state
            return next_state, reward, done

        # ---- Normal step ------------------------------------------------
        row, col, dist_bin = decode(self.state)

        dr, dc, dd = ACTION_DELTAS[action]

        # Action slip noise — models imperfect drone response
        if self.rng.random() < P_ACTION_SLIP:
            slip = self.rng.integers(0, NUM_ACTIONS)
            sr, sc, sd = ACTION_DELTAS[slip]
            dr += sr;  dc += sc;  dd += sd

        # Ball position drift — models target moving independently
        if self.rng.random() < P_BALL_DRIFT:
            drift = self.rng.integers(0, 4)   # position actions only
            fr, fc, _ = ACTION_DELTAS[drift]
            dr += fr;  dc += fc

        # Distance drift — models person walking toward/away from drone
        if self.rng.random() < P_DIST_DRIFT:
            dd += self.rng.choice([-1, 1])

        new_row  = row     + dr
        new_col  = col     + dc
        new_dist = dist_bin + dd

        # Check if ball left the frame
        if not (0 <= new_row <= 2 and 0 <= new_col <= 2):
            next_state = LOST
            reward     = R_LOST
            done       = True
            self.state = next_state
            return next_state, reward, done

        # Clamp distance bin — can't go beyond FAR or CLOSE
        new_dist = int(np.clip(new_dist, FAR, CLOSE))

        next_state = encode(new_row, new_col, new_dist)

        # Reward = position component + distance component
        cell_type = _CELL_TYPE[(new_row, new_col)]
        reward    = _POS_REWARD[cell_type] + _DIST_REWARD[new_dist]

        done = (self.steps >= MAX_STEPS)

        self.state = next_state
        return next_state, reward, done

    # ------------------------------------------------------------------
    def render(self):
        """Print a simple ASCII view of the current state."""
        print(f"\n  Step {self.steps}")
        if self.state == LOST:
            print("  [ LOST — ball not in frame ]")
            return
        row, col, dist_bin = decode(self.state)
        dist_label = {FAR: "FAR", IN_RANGE: "IN_RANGE", CLOSE: "CLOSE"}[dist_bin]
        for r in range(3):
            row_str = "  "
            for c in range(3):
                if r == row and c == col:
                    row_str += "[ O ]"
                elif r == 1 and c == 1:
                    row_str += "[ + ]"
                else:
                    row_str += "[   ]"
            print(row_str)
        print(f"  Distance: {dist_label}")

    # ------------------------------------------------------------------
    def _random_noncenter_cell(self):
        idx = self.rng.integers(0, len(_NONCENTER_CELLS))
        return _NONCENTER_CELLS[idx]

    def _recovery_spawn(self):
        """Re-spawn ball on edge (or edge+corner) after LOST, always IN_RANGE."""
        pool = _ALL_CELLS if RECOVER_TO_CORNERS else _EDGE_CELLS
        idx  = self.rng.integers(0, len(pool))
        r, c = pool[idx]
        return encode(r, c, IN_RANGE)


# =============================================================================
# Quick sanity check
# =============================================================================
if __name__ == "__main__":
    env = BallTrackingEnv(seed=42)
    state = env.reset()
    row, col, dist = decode(state)
    print("=== BallTrackingEnv sanity check ===")
    print(f"Initial state: {state}  → cell ({row},{col})  dist={dist}")
    print(f"NUM_STATES={NUM_STATES}  NUM_ACTIONS={NUM_ACTIONS}  LOST={LOST}")
    env.render()

    total_reward = 0
    for t in range(10):
        action = np.random.randint(0, NUM_ACTIONS)
        next_state, reward, done = env.step(action)
        ns_str = f"LOST" if next_state == LOST else str(decode(next_state))
        print(f"  t={t+1}  a={ACTION_NAMES[action]:<14} → {ns_str:<30} r={reward:+.2f}  done={done}")
        env.render()
        total_reward += reward
        if done:
            break

    print(f"\nTotal reward: {total_reward:+.2f}")
    print("\nEncode/decode roundtrip check:")
    for r in range(3):
        for c in range(3):
            for d in range(NUM_DIST_BINS):
                s = encode(r, c, d)
                assert decode(s) == (r, c, d), f"Mismatch at ({r},{c},{d})"
    print("  All roundtrips OK")