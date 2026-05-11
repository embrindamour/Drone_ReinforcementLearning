# =============================================================================
# sim_env.py
# Simulated 10-state MDP for red ball tracking (pre-training)
# States: 3x3 grid of ball centroid position + LOST state
#
# SETUP:- Tello streams a full 960x720 pixel frame
#       - 3x3 grid discretization of 9 possible zones:
#                | zone 0 | zone 1 | zone 2 |
#                | zone 3 | zone 4 | zone 5 |   (4 = center)
#                | zone 6 | zone 7 | zone 8 |
#       - Each zone covers roughly 320x240 pixels
#       - red_detect.py finds the centroid of the red blob at full resolution
#       - discretization maps (cx, cy) to state integer 0-8
#
# STATE DESIGN:
#   LOST (state 9) is TERMINAL — ball exiting the frame ends the episode
#   with R_LOST penalty, analogous to falling off the cliff in Sutton &
#   Barto Example 6.6. This preserves the Markov property: a single
#   non-terminal LOST state would require the agent to know which direction
#   the ball exited (history) to choose the correct recovery action.
#
#   CENTER (state 4) is NON-TERMINAL — the task is continuous tracking of
#   a moving target, not one-shot goal reaching. The +R_CENTER reward
#   incentivizes the agent to reach and stay at center without termination.
#
# ACTION DESIGN:
#   5 actions: move_left, move_right, move_up, move_down, hover.
#   Hover (delta = (0,0)) allows the agent to hold position when the ball
#   is already centered. Without hover, the agent is forced to pick a
#   directional action from center, which actively shifts the ball away.
#   On the real Tello, hover maps to sending no movement command.
# =============================================================================

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

P_ACTION_SLIP = 0.10    # prob of one extra random step (WiFi lag / motor jitter)
P_BALL_DRIFT  = 0.20    # prob of ball drifting one cell (person moving target)

R_CENTER   = +1.0
R_ADJACENT = +0.3
R_CORNER   = -0.1
R_LOST     = -1.0       # terminal

MAX_STEPS = 50          # episode timeout

# Action deltas: (delta_row, delta_col) shift of the BALL in the frame.
# drone moves LEFT  → camera left  → ball shifts RIGHT (+col)
# drone moves RIGHT → camera right → ball shifts LEFT  (-col)
# drone moves UP    → camera rises → ball shifts DOWN  (+row)
# drone moves DOWN  → camera falls → ball shifts UP    (-row)
# hover             → no movement  → ball unaffected   (0, 0)
ACTION_DELTAS = {
    0: ( 0, +1),   # move_left  → ball right
    1: ( 0, -1),   # move_right → ball left
    2: (+1,  0),   # move_up    → ball down
    3: (-1,  0),   # move_down  → ball up
    4: ( 0,  0),   # hover      → no shift
}

ACTION_NAMES = {
    0: "move_left",
    1: "move_right",
    2: "move_up",
    3: "move_down",
    4: "hover",
}

# =============================================================================
# END HYPERPARAMETERS
# =============================================================================

import numpy as np

LOST        = 9
NUM_STATES  = 10
NUM_ACTIONS = len(ACTION_DELTAS)   # 5

CENTER   = "center"
ADJACENT = "adjacent"
CORNER   = "corner"

_CELL_TYPE = {
    (0, 0): CORNER,
    (0, 1): ADJACENT,
    (0, 2): CORNER,
    (1, 0): ADJACENT,
    (1, 1): CENTER,
    (1, 2): ADJACENT,
    (2, 0): CORNER,
    (2, 1): ADJACENT,
    (2, 2): CORNER,
}

_REWARD_MAP = {
    CENTER:   R_CENTER,
    ADJACENT: R_ADJACENT,
    CORNER:   R_CORNER,
}

_ALL_CELLS       = list(_CELL_TYPE.keys())
_NONCENTER_CELLS = [(r, c) for (r, c) in _ALL_CELLS if (r, c) != (1, 1)]


def _encode(row, col):
    return row * 3 + col

def _decode(state):
    return divmod(state, 3)


class BallTrackingEnv:
    """
    Tabular MDP for red ball tracking.

    Observation : integer in {0,..,9}
                  0-8 → grid cell  (row = s//3, col = s%3)
                  9   → LOST — TERMINAL

    Action      : integer in {0,1,2,3,4}  (see ACTION_NAMES / ACTION_DELTAS)

    Episode ends when:
      (a) Ball exits frame → next_state == LOST, done=True, reward=R_LOST
      (b) MAX_STEPS reached → done=True (timeout)
    """

    def __init__(self, seed=None):
        self.rng   = np.random.default_rng(seed)
        self.state = None
        self.steps = 0

    def reset(self):
        """Start episode with ball at a random non-center cell."""
        r, c       = self._random_noncenter_cell()
        self.state = _encode(r, c)
        self.steps = 0
        return self.state

    def step(self, action):
        """
        Returns (next_state, reward, done).
        done=True on LOST (terminal) or MAX_STEPS (timeout).
        """
        assert action in ACTION_DELTAS, f"Invalid action: {action}"
        self.steps += 1

        row, col = _decode(self.state)
        dr, dc   = ACTION_DELTAS[action]

        # Action slip
        if self.rng.random() < P_ACTION_SLIP:
            slip_a = self.rng.integers(0, NUM_ACTIONS)
            sr, sc = ACTION_DELTAS[slip_a]
            dr += sr
            dc += sc

        # Ball drift
        if self.rng.random() < P_BALL_DRIFT:
            drift_a = self.rng.integers(0, NUM_ACTIONS)
            fr, fc  = ACTION_DELTAS[drift_a]
            dr += fr
            dc += fc

        new_row = row + dr
        new_col = col + dc

        # Ball exited frame — TERMINAL
        if not (0 <= new_row <= 2 and 0 <= new_col <= 2):
            self.state = LOST
            return LOST, R_LOST, True

        next_state = _encode(new_row, new_col)
        reward     = _REWARD_MAP[_CELL_TYPE[(new_row, new_col)]]
        done       = (self.steps >= MAX_STEPS)
        self.state = next_state
        return next_state, reward, done

    def render(self):
        print(f"\n  Step {self.steps}")
        if self.state == LOST:
            print("  [ LOST — ball not in frame ]")
            return
        ball_row, ball_col = _decode(self.state)
        for r in range(3):
            row_str = "  "
            for c in range(3):
                if r == ball_row and c == ball_col:
                    row_str += "[ O ]"
                elif r == 1 and c == 1:
                    row_str += "[ + ]"
                else:
                    row_str += "[   ]"
            print(row_str)

    def _random_noncenter_cell(self):
        idx = self.rng.integers(0, len(_NONCENTER_CELLS))
        return _NONCENTER_CELLS[idx]


# =============================================================================
# Sanity check
# =============================================================================
if __name__ == "__main__":
    env   = BallTrackingEnv(seed=42)
    state = env.reset()
    print("=== BallTrackingEnv sanity check ===")
    print(f"  NUM_STATES={NUM_STATES}  NUM_ACTIONS={NUM_ACTIONS}")
    print(f"  Actions: {list(ACTION_NAMES.values())}")
    print(f"  LOST=terminal  CENTER=non-terminal  hover=(0,0)")
    print(f"\nInitial state: {state} → cell {_decode(state)}")
    env.render()

    total_reward = 0
    for t in range(15):
        action = np.random.randint(0, NUM_ACTIONS)
        ns, r, done = env.step(action)
        print(f"\nAction: {ACTION_NAMES[action]:<13} → state {ns}  "
              f"reward {r:+.1f}  done {done}")
        env.render()
        total_reward += r
        state = ns
        if done:
            print("  [Episode ended]")
            break

    print(f"\nTotal reward: {total_reward:+.1f}  over {env.steps} steps")
    print("\nEncode/decode roundtrip:")
    for r in range(3):
        for c in range(3):
            s = _encode(r, c)
            assert _decode(s) == (r, c)
    print("  OK")