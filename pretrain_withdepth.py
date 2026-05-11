# =============================================================================
# pretrain_withdepth.py
# Tabular Q-learning on BallTrackingEnv (with depth) sim — produces a
# pre-trained Q-table to warm-start real-world training on the Tello.
# State space: 3x3 grid × 3 distance bins + LOST = 28 states
# Action space: yaw_right, yaw_left, move_up, move_down, move_forward
# =============================================================================

# =============================================================================
# HYPERPARAMETERS — tweak these before pre-training
# =============================================================================

NUM_EPISODES        = 8_000     # total training episodes in sim
MAX_STEPS_PER_EP    = 50        # should match sim_env_withdepth.MAX_STEPS

ALPHA               = 0.1       # learning rate
GAMMA               = 0.95      # discount factor

# Epsilon-greedy exploration schedule
EPSILON_START       = 1.0       # start fully random
EPSILON_END         = 0.05      # floor — never fully greedy
EPSILON_DECAY       = 0.999     # multiplied each episode

# How often to print a progress summary (in episodes)
LOG_INTERVAL        = 500

# Path to save the pre-trained Q-table
SAVE_PATH           = "q_table_pretrained_withdepth.npy"

# Random seed (set to None for non-deterministic runs)
SEED                = 42

# Smoothing window for reward/length curves in the visualization
SMOOTH_WINDOW       = 100

# =============================================================================
# END HYPERPARAMETERS
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sim_env_withdepth import (
    BallTrackingEnv,
    NUM_STATES,
    NUM_ACTIONS,
    ACTION_NAMES,
    LOST,
    FAR,
    IN_RANGE,
    CLOSE,
    encode,
    decode,
    _decode,
    _DIST_LABELS,
)


def train():
    rng = np.random.default_rng(SEED)
    env = BallTrackingEnv(seed=SEED)

    # Q-table initialized to zero — shape (NUM_STATES, NUM_ACTIONS)
    # NUM_STATES=28, NUM_ACTIONS=5 for the withdepth variant
    Q = np.zeros((NUM_STATES, NUM_ACTIONS))

    epsilon = EPSILON_START

    # Tracking stats
    ep_rewards     = []
    ep_lengths     = []
    ep_lost_counts = []   # how many LOST transitions per episode

    for episode in range(1, NUM_EPISODES + 1):

        state        = env.reset()
        total_reward = 0.0
        lost_count   = 0

        for step in range(MAX_STEPS_PER_EP):

            # --- Epsilon-greedy action selection -------------------------
            if rng.random() < epsilon:
                action = rng.integers(0, NUM_ACTIONS)
            else:
                action = int(np.argmax(Q[state]))

            # --- Step environment ----------------------------------------
            next_state, reward, done = env.step(action)

            # --- Q-learning update ---------------------------------------
            best_next = np.max(Q[next_state])
            td_target = reward + GAMMA * best_next
            td_error  = td_target - Q[state, action]
            Q[state, action] += ALPHA * td_error

            total_reward += reward
            if next_state == LOST:
                lost_count += 1

            state = next_state
            if done:
                break

        # --- Episode bookkeeping -----------------------------------------
        ep_rewards.append(total_reward)
        ep_lengths.append(step + 1)
        ep_lost_counts.append(lost_count)

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # --- Logging ---------------------------------------------------------
        if episode % LOG_INTERVAL == 0:
            window     = ep_rewards[-LOG_INTERVAL:]
            avg_reward = np.mean(window)
            avg_length = np.mean(ep_lengths[-LOG_INTERVAL:])
            avg_lost   = np.mean(ep_lost_counts[-LOG_INTERVAL:])
            print(
                f"Episode {episode:>6} / {NUM_EPISODES}  |  "
                f"ε={epsilon:.3f}  |  "
                f"avg reward={avg_reward:>+6.2f}  |  "
                f"avg length={avg_length:>5.1f}  |  "
                f"avg lost/ep={avg_lost:>4.2f}"
            )

    # -------------------------------------------------------------------------
    # Save Q-table
    # -------------------------------------------------------------------------
    np.save(SAVE_PATH, Q)
    print(f"\nQ-table saved to {SAVE_PATH}  (shape {Q.shape})")

    # -------------------------------------------------------------------------
    # Human-readable policy summary
    # Grouped by distance bin so it's easy to see how depth changes the policy
    # -------------------------------------------------------------------------
    print("\n=== Learned Policy ===")
    print(f"{'State':<32} {'Best Action':<16} {'Q-values'}")
    print("-" * 80)
    for dist_bin, dist_label in _DIST_LABELS.items():
        print(f"\n  -- Distance bin: {dist_label} --")
        for row in range(3):
            for col in range(3):
                s      = encode(row, col, dist_bin)
                best_a = int(np.argmax(Q[s]))
                qvals  = "  ".join(f"{v:+.3f}" for v in Q[s])
                label  = f"cell ({row},{col}) {dist_label:<9} s={s}"
                print(f"  {label:<30} {ACTION_NAMES[best_a]:<16} [{qvals}]")
    # LOST state
    best_a = int(np.argmax(Q[LOST]))
    qvals  = "  ".join(f"{v:+.3f}" for v in Q[LOST])
    print(f"\n  {'LOST (27)':<30} {ACTION_NAMES[best_a]:<16} [{qvals}]")

    # -------------------------------------------------------------------------
    # Sanity checks on the learned policy
    # -------------------------------------------------------------------------
    print("\n=== Policy Sanity Checks ===")
    _run_sanity_checks(Q)

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    _plot_training_curves(ep_rewards, ep_lengths)
    plt.show()

    return Q


def _run_sanity_checks(Q):
    """
    Verify the learned policy makes directional sense.
    These are soft checks — a warning doesn't mean training failed,
    but consistent failures suggest a hyperparameter or sign problem.
    """
    checks = [
        # (state, expected_action, description)
        # Position checks — tested at IN_RANGE distance so distance doesn't
        # interfere with the expected positional correction action
        (encode(1, 0, IN_RANGE), 1, "ball left,   in-range  → expect yaw_left     to recenter ball"),
        (encode(1, 2, IN_RANGE), 0, "ball right,  in-range  → expect yaw_right    to recenter ball"),
        (encode(0, 1, IN_RANGE), 2, "ball top,    in-range  → expect move_up      to recenter ball"),
        (encode(2, 1, IN_RANGE), 3, "ball bottom, in-range  → expect move_down    to recenter ball"),
        # Distance check — ball is centered but too far → drone should move forward
        (encode(1, 1, FAR),      4, "ball center, far       → expect move_forward to close distance"),
    ]
    all_passed = True
    for state, expected_action, desc in checks:
        learned = int(np.argmax(Q[state]))
        passed  = learned == expected_action
        status  = "PASS" if passed else "WARN"
        if not passed:
            all_passed = False
        print(
            f"  [{status}]  {desc}  "
            f"→ learned: {ACTION_NAMES[learned]}"
            + ("" if passed else f"  (expected {ACTION_NAMES[expected_action]})")
        )
    if all_passed:
        print("\n  All directional checks passed.")
    else:
        print(
            "\n  Some checks failed. Before assuming the policy is wrong, verify "
            "that ACTION_DELTAS in sim_env_withdepth.py matches your drone's real "
            "behavior. A WARN here most likely means a sign flip in ACTION_DELTAS, "
            "not a learning failure."
        )


def _smooth(values, window):
    """Simple rolling average for plotting."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def _plot_training_curves(ep_rewards, ep_lengths):
    """
    Two simple plots:
      Top:    episode reward + running average — shows if the agent is
              learning to keep the ball in frame and accumulate positive reward.
      Bottom: episode length + running average — shows how long the agent
              keeps the ball in frame before losing it each episode.
    """
    episodes = np.arange(1, len(ep_rewards) + 1)

    # Running (cumulative) average up to each episode
    running_avg_reward = np.cumsum(ep_rewards) / episodes
    running_avg_length = np.cumsum(ep_lengths) / episodes

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.suptitle("Sim Pre-Training Metrics (with depth)", fontsize=13)

    # --- Episode reward ------------------------------------------------------
    ax1.plot(episodes, ep_rewards,
             color="steelblue", alpha=0.25, linewidth=0.6, label="per episode")
    ax1.plot(episodes, running_avg_reward,
             color="steelblue", linewidth=2.0, label="running average")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Reward")
    ax1.set_title("Episode Reward")
    ax1.legend(fontsize=8)

    # --- Episode length ------------------------------------------------------
    ax2.plot(episodes, ep_lengths,
             color="coral", alpha=0.25, linewidth=0.6, label="per episode")
    ax2.plot(episodes, running_avg_length,
             color="coral", linewidth=2.0, label="running average")
    ax2.set_ylabel("Steps")
    ax2.set_xlabel("Episode")
    ax2.set_title("Episode Length")
    ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()


if __name__ == "__main__":
    Q = train()