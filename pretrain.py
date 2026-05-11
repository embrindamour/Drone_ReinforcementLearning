# =============================================================================
# pretrain.py
# Trains Q-learning and SARSA side by side on BallTrackingEnv.
# 5-action space: move_left, move_right, move_up, move_down, hover.
#
# Hover allows the agent to hold position when the ball is centered.
# Without hover the agent was forced to pick a directional action from
# center, actively shifting the ball away each step. With hover, the
# expected sanity check at s=4 (center) is action=hover.
#
# LOST=terminal (cliff analogy). CENTER=non-terminal (tracking task).
# =============================================================================

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

NUM_EPISODES     = 8000
MAX_STEPS_PER_EP = 50

ALPHA            = 0.1
GAMMA            = 0.95

EPSILON_START    = 1.0
EPSILON_END      = 0.01
EPSILON_DECAY = 0.9985

LOG_INTERVAL     = 1000

SAVE_PATH_QL     = "q_table_qlearning.npy"
SAVE_PATH_SARSA  = "q_table_sarsa.npy"

SEED             = 42

EVAL_EPISODES    = 500
SMOOTH_WINDOW    = 200

# =============================================================================
# END HYPERPARAMETERS
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sim_env import (
    BallTrackingEnv,
    NUM_STATES, NUM_ACTIONS,
    ACTION_NAMES, ACTION_DELTAS,
    LOST, _decode, _encode,
    MAX_STEPS,
    _CELL_TYPE,
    CENTER, ADJACENT, CORNER,
)

# Drone movement direction in plot coords for arrow drawing.
# Cell (r,c) → plot position (c, 2-r). y increases upward.
# Hover gets no arrow — drawn as a circle instead.
_ARROW_DRONE = {
    0: (-0.28,  0.00),   # move_left
    1: (+0.28,  0.00),   # move_right
    2: ( 0.00, +0.28),   # move_up
    3: ( 0.00, -0.28),   # move_down
    4: None,             # hover — no arrow
}

_CELL_COLORS = {
    CENTER:   '#b6e8b6',
    ADJACENT: '#fff3b0',
    CORNER:   '#ffd6b0',
}


# =============================================================================
# Helpers
# =============================================================================

def _epsilon_greedy(state, Q, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, NUM_ACTIONS))
    return int(np.argmax(Q[state]))


def _rolling_avg(values, window):
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return np.cumsum(arr) / np.arange(1, len(arr) + 1)
    return np.convolve(arr, np.ones(window) / window, mode='valid')


# =============================================================================
# Training
# =============================================================================

def train_agent(algorithm):
    """
    Train using 'qlearning' or 'sarsa'.
    LOST is terminal: bootstrap target = reward only when next_state==LOST.
    Returns (Q, metrics).
    """
    assert algorithm in ('qlearning', 'sarsa')

    rng = np.random.default_rng(SEED)
    env = BallTrackingEnv(seed=SEED)
    Q   = np.zeros((NUM_STATES, NUM_ACTIONS))

    epsilon = EPSILON_START

    ep_rewards = []
    ep_lengths = []
    ep_lost    = []

    label = algorithm.upper()
    print(f"\n{'='*62}")
    print(f"  TRAINING: {label}")
    print(f"{'='*62}")
    print(f"  Episodes={NUM_EPISODES}  α={ALPHA}  γ={GAMMA}  "
          f"ε: {EPSILON_START}→{EPSILON_END} (decay {EPSILON_DECAY})")
    print(f"  MAX_STEPS={MAX_STEPS_PER_EP}  LOST=terminal  "
          f"NUM_ACTIONS={NUM_ACTIONS} (incl. hover)\n")

    for episode in range(1, NUM_EPISODES + 1):

        state        = env.reset()
        total_reward = 0.0
        lost_ep      = 0

        if algorithm == 'sarsa':
            action = _epsilon_greedy(state, Q, epsilon, rng)

        for step in range(MAX_STEPS_PER_EP):

            if algorithm == 'qlearning':
                action = _epsilon_greedy(state, Q, epsilon, rng)

            next_state, reward, done = env.step(action)
            total_reward += reward
            is_terminal   = (next_state == LOST)

            # --- TD update -----------------------------------------------
            if is_terminal:
                td_target = reward
            elif algorithm == 'qlearning':
                td_target = reward + GAMMA * np.max(Q[next_state])
            else:   # sarsa
                next_action = _epsilon_greedy(next_state, Q, epsilon, rng)
                td_target   = reward + GAMMA * Q[next_state, next_action]

            Q[state, action] += ALPHA * (td_target - Q[state, action])
            # -------------------------------------------------------------

            state = next_state

            if is_terminal:
                lost_ep = 1
                break
            if done:
                break
            if algorithm == 'sarsa':
                action = next_action

        ep_rewards.append(total_reward)
        ep_lengths.append(step + 1)
        ep_lost.append(lost_ep)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % LOG_INTERVAL == 0:
            w = slice(-LOG_INTERVAL, None)
            print(
                f"  Ep {episode:>5}/{NUM_EPISODES}  "
                f"ε={epsilon:.4f}  "
                f"avg_reward={np.mean(ep_rewards[w]):>+6.2f}  "
                f"avg_len={np.mean(ep_lengths[w]):>5.1f}  "
                f"lost%={np.mean(ep_lost[w])*100:>5.1f}%"
            )

    return Q, {'rewards': ep_rewards, 'lengths': ep_lengths, 'lost': ep_lost}


# =============================================================================
# Evaluation
# =============================================================================

def run_evaluation(Q, label):
    env = BallTrackingEnv(seed=SEED + 1)

    rewards         = []
    lengths         = []
    lost_eps        = []
    steps_to_center = []
    center_fracs    = []
    hover_fracs     = []   # new: how often agent hovers

    for _ in range(EVAL_EPISODES):
        state        = env.reset()
        total_r      = 0.0
        first_center = None
        center_steps = 0
        hover_steps  = 0

        for step in range(MAX_STEPS_PER_EP):
            action              = int(np.argmax(Q[state]))
            next_state, r, done = env.step(action)
            total_r            += r

            if action == 4:
                hover_steps += 1
            if next_state == 4:
                center_steps += 1
                if first_center is None:
                    first_center = step + 1

            state = next_state
            if done:
                break

        ep_len = step + 1
        rewards.append(total_r)
        lengths.append(ep_len)
        lost_eps.append(1 if state == LOST else 0)
        center_fracs.append(center_steps / ep_len)
        hover_fracs.append(hover_steps   / ep_len)
        if first_center is not None:
            steps_to_center.append(first_center)

    reach_rate = len(steps_to_center) / EVAL_EPISODES * 100

    print(f"\n  --- Greedy Evaluation: {label} ({EVAL_EPISODES} episodes) ---")
    print(f"  avg reward           : {np.mean(rewards):>+7.3f}  (std {np.std(rewards):.3f})")
    print(f"  avg episode length   : {np.mean(lengths):>7.2f}  (std {np.std(lengths):.2f})")
    print(f"  % eps ended in LOST  : {np.mean(lost_eps)*100:>7.1f}%")
    print(f"  % steps in center    : {np.mean(center_fracs)*100:>7.1f}%")
    print(f"  % steps hovering     : {np.mean(hover_fracs)*100:>7.1f}%")
    print(f"  reached center       : {reach_rate:>7.1f}% of episodes")
    if steps_to_center:
        print(f"  steps to 1st center  : "
              f"avg={np.mean(steps_to_center):.2f}  "
              f"min={np.min(steps_to_center)}  "
              f"max={np.max(steps_to_center)}")

    return {
        'avg_reward':       np.mean(rewards),
        'std_reward':       np.std(rewards),
        'avg_length':       np.mean(lengths),
        'lost_pct':         np.mean(lost_eps)    * 100,
        'center_pct':       np.mean(center_fracs)* 100,
        'hover_pct':        np.mean(hover_fracs) * 100,
        'reach_rate':       reach_rate,
        'avg_steps_center': np.mean(steps_to_center) if steps_to_center else float('nan'),
    }


# =============================================================================
# Sanity checks
# =============================================================================

def run_sanity_checks(Q, label):
    """
    Directional checks for non-center cells + hover check at center.
    All 5 should pass on a well-trained agent.
    """
    checks = [
        (3, 0, "ball at (1,0) left    → expect move_left"),
        (5, 1, "ball at (1,2) right   → expect move_right"),
        (1, 2, "ball at (0,1) top     → expect move_up"),
        (7, 3, "ball at (2,1) bottom  → expect move_down"),
        (4, 4, "ball at (1,1) center  → expect hover"),      # new
    ]
    print(f"\n  --- Policy Sanity Checks: {label} ---")
    all_pass = True
    for state, expected, desc in checks:
        learned = int(np.argmax(Q[state]))
        ok      = learned == expected
        if not ok:
            all_pass = False
        tag    = "PASS" if ok else "WARN"
        suffix = "" if ok else f"  (expected {ACTION_NAMES[expected]})"
        print(f"  [{tag}]  {desc}  →  {ACTION_NAMES[learned]}{suffix}")
    if all_pass:
        print("         All checks passed.")
    else:
        print("         Some checks failed — see notes above.")


# =============================================================================
# Comparison table
# =============================================================================

def print_comparison(eval_ql, eval_sarsa, metrics_ql, metrics_sarsa):
    w = slice(-LOG_INTERVAL, None)

    def tm(m, key): return np.mean(m[key][w])

    ql_r, ql_l, ql_lst = tm(metrics_ql,'rewards'), tm(metrics_ql,'lengths'), tm(metrics_ql,'lost')*100
    sa_r, sa_l, sa_lst = tm(metrics_sarsa,'rewards'), tm(metrics_sarsa,'lengths'), tm(metrics_sarsa,'lost')*100

    print(f"\n{'='*64}")
    print(f"  FINAL COMPARISON  (last {LOG_INTERVAL} train eps / {EVAL_EPISODES} eval eps)")
    print(f"{'='*64}")
    print(f"  {'Metric':<34} {'Q-Learning':>12}  {'SARSA':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Train avg reward (last window)':<34} {ql_r:>+12.3f}  {sa_r:>+12.3f}")
    print(f"  {'Train avg episode length':<34} {ql_l:>12.2f}  {sa_l:>12.2f}")
    print(f"  {'Train lost% (last window)':<34} {ql_lst:>11.1f}%  {sa_lst:>11.1f}%")
    print(f"  {'-'*60}")
    print(f"  {'Eval avg reward':<34} {eval_ql['avg_reward']:>+12.3f}  {eval_sarsa['avg_reward']:>+12.3f}")
    print(f"  {'Eval avg episode length':<34} {eval_ql['avg_length']:>12.2f}  {eval_sarsa['avg_length']:>12.2f}")
    print(f"  {'Eval % eps ended LOST':<34} {eval_ql['lost_pct']:>11.1f}%  {eval_sarsa['lost_pct']:>11.1f}%")
    print(f"  {'Eval % steps in center':<34} {eval_ql['center_pct']:>11.1f}%  {eval_sarsa['center_pct']:>11.1f}%")
    print(f"  {'Eval % steps hovering':<34} {eval_ql['hover_pct']:>11.1f}%  {eval_sarsa['hover_pct']:>11.1f}%")
    print(f"  {'Eval reached center rate':<34} {eval_ql['reach_rate']:>11.1f}%  {eval_sarsa['reach_rate']:>11.1f}%")
    print(f"  {'Eval avg steps to 1st center':<34} {eval_ql['avg_steps_center']:>12.2f}  {eval_sarsa['avg_steps_center']:>12.2f}")
    print(f"{'='*64}")

    print(f"\n  Interpretation:")
    center_delta = eval_sarsa['center_pct'] - eval_ql['center_pct']
    lost_delta   = eval_ql['lost_pct']      - eval_sarsa['lost_pct']
    hover_ql     = eval_ql['hover_pct']
    hover_sa     = eval_sarsa['hover_pct']

    if hover_ql > 5 or hover_sa > 5:
        better = "Q-Learning" if hover_ql > hover_sa else "SARSA"
        print(f"  Hover is being used: QL={hover_ql:.1f}%  SARSA={hover_sa:.1f}%")
        print(f"  {better} hovers more — holds center position more aggressively.")
    if center_delta > 0:
        print(f"  SARSA spends {center_delta:.1f}pp more time in center.")
    elif center_delta < 0:
        print(f"  Q-Learning spends {-center_delta:.1f}pp more time in center.")
    if lost_delta > 0.5:
        print(f"  SARSA ends {lost_delta:.1f}pp fewer episodes in LOST — safer policy.")
    elif lost_delta < -0.5:
        print(f"  Q-Learning ends {-lost_delta:.1f}pp fewer episodes in LOST.")
    else:
        print(f"  LOST% difference is <0.5pp — algorithms converged similarly.")


# =============================================================================
# Q-table printer
# =============================================================================

def print_qtable(Q, label):
    print(f"\n  --- Q-Table: {label} ---")
    header = f"  {'State':<22} {'Best Action':<14} Q-values [L, R, U, D, HOV]"
    print(header)
    print(f"  {'-'*70}")
    for s in range(NUM_STATES):
        if s == LOST:
            state_label = "LOST (9)  [terminal]"
        else:
            r, c = _decode(s)
            tag  = "CENTER" if s == 4 else _CELL_TYPE[(r,c)]
            state_label = f"({r},{c}) s={s} {tag}"
        best_a = int(np.argmax(Q[s]))
        qvals  = "  ".join(f"{v:>+6.3f}" for v in Q[s])
        print(f"  {state_label:<22} {ACTION_NAMES[best_a]:<14} [{qvals}]")


# =============================================================================
# Training curves
# =============================================================================

def plot_training_curves(metrics_ql, metrics_sarsa):
    episodes = np.arange(1, NUM_EPISODES + 1)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Training Curves: Q-Learning vs SARSA  (hover action included)",
                 fontsize=12, fontweight='bold')

    data_rows = [
        (metrics_ql['rewards'],  metrics_sarsa['rewards'],  'Episode Reward',        'steelblue'),
        (metrics_ql['lengths'],  metrics_sarsa['lengths'],  'Episode Length (steps)', 'seagreen'),
        (metrics_ql['lost'],     metrics_sarsa['lost'],     'Ended in LOST (0/1)',    'coral'),
    ]

    for row, (ql_d, sa_d, ylabel, color) in enumerate(data_rows):
        for col, (data, title) in enumerate([(ql_d, 'Q-Learning'), (sa_d, 'SARSA')]):
            ax       = axes[row, col]
            arr      = np.array(data, dtype=float)
            smoothed = _rolling_avg(arr, SMOOTH_WINDOW)
            offset   = len(arr) - len(smoothed)
            ax.plot(episodes,         arr,      color=color, alpha=0.15, linewidth=0.5)
            ax.plot(episodes[offset:], smoothed, color=color, linewidth=2.0,
                    label=f'{SMOOTH_WINDOW}-ep avg')
            ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight='bold')
            if row == 2:
                ax.set_xlabel("Episode", fontsize=9)
                ax.set_ylim(-0.05, 1.05)
            if row == 0:
                ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    print("\n  Saved: training_curves.png")


# =============================================================================
# Policy grid + visitation heatmap
# =============================================================================

def plot_policy_grid(Q_ql, Q_sarsa):
    """
    3x3 grid per algorithm.
    Cell brightness = greedy-policy visitation frequency.
    Arrow = drone movement direction. Hover = filled circle (no arrow).
    """

    def _get_visitation(Q):
        env   = BallTrackingEnv(seed=SEED + 99)
        visit = np.zeros(9, dtype=float)
        for _ in range(EVAL_EPISODES):
            state = env.reset()
            for _ in range(MAX_STEPS_PER_EP):
                if state != LOST:
                    visit[state] += 1
                action              = int(np.argmax(Q[state]))
                next_state, _, done = env.step(action)
                state               = next_state
                if done:
                    break
        total = visit.sum()
        return visit / total if total > 0 else visit

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        "Learned Policy Grid\n"
        "Arrow = drone movement direction  |  ● = hover  |  "
        "Brightness = visitation frequency",
        fontsize=11, fontweight='bold'
    )

    for ax, Q, title, visit in [
        (axes[0], Q_ql,    'Q-Learning', _get_visitation(Q_ql)),
        (axes[1], Q_sarsa, 'SARSA',      _get_visitation(Q_sarsa)),
    ]:
        ax.set_xlim(-0.1, 3.1)
        ax.set_ylim(-0.3, 3.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

        max_v = visit.max() if visit.max() > 0 else 1.0

        for (r, c), ctype in _CELL_TYPE.items():
            s      = _encode(r, c)
            px     = c           # plot x = col
            py     = 2 - r       # plot y: row 0 at top

            # Cell fill: blend base color with white by visitation
            base   = np.array(plt.matplotlib.colors.to_rgb(_CELL_COLORS[ctype]))
            white  = np.array([1.0, 1.0, 1.0])
            blend  = 0.20 + 0.80 * (visit[s] / max_v)
            fcolor = blend * base + (1 - blend) * white

            ax.add_patch(plt.Rectangle(
                (px, py), 1, 1,
                linewidth=2, edgecolor='#333', facecolor=fcolor
            ))

            # State index and visitation %
            ax.text(px + 0.5, py + 0.88, f"s={s}",
                    ha='center', va='top', fontsize=7.5, color='#444')
            ax.text(px + 0.5, py + 0.10, f"{visit[s]*100:.1f}%",
                    ha='center', va='bottom', fontsize=7.5, color='#333')

            cx, cy = px + 0.5, py + 0.5

            # Center target marker
            if (r, c) == (1, 1):
                ax.plot(cx, cy, '+', markersize=20,
                        color='#1a6e1a', markeredgewidth=2.5, zorder=6)
                ax.text(cx, cy - 0.23, 'TARGET',
                        ha='center', va='top', fontsize=7,
                        color='#1a6e1a', fontweight='bold')

            # Greedy action
            best_a  = int(np.argmax(Q[s]))
            arrow_d = _ARROW_DRONE[best_a]

            if arrow_d is None:
                # Hover — draw filled circle
                ax.plot(cx, cy, 'o', markersize=13,
                        color='#1a1a8c', zorder=7, alpha=0.85)
                ax.text(cx, cy, 'HOV',
                        ha='center', va='center', fontsize=6,
                        color='white', fontweight='bold', zorder=8)
            else:
                dx, dy = arrow_d
                ax.annotate(
                    '',
                    xy=(cx + dx, cy + dy),
                    xytext=(cx - dx * 0.4, cy - dy * 0.4),
                    arrowprops=dict(arrowstyle='->', color='#1a1a8c',
                                   lw=2.0, mutation_scale=16),
                    zorder=7
                )
                ax.text(cx + dx * 1.12, cy + dy * 1.12,
                        ACTION_NAMES[best_a].replace('move_', ''),
                        ha='center', va='center',
                        fontsize=6.5, color='#1a1a8c',
                        fontweight='bold', zorder=8)

        # Row / col axis labels
        for i in range(3):
            ax.text(-0.05, i + 0.5, f"r={2-i}",
                    ha='right', va='center', fontsize=8, color='gray')
            ax.text(i + 0.5, -0.08, f"c={i}",
                    ha='center', va='top', fontsize=8, color='gray')

        # Legend
        legend_elems = [
            mpatches.Patch(facecolor=_CELL_COLORS[CENTER],   label='Center  +1.0'),
            mpatches.Patch(facecolor=_CELL_COLORS[ADJACENT],  label='Edge    +0.3'),
            mpatches.Patch(facecolor=_CELL_COLORS[CORNER],    label='Corner  −0.1'),
        ]
        ax.legend(handles=legend_elems, loc='lower center',
                  bbox_to_anchor=(0.5, -0.12), ncol=3,
                  fontsize=8, framealpha=0.85)

    plt.tight_layout()
    plt.savefig("policy_grid.png", dpi=150, bbox_inches='tight')
    print("  Saved: policy_grid.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    Q_ql,    metrics_ql    = train_agent('qlearning')
    Q_sarsa, metrics_sarsa = train_agent('sarsa')

    np.save(SAVE_PATH_QL,    Q_ql)
    np.save(SAVE_PATH_SARSA, Q_sarsa)
    print(f"\n  Q-tables saved: {SAVE_PATH_QL}, {SAVE_PATH_SARSA}")

    print_qtable(Q_ql,    "Q-Learning")
    print_qtable(Q_sarsa, "SARSA")

    run_sanity_checks(Q_ql,    "Q-Learning")
    run_sanity_checks(Q_sarsa, "SARSA")

    print(f"\n{'='*64}")
    print(f"  GREEDY EVALUATION")
    print(f"{'='*64}")
    eval_ql    = run_evaluation(Q_ql,    "Q-Learning")
    eval_sarsa = run_evaluation(Q_sarsa, "SARSA")

    print_comparison(eval_ql, eval_sarsa, metrics_ql, metrics_sarsa)

    print(f"\n  Generating plots...")
    plot_training_curves(metrics_ql, metrics_sarsa)
    plot_policy_grid(Q_ql, Q_sarsa)
    plt.show()