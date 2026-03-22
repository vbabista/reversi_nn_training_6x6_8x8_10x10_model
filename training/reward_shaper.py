"""
training/reward_shaper.py  –  Hustý delta reward + margin-based final reward.

KLÍČOVÉ NOVINKY v2:
══════════════════════════════════════════════════════════════════════

1. MARGIN-BASED FINAL REWARD
   Místo flat +1/-1 pro výhru/prohru → škála podle diskové marže:
     diff = (my_discs - opp_discs) / total_discs

     Dominance  (>=75%) → +1.00     Dominována (<=25%) → -1.00
     Silná výhra (60%)  → +0.75     Silná prohra (40%) → -0.75
     Normální    (54%)  → +0.50     Normální     (46%) → -0.50
     Těsná výhra (51%)  → +0.25     Těsná prohra (49%) → -0.25
     Remíza      (50%)  → +0.08

   Proč?
   - Gradient existuje i pro různě velké výhry/prohry
   - Model se explicitně učí DOMINOVAT, ne jen vyhrát o 1 disk
   - Detekuje pokrok i bez explicitního výsledku

2. DELTA REWARD (nezměněno z v1)
   r_t = w_corner*Δcorners + w_mob*Δmobility + w_stab*Δstability + force_bonus

3. TD(λ) S MARGIN RESULT (aktualizováno)
   final_target = α*margin_result + (1-α)*G_t^λ
"""

import numpy as np
from typing import Optional, List, Dict

from core.board import get_valid_moves, count_pieces, EMPTY, BLACK, WHITE
from config import (
    REWARD_CORNER_WEIGHT, REWARD_MOBILITY_WEIGHT,
    REWARD_STABILITY_WEIGHT, REWARD_FORCE_PASS,
    MARGIN_THRESHOLDS, IMPROVEMENT_EMA_ALPHA,
)


# ══════════════════════════════════════════════════════════════════════
# MARGIN-BASED GAME RESULT
# ══════════════════════════════════════════════════════════════════════

def margin_game_result(board: np.ndarray, color: int) -> float:

    b, w = count_pieces(board)
    mine = b if color == BLACK else w
    opp  = w if color == BLACK else b
    total = mine + opp
    if total == 0:
        return 0.08

    diff = (mine - opp) / total  # v [-1, 1]

    for threshold, reward, label in MARGIN_THRESHOLDS:
        if diff >= threshold:
            return reward

    return -1.0   # shouldn't reach here


def margin_label(board: np.ndarray, color: int) -> str:

    b, w = count_pieces(board)
    mine = b if color == BLACK else w
    opp  = w if color == BLACK else b
    total = mine + opp
    if total == 0:
        return 'draw'
    diff = (mine - opp) / total
    for threshold, reward, label in MARGIN_THRESHOLDS:
        if diff >= threshold:
            return label
    return 'dominated'


# ══════════════════════════════════════════════════════════════════════
# POZIČNÍ METRIKY (pro delta reward)
# ══════════════════════════════════════════════════════════════════════

def _corners(board: np.ndarray, color: int) -> float:
    n = board.shape[0]
    opp = 1 - color
    crn = [(0,0),(0,n-1),(n-1,0),(n-1,n-1)]
    my  = sum(1 for r,c in crn if board[r,c] == color)
    op  = sum(1 for r,c in crn if board[r,c] == opp)
    total = my + op
    return (my - op) / total if total > 0 else 0.0

def _mobility_ratio(board: np.ndarray, color: int) -> float:
    opp  = 1 - color
    my_m = len(get_valid_moves(board, color))
    op_m = len(get_valid_moves(board, opp))
    total = my_m + op_m
    return (my_m - op_m) / total if total > 0 else 0.0

def _stability_score(board: np.ndarray, color: int) -> float:
    n   = board.shape[0]
    opp = 1 - color
    crns  = [(0,0),(0,n-1),(n-1,0),(n-1,n-1)]
    edges = ([(0,c) for c in range(1,n-1)] + [(n-1,c) for c in range(1,n-1)] +
             [(r,0) for r in range(1,n-1)] + [(r,n-1) for r in range(1,n-1)])
    my  = 3*sum(1 for r,c in crns if board[r,c]==color) + sum(1 for r,c in edges if board[r,c]==color)
    op  = 3*sum(1 for r,c in crns if board[r,c]==opp)   + sum(1 for r,c in edges if board[r,c]==opp)
    total = my + op
    return (my - op) / total if total > 0 else 0.0

def _disc_ratio(board: np.ndarray, color: int) -> float:
    opp  = 1 - color
    my_d = int((board == color).sum())
    op_d = int((board == opp).sum())
    total = my_d + op_d
    return (my_d - op_d) / total if total > 0 else 0.0

def _phase(board: np.ndarray) -> float:
    return float((board != EMPTY).sum()) / board.size


# ══════════════════════════════════════════════════════════════════════
# REWARD SHAPER
# ══════════════════════════════════════════════════════════════════════

class RewardShaper:

    def __init__(self):
        self._prev_corner = 0.0
        self._prev_mob    = 0.0
        self._prev_stab   = 0.0
        self._prev_disc   = 0.0

        # EMA pro improvement bonus
        self._ema_win    = 0.0
        self._ema_corner = 0.0
        self._ema_mob    = 0.0
        self._ema_stab   = 0.0
        self._ema_force  = 0.0
        self._ema_margin = 0.0   # NOVÉ: EMA margin skóre
        self._games_played = 0

        # Per-game tracking
        self._cur_corners: List[float] = []
        self._cur_mobs:    List[float] = []
        self._cur_stabs:   List[float] = []
        self._cur_forces:  int = 0

    def reset(self, initial_board: Optional[np.ndarray] = None,
              color: Optional[int] = None):
        if initial_board is not None and color is not None:
            self._prev_corner = _corners(initial_board, color)
            self._prev_mob    = _mobility_ratio(initial_board, color)
            self._prev_stab   = _stability_score(initial_board, color)
            self._prev_disc   = _disc_ratio(initial_board, color)
        else:
            self._prev_corner = 0.0
            self._prev_mob    = 0.0
            self._prev_stab   = 0.0
            self._prev_disc   = 0.0

        self._cur_corners = []
        self._cur_mobs    = []
        self._cur_stabs   = []
        self._cur_forces  = 0

    def step_reward(self, board_before: np.ndarray, board_after: np.ndarray,
                    color: int) -> float:

        phase = _phase(board_after)

        cur_corner = _corners(board_after, color)
        cur_mob    = _mobility_ratio(board_after, color)
        cur_stab   = _stability_score(board_after, color)
        cur_disc   = _disc_ratio(board_after, color)

        d_corner = cur_corner - self._prev_corner
        d_mob    = cur_mob    - self._prev_mob
        d_stab   = cur_stab   - self._prev_stab
        d_disc   = cur_disc   - self._prev_disc

        # Force-pass bonus
        opp_moves  = len(get_valid_moves(board_after, 1 - color))
        force_bonus = 0.0
        if   opp_moves == 0: force_bonus = REWARD_FORCE_PASS;  self._cur_forces += 1
        elif opp_moves == 1: force_bonus = 0.20
        elif opp_moves <= 2: force_bonus = 0.08

        # Fázové váhy disku
        if   phase < 0.30: w_disc = 0.02;  w_mob = 0.35
        elif phase < 0.65: w_disc = 0.08;  w_mob = 0.20
        else:              w_disc = 0.25;  w_mob = 0.10

        raw = (REWARD_CORNER_WEIGHT    * d_corner +
               w_mob                   * d_mob    +
               REWARD_STABILITY_WEIGHT * d_stab   +
               w_disc                  * d_disc   +
               force_bonus)

        reward = float(np.tanh(raw * 2.5))

        self._prev_corner = cur_corner
        self._prev_mob    = cur_mob
        self._prev_stab   = cur_stab
        self._prev_disc   = cur_disc

        self._cur_corners.append(cur_corner)
        self._cur_mobs.append(cur_mob)
        self._cur_stabs.append(cur_stab)

        return reward

    def compute_td_targets(self,
                            step_rewards:    List[float],
                            states:          List[np.ndarray],
                            nn_target,
                            final_result:    float,   # margin-based result
                            global_progress: float,
                            gamma:           float = 0.97,
                            td_lambda:       float = 0.80,
                            ) -> List[float]:

        T = len(step_rewards)
        if T == 0:
            return []

        v_next = np.zeros(T + 1, dtype=np.float32)
        for i, s in enumerate(states):
            x = s.reshape(1, -1)
            v_next[i] = float(nn_target.forward(x)[0, 0])

        alpha = 0.20 + 0.65 * global_progress

        targets  = np.zeros(T, dtype=np.float32)
        g_lambda = final_result

        for t in reversed(range(T)):
            r_t  = step_rewards[t]
            v_t1 = float(v_next[t + 1])
            g_lambda = r_t + gamma * ((1.0 - td_lambda) * v_t1 + td_lambda * g_lambda)
            targets[t] = float(np.clip(
                alpha * final_result + (1.0 - alpha) * g_lambda,
                -1.0, 1.0
            ))

        return targets.tolist()

    def game_improvement_bonus(self, board_final: np.ndarray,
                                color: int) -> Dict[str, float]:

        self._games_played += 1

        # Margin skóre
        margin_val = margin_game_result(board_final, color)

        cur_corner = float(np.mean(self._cur_corners)) if self._cur_corners else 0.0
        cur_mob    = float(np.mean(self._cur_mobs))    if self._cur_mobs    else 0.0
        cur_stab   = float(np.mean(self._cur_stabs))   if self._cur_stabs   else 0.0
        cur_force  = float(self._cur_forces)
        cur_win    = 1.0 if margin_val > 0 else (0.5 if margin_val == 0.08 else 0.0)

        bonuses = {}

        if self._games_played < 5:
            for k in ['win_bonus','corner_bonus','mob_bonus','stab_bonus',
                      'force_bonus','margin_bonus','total']:
                bonuses[k] = 0.0
        else:
            def imp(current, ema, scale=1.0):
                delta = current - ema
                return float(np.clip(delta / max(abs(ema)+0.1, 0.1), -1, 1)) * scale

            bonuses['win_bonus']    = imp(cur_win,    self._ema_win,    0.15)
            bonuses['corner_bonus'] = imp(cur_corner, self._ema_corner, 0.12)
            bonuses['mob_bonus']    = imp(cur_mob,    self._ema_mob,    0.08)
            bonuses['stab_bonus']   = imp(cur_stab,   self._ema_stab,   0.08)
            bonuses['force_bonus']  = imp(cur_force,  self._ema_force,  0.05)
            # Bonus za lepší margin než průměr (dominuješ víc než obvykle?)
            bonuses['margin_bonus'] = imp(margin_val, self._ema_margin, 0.12)
            bonuses['total'] = float(np.clip(
                sum(v for k,v in bonuses.items() if k != 'total'),
                -0.30, 0.30
            ))

        α = IMPROVEMENT_EMA_ALPHA
        self._ema_win    = α*cur_win    + (1-α)*self._ema_win
        self._ema_corner = α*cur_corner + (1-α)*self._ema_corner
        self._ema_mob    = α*cur_mob    + (1-α)*self._ema_mob
        self._ema_stab   = α*cur_stab   + (1-α)*self._ema_stab
        self._ema_force  = α*cur_force  + (1-α)*self._ema_force
        self._ema_margin = α*margin_val + (1-α)*self._ema_margin

        return bonuses

    @property
    def ema_stats(self) -> Dict[str, float]:
        return {
            'ema_win':    round(self._ema_win,    3),
            'ema_corner': round(self._ema_corner, 3),
            'ema_mob':    round(self._ema_mob,    3),
            'ema_margin': round(self._ema_margin, 3),
        }
