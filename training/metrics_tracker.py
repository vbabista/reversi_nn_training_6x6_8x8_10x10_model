"""
training/metrics_tracker.py  –  Sledování metrik tréninku a logování.

Sbírá statistiky per-epizoda a zapisuje je do CSV pro analýzu.
Poskytuje rolling window průměry pro výpis do konzole.

Metriky per-epizoda:
    - výsledek (výhra/remíza/prohra)
    - finální skóre (AI disky vs. oponent disky)
    - délka hry (počet tahů)
    - průměrná TD loss v epizodě
    - počet force-pass donucení soupeře
    - rohová výhoda v průběhu hry
    - průměrná mobilita
    - fáze oponenta (random/greedy/minimax_dN)

CSV schéma:
    episode, size, phase, opponent, result, ai_discs, opp_discs,
    n_moves, loss, corners_avg, mob_avg, forced_passes, win_rate_100,
    lr, epsilon
"""

import os
import csv
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any


class MetricsTracker:

    def __init__(self, board_size: int, log_dir: str = 'logs', window: int = 200):
        self.board_size = board_size
        self.log_dir    = log_dir
        self.window     = window
        self.episode    = 0

        os.makedirs(log_dir, exist_ok=True)

        self._csv_path = os.path.join(log_dir, f'training_{board_size}x{board_size}.csv')
        self._csv_file = open(self._csv_path, 'w', newline='', buffering=1)
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'episode', 'size', 'curriculum_phase', 'opponent',
            'result', 'ai_discs', 'opp_discs', 'n_ai_moves',
            'td_loss', 'corners_avg', 'mob_avg', 'forced_passes',
            'win_rate_100', 'loss_rate_100', 'lr', 'epsilon',
            'improvement_bonus',
        ])

        # Rolling windows
        self._results      = deque(maxlen=window)   # 1=win, 0.5=draw, 0=loss
        self._losses       = deque(maxlen=window)   # TD loss per episode
        self._disc_margins = deque(maxlen=window)   # AI - OPP discs
        self._game_lengths = deque(maxlen=window)
        self._corners      = deque(maxlen=window)
        self._mobs         = deque(maxlen=window)
        self._forced       = deque(maxlen=window)

        # Celkové počítadla
        self._total_wins   = 0
        self._total_draws  = 0
        self._total_losses = 0

    def record_episode(self,
                        result:      float,        # +1/0/-1 z game_result()
                        ai_discs:    int,
                        opp_discs:   int,
                        n_ai_moves:  int,
                        td_loss:     float,
                        corners_avg: float,
                        mob_avg:     float,
                        forced_passes: int,
                        curriculum_phase: str,
                        opponent_name:    str,
                        lr:          float,
                        epsilon:     float,
                        improvement_bonus: float = 0.0):

        self.episode += 1

        # Win/draw/loss
        if   result > 0:   r_scalar = 1.0;  self._total_wins   += 1
        elif result == 0:  r_scalar = 0.5;  self._total_draws  += 1
        else:              r_scalar = 0.0;  self._total_losses += 1

        self._results.append(r_scalar)
        self._losses.append(td_loss)
        self._disc_margins.append(ai_discs - opp_discs)
        self._game_lengths.append(n_ai_moves)
        self._corners.append(corners_avg)
        self._mobs.append(mob_avg)
        self._forced.append(forced_passes)

        # Rolling win/loss rates
        wr = float(np.mean([r == 1.0 for r in self._results]))
        lr_rate = float(np.mean([r == 0.0 for r in self._results]))

        # Zapsat do CSV
        self._csv_writer.writerow([
            self.episode, self.board_size, curriculum_phase, opponent_name,
            f'{result:.0f}', ai_discs, opp_discs, n_ai_moves,
            f'{td_loss:.6f}', f'{corners_avg:.4f}', f'{mob_avg:.4f}',
            forced_passes, f'{wr:.4f}', f'{lr_rate:.4f}',
            f'{lr:.2e}', f'{epsilon:.4f}',
            f'{improvement_bonus:.4f}',
        ])

    def console_summary(self, episode_total: int) -> str:

        if not self._results:
            return f'Ep {self.episode}/{episode_total} | Ještě žádná data'

        wr  = float(np.mean([r == 1.0 for r in self._results])) * 100
        dr  = float(np.mean([r == 0.5 for r in self._results])) * 100
        avg_loss  = float(np.mean(self._losses)) if self._losses else 0.0
        avg_margin = float(np.mean(self._disc_margins)) if self._disc_margins else 0.0
        avg_corners = float(np.mean(self._corners)) if self._corners else 0.0
        avg_forced  = float(np.mean(self._forced))  if self._forced  else 0.0

        pct = 100.0 * self.episode / max(1, episode_total)

        return (
            f'[{pct:5.1f}%] Ep {self.episode:>6}/{episode_total} | '
            f'W:{wr:5.1f}% D:{dr:4.1f}% | '
            f'Loss:{avg_loss:.4f} | '
            f'Margin:{avg_margin:+.1f} | '
            f'Corners:{avg_corners:+.2f} | '
            f'Force:{avg_forced:.1f}'
        )

    def phase_summary(self) -> str:

        if not self._results:
            return '(žádná data)'
        wr = float(np.mean([r == 1.0 for r in self._results])) * 100
        avg_margin = float(np.mean(self._disc_margins))
        return f'Win: {wr:.1f}%, Margin: {avg_margin:+.1f} disků'

    @property
    def rolling_win_rate(self) -> float:

        if not self._results:
            return 0.0
        return float(np.mean([r == 1.0 for r in self._results]))

    @property
    def rolling_avg_loss(self) -> float:
        if not self._losses:
            return 0.0
        return float(np.mean(self._losses))

    @property
    def total_summary(self) -> Dict[str, Any]:
        total = self._total_wins + self._total_draws + self._total_losses
        return {
            'episodes': self.episode,
            'wins':     self._total_wins,
            'draws':    self._total_draws,
            'losses':   self._total_losses,
            'win_rate': self._total_wins / max(1, total),
        }

    def close(self):
        self._csv_file.close()

    def __repr__(self) -> str:
        return (f'MetricsTracker(size={self.board_size}×{self.board_size}, '
                f'ep={self.episode}, csv={self._csv_path})')
