"""
training/curriculum.py  –  Adaptivní curriculum manager (v2, 20 fází).

Podporuje všechny typy oponentů:
  random, greedy, minimax_py, minimax_c, self, mixed, elite
"""

import random
import numpy as np
from collections import deque
from typing import Optional, Tuple, Any

from config import CURRICULUM, CURRICULUM_PROMOTE_THRESHOLD, CURRICULUM_EVAL_WINDOW
from opponents.random_opponent   import RandomOpponent
from opponents.greedy_opponent   import GreedyOpponent
from opponents.minimax_opponent  import MinimaxOpponent
from opponents.c_minimax_bridge  import CMinimax, C_ENGINE_AVAILABLE


class CurriculumManager:


    def __init__(self, board_size: int, total_eps: int, adaptive: bool = True):
        self.board_size  = board_size
        self.total_eps   = total_eps
        self.adaptive    = adaptive

        self._phase_idx         = 0
        self._episodes_in_phase = 0
        self._total_episodes    = 0
        self._results           = deque(maxlen=CURRICULUM_EVAL_WINDOW)

        # Scheduled episode counts per phase
        self._phase_eps = [int(total_eps * p[1]) for p in CURRICULUM]
        diff = total_eps - sum(self._phase_eps)
        self._phase_eps[-1] += diff

        # Cache oponentů (nevytváříme znovu)
        self._cache = {}

        if not C_ENGINE_AVAILABLE:
            print('  [Curriculum] VAROVÁNÍ: C engine nedostupný, minimax_c fáze '
                  'použijí Python fallback.')

        print(f'  [Curriculum] {board_size}×{board_size}: {len(CURRICULUM)} fází '
              f'({"adaptive" if adaptive else "scheduled"})')
        _dscale = {6: 1.25, 8: 1.00, 10: 0.70}.get(board_size, 1.0)
        for i, (name, frac, otype, params) in enumerate(CURRICULUM):
            n = self._phase_eps[i]
            engine_info = ''
            if otype == 'minimax_c':
                if 'target_depth' in params:
                    ref_d  = params['target_depth']
                    actual = max(2, round(ref_d * _dscale))
                    c_str  = 'C' if C_ENGINE_AVAILABLE else 'Py-fallback'
                    engine_info = f' [{c_str}, d={actual} (ref {ref_d})]'
                else:
                    t     = params.get('time_limit_ms', 1000)
                    c_str = 'C' if C_ENGINE_AVAILABLE else 'Py-fallback'
                    engine_info = f' [{c_str}, t={t}ms]'
            elif otype == 'minimax_py':
                engine_info = f' [Python, d={params.get("depth",3)}]'
            print(f'    {i+1:2d}. {name:<20} {n:>5} ep  ({frac*100:.0f}%){engine_info}')


    _DEPTH_SCALE = {6: 1.25, 8: 1.00, 10: 0.70}

    def _scale_depth(self, ref_depth: int) -> int:
        """Vrátí hloubku škálovanou dle velikosti desky, min 2."""
        scale = self._DEPTH_SCALE.get(self.board_size, 1.0)
        return max(2, round(ref_depth * scale))

    def _build(self, phase_type: str, params: dict) -> Any:
        key = f'{phase_type}_{sorted(params.items())}'
        if key in self._cache:
            return self._cache[key]

        if phase_type == 'random':
            opp = RandomOpponent(self.board_size)
        elif phase_type == 'greedy':
            opp = GreedyOpponent(self.board_size)
        elif phase_type == 'minimax_py':
            opp = MinimaxOpponent(params.get('depth', 3), self.board_size)
        elif phase_type == 'minimax_c':
            # FIX: Použij target_depth (depth-only mode) místo time_limit_ms
            if 'target_depth' in params:
                td = self._scale_depth(params['target_depth'])
                opp = CMinimax(target_depth=td, board_size=self.board_size)
            else:
                # Zpětná kompatibilita: pokud někdo ještě posílá time_limit_ms
                tlimit = params.get('time_limit_ms', 1000)
                opp = CMinimax(max_depth=18, time_limit_ms=tlimit,
                               board_size=self.board_size)
        elif phase_type in ('self', 'mixed', 'elite'):
            opp = None
        else:
            raise ValueError(f'Neznámý typ: {phase_type}')

        self._cache[key] = opp
        return opp

    def get_opponent(self, nn=None) -> Tuple[Any, str, str]:

        name, _, phase_type, params = CURRICULUM[self._phase_idx]

        if phase_type == 'self':
            return None, name, 'self-play'

        elif phase_type == 'mixed':
            if random.random() < params.get('self_ratio', 0.4):
                return None, name, 'self-play'
            # FIX: target_depth místo time_limit_ms
            if 'target_depth' in params:
                td = self._scale_depth(params['target_depth'])
                opp = CMinimax(target_depth=td, board_size=self.board_size)
            else:
                tlimit = params.get('time_limit_ms', 1000)
                opp = CMinimax(18, tlimit, self.board_size)
            return opp, name, repr(opp)

        elif phase_type == 'elite':
            # All-round mix: všechny typy oponentů náhodně
            r = random.random()
            self_r = params.get('self_ratio', 0.4)
            py_r   = params.get('py_ratio', 0.10)

            if r < self_r:
                return None, name, 'self-play'
            elif r < self_r + py_r:
                depths = params.get('py_depths', [1, 2, 3, 4])
                depth  = random.choice(depths)
                opp    = MinimaxOpponent(depth, self.board_size)
                return opp, name, repr(opp)
            else:
                # FIX: Nový klíč target_depths místo time_limits
                if 'target_depths' in params:
                    ref_td = random.choice(params['target_depths'])
                    td     = self._scale_depth(ref_td)
                    opp    = CMinimax(target_depth=td, board_size=self.board_size)
                else:
                    # Zpětná kompatibilita
                    tlimits = params.get('time_limits', [1000])
                    tlimit  = random.choice(tlimits)
                    opp     = CMinimax(18, tlimit, self.board_size)
                return opp, name, repr(opp)

        else:
            opp = self._build(phase_type, params)
            return opp, name, repr(opp) if opp else 'None'

    def record_result(self, result: float):
        self._total_episodes    += 1
        self._episodes_in_phase += 1
        win = 1.0 if result > 0 else (0.5 if abs(result - 0.08) < 0.01 else 0.0)
        self._results.append(win)

        if self.adaptive:
            self._maybe_promote()
        else:
            self._scheduled_advance()

    def _maybe_promote(self):
        min_eps = max(CURRICULUM_EVAL_WINDOW, 200)
        if self._episodes_in_phase < min_eps:
            return

        wr = float(np.mean(list(self._results)))

        if wr >= CURRICULUM_PROMOTE_THRESHOLD:
            self._advance(f'wr={wr:.1%} > {CURRICULUM_PROMOTE_THRESHOLD:.0%}')
            return

        # Backtrack při velmi nízké výhřivosti
        if (self._episodes_in_phase > 550 and
                len(self._results) == CURRICULUM_EVAL_WINDOW and
                wr < 0.14 and self._phase_idx > 0):
            print(f'  [Curriculum] Backtrack: wr={wr:.1%} po {self._episodes_in_phase} ep')
            self._phase_idx          = max(0, self._phase_idx - 1)
            self._episodes_in_phase  = 0
            self._results.clear()

    def _scheduled_advance(self):
        if self._episodes_in_phase >= self._phase_eps[self._phase_idx]:
            self._advance('scheduled')

    def _advance(self, reason: str = ''):
        if self._phase_idx < len(CURRICULUM) - 1:
            old = CURRICULUM[self._phase_idx][0]
            self._phase_idx        += 1
            self._episodes_in_phase = 0
            self._results.clear()
            new = CURRICULUM[self._phase_idx][0]
            print(f'\n  ══ Curriculum: {old} → {new} [{reason}] ══\n')

    @property
    def phase_name(self) -> str:
        return CURRICULUM[self._phase_idx][0]

    @property
    def phase_index(self) -> int:
        return self._phase_idx

    @property
    def rolling_win_rate(self) -> float:
        if not self._results: return 0.0
        return float(np.mean(list(self._results)))

    @property
    def progress(self) -> float:
        return min(1.0, self._total_episodes / max(1, self.total_eps))

    def __repr__(self) -> str:
        return (f'CurriculumManager(size={self.board_size}, '
                f'phase={self._phase_idx+1}/{len(CURRICULUM)} '
                f'[{self.phase_name}], wr={self.rolling_win_rate:.1%})')
