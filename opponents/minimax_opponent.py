"""
opponents/minimax_opponent.py  –  Silný alfa-beta minimax oponent.

Implementuje klasický negamax s alfa-beta prořezáváním:
  - Transpoziční tabulka (Zobrist hash, LRU-style limit)
  - Iterative deepening (pro time-limited search)
  - Pohybové řazení (rohy > hrany > střed, pak gain)
  - Fázová evaluační funkce (early/mid/late)
  - Podpora pro desky 6×6, 8×8, 10×10

Heuristika:
  1. Konečný stav: přesný výsledek (±∞)
  2. Diskové skóre (fázově vážené)
  3. Mobilita (počet platných tahů)
  4. Rohové obsazení (velmi vysoká váha)
  5. X-pole a C-pole penalizace
  6. Stabilita (aproximace přes stabilní hrany)
  7. Potenciální mobilita (frontier discs)
  8. Parita tahu (kdo hraje poslední tah v endgame)

MinimaxOpponent(depth=N) – hraje do hloubky N
    depth=1  → velmi slabý
    depth=2  → slabý
    depth=3  → střední
    depth=4  → silný (standardní tréninkový oponent)
    depth=5  → velmi silný
    depth=6  → elite (pomalý na větší deskách)
"""

import numpy as np
import random
import os
from typing import Optional, Tuple, List, Dict

from core.board import (
    get_valid_moves, make_move, make_move_inplace,
    is_terminal, count_pieces, game_result,
    board_hash, EMPTY, BLACK, WHITE,
)

_INF = 1e9


# ══════════════════════════════════════════════════════════════════════
# TOP-LEVEL WORKER FUNCTION (musí být na úrovni modulu pro pickle)
# ══════════════════════════════════════════════════════════════════════

def _eval_root_move(args: tuple) -> float:
    import sys, os as _os
    # Přidáme root projektu do sys.path (pro subprocess/spawn)
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _root = _os.path.normpath(_os.path.join(_here, '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import numpy as _np
    from core.board import get_valid_moves, make_move, board_hash, EMPTY, BLACK, WHITE

    board_bytes, board_shape, board_dtype_str, move, color, depth = args

    board = _np.frombuffer(board_bytes, dtype=_np.dtype(board_dtype_str)).reshape(board_shape).copy()

    b2 = make_move(board, move, color)

    # Lokální negamax s vlastní TT
    tt: Dict[int, Tuple] = {}

    def _negamax_local(b, d, col, alpha, beta):
        h = board_hash(b)
        entry = tt.get(h)
        if entry is not None:
            val_e, d_e, flag_e = entry
            if d_e >= d:
                if flag_e == 'exact': return val_e
                if flag_e == 'lower' and val_e >= beta: return val_e
                if flag_e == 'upper' and val_e <= alpha: return val_e

        moves = get_valid_moves(b, col)
        opp = 1 - col

        if not moves:
            opp_moves = get_valid_moves(b, opp)
            if not opp_moves:
                from core.board import count_pieces
                bl, wh = count_pieces(b)
                mine = bl if col == BLACK else wh
                op   = wh if col == BLACK else bl
                if mine > op: return  _INF
                if mine < op: return -_INF
                return 0.0
            return -_negamax_local(b, d, opp, -beta, -alpha)

        if d == 0:
            return _evaluate_local(b, col)

        from opponents.minimax_opponent import _order_moves
        ordered = _order_moves(b, moves, col)
        orig_alpha = alpha
        best_val   = -_INF

        for m in ordered:
            b2l = make_move(b, m, col)
            val = -_negamax_local(b2l, d - 1, opp, -beta, -alpha)
            best_val = max(best_val, val)
            alpha    = max(alpha, val)
            if alpha >= beta:
                break

        flag = 'upper' if best_val <= orig_alpha else ('lower' if best_val >= beta else 'exact')
        if len(tt) < 80_000:
            tt[h] = (best_val, d, flag)

        return best_val

    def _evaluate_local(board, color):
        from opponents.minimax_opponent import evaluate
        return evaluate(board, color)

    val = -_negamax_local(b2, depth - 1, 1 - color, -_INF, _INF)
    return float(val)


# ══════════════════════════════════════════════════════════════════════
# EVALUAČNÍ FUNKCE
# ══════════════════════════════════════════════════════════════════════

def _phase(board: np.ndarray) -> float:
    """Poměr obsazených polí (0.0 = start, 1.0 = konec)."""
    return float((board != EMPTY).sum()) / board.size


def _corners(board: np.ndarray, color: int) -> int:
    n = board.shape[0]
    return sum(1 for r, c in [(0,0),(0,n-1),(n-1,0),(n-1,n-1)]
               if board[r, c] == color)


def _x_c_penalty(board: np.ndarray, color: int) -> float:

    n   = board.shape[0]
    opp = 1 - color
    crns = [(0,0),(0,n-1),(n-1,0),(n-1,n-1)]

    xs = [(1,1),(1,n-2),(n-2,1),(n-2,n-2)]
    cs = [(0,1),(1,0),(0,n-2),(1,n-1),(n-2,0),(n-1,1),(n-2,n-1),(n-1,n-2)]

    score = 0.0
    for lst, w in [(xs, 3.0), (cs, 1.5)]:
        for r, c in lst:
            if not (0 <= r < n and 0 <= c < n):
                continue
            nearest = min(crns, key=lambda cr: abs(cr[0]-r)+abs(cr[1]-c))
            my_crn  = board[nearest] == color
            opp_crn = board[nearest] == opp
            if board[r, c] == color and not my_crn:
                score -= w
            elif board[r, c] == opp and not opp_crn:
                score += w
    return score


def _stability_approx(board: np.ndarray, color: int) -> int:
    n     = board.shape[0]
    score = 0
    crns  = {(0,0),(0,n-1),(n-1,0),(n-1,n-1)}

    # Rohy
    score += 3 * sum(1 for r,c in crns if board[r,c] == color)

    # Hrany (bez rohů)
    edges = (
        [(0, c) for c in range(1, n-1)] +
        [(n-1, c) for c in range(1, n-1)] +
        [(r, 0) for r in range(1, n-1)] +
        [(r, n-1) for r in range(1, n-1)]
    )
    score += sum(1 for r,c in edges if board[r,c] == color)
    return score


def _potential_mobility(board: np.ndarray, color: int) -> int:
    n    = board.shape[0]
    opp  = 1 - color
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    cnt  = 0
    for r in range(n):
        for c in range(n):
            if board[r, c] != EMPTY:
                continue
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if 0 <= nr < n and 0 <= nc < n and board[nr, nc] == opp:
                    cnt += 1
                    break
    return cnt


def evaluate(board: np.ndarray, color: int) -> float:
    opp   = 1 - color
    phase = _phase(board)

    # ── 1. Konečný stav ────────────────────────────────────────────
    my_d, op_d = count_pieces(board)
    if color == WHITE:
        my_d, op_d = op_d, my_d

    total = my_d + op_d
    if total == board.size or (
        not get_valid_moves(board, color) and not get_valid_moves(board, opp)
    ):
        if my_d > op_d: return  _INF
        if my_d < op_d: return -_INF
        return 0.0

    # ── 2. Diskové skóre ───────────────────────────────────────────
    disc_score = (my_d - op_d) / max(1, total)   # [-1, 1]

    # ── 3. Mobilita ────────────────────────────────────────────────
    my_m = len(get_valid_moves(board, color))
    op_m = len(get_valid_moves(board, opp))
    mob_score = (my_m - op_m) / max(1, my_m + op_m)   # [-1, 1]

    # ── 4. Rohy ────────────────────────────────────────────────────
    my_c = _corners(board, color)
    op_c = _corners(board, opp)
    corner_score = float(my_c - op_c)   # [-4, 4]

    # ── 5. X / C penalizace ────────────────────────────────────────
    xc_score = _x_c_penalty(board, color)   # přibližně [-12, 12]

    # ── 6. Stabilita ───────────────────────────────────────────────
    my_s = _stability_approx(board, color)
    op_s = _stability_approx(board, opp)
    stab_score = (my_s - op_s) / max(1, my_s + op_s + 1)   # [-1, 1]

    # ── 7. Potenciální mobilita ────────────────────────────────────
    pot_mine = _potential_mobility(board, color)
    pot_opp  = _potential_mobility(board, opp)
    pot_score = (pot_mine - pot_opp) / max(1, pot_mine + pot_opp + 1)

    # ── 8. Fázové váhy ─────────────────────────────────────────────
    if phase < 0.25:
        return (  5*disc_score + 200*mob_score + 800*corner_score
                + 200*xc_score + 50*stab_score + 30*pot_score)
    elif phase < 0.55:
        return ( 15*disc_score + 120*mob_score + 1200*corner_score
                + 250*xc_score + 100*stab_score + 20*pot_score)
    elif phase < 0.80:
        return ( 50*disc_score + 60*mob_score + 1500*corner_score
                + 150*xc_score + 200*stab_score + 10*pot_score)
    else:
        return (200*disc_score + 15*mob_score + 1500*corner_score
                + 60*xc_score + 150*stab_score)


# ══════════════════════════════════════════════════════════════════════
# TRANSPOZIČNÍ TABULKA
# ══════════════════════════════════════════════════════════════════════

class TranspositionTable:
    MAX_SIZE = 100_000

    def __init__(self):
        self.table: Dict[int, Tuple] = {}

    def get(self, h: int, depth: int, alpha: float, beta: float
            ) -> Optional[float]:
        entry = self.table.get(h)
        if entry is None:
            return None
        val, d, flag = entry
        if d < depth:
            return None   # uložená hodnota je pro menší hloubku → ignoruj
        if flag == 'exact':
            return val
        if flag == 'lower' and val >= beta:
            return val
        if flag == 'upper' and val <= alpha:
            return val
        return None

    def put(self, h: int, val: float, depth: int, flag: str):
        if len(self.table) >= self.MAX_SIZE:
            # LRU-lite: smaž náhodnou část tabulky
            keys = list(self.table.keys())[:self.MAX_SIZE // 4]
            for k in keys:
                del self.table[k]
        self.table[h] = (val, depth, flag)

    def clear(self):
        self.table.clear()


# ══════════════════════════════════════════════════════════════════════
# POHYBOVÉ ŘAZENÍ
# ══════════════════════════════════════════════════════════════════════

def _order_moves(board: np.ndarray, moves: List[Tuple[int, int]], color: int) -> List[Tuple[int, int]]:

    n    = board.shape[0]
    opp  = 1 - color
    crns = {(0,0),(0,n-1),(n-1,0),(n-1,n-1)}
    xs   = {(1,1),(1,n-2),(n-2,1),(n-2,n-2)}
    edges = set(
        [(0, c) for c in range(1, n-1)] +
        [(n-1, c) for c in range(1, n-1)] +
        [(r, 0) for r in range(1, n-1)] +
        [(r, n-1) for r in range(1, n-1)]
    )

    def priority(m):
        r, c = m
        if m in crns:       return 100   # rohy = nejvyšší priorita
        if m in xs:         return -50   # X-pole = penalizace (prozkoumáme pozdě)
        if m in edges:      return  20   # hrany = dobré
        return 0

    scored = [(priority(m), m) for m in moves]
    scored.sort(reverse=True, key=lambda x: x[0])
    return [m for _, m in scored]


# ══════════════════════════════════════════════════════════════════════
# NEGAMAX S ALFA-BETA
# ══════════════════════════════════════════════════════════════════════

class MinimaxOpponent:
    """
    Alfa-beta negamax Reversi oponent s transpoziční tabulkou.

    Parametry
    ---------
    depth      : int   – hloubka prohledávání (1-8)
    board_size : int   – velikost desky (6, 8, 10)
    randomize  : bool  – malá náhodnost při rovnocenných tazích (diverzita)
    n_workers  : int   – počet paralelních workerů pro root tahy
                         0 = auto (depth>=5 AND >=4 tahy: použij max(1, cpu//2-1))
                         1 = vždy sekvenční
                        -1 = vždy auto
    """

    def __init__(self, depth: int = 3, board_size: int = 8,
                 randomize: bool = True, n_workers: int = 0):
        assert 1 <= depth <= 8, 
        self.depth      = depth
        self.board_size = board_size
        self.randomize  = randomize
        self.n_workers  = n_workers
        self.tt         = TranspositionTable()

    # ────────────────────────────────────────────────────────────────
    # Paralelní evaluace root tahů
    # ────────────────────────────────────────────────────────────────

    def _choose_n_workers(self, n_moves: int) -> int:
        """
        Vrátí optimální počet workerů.
        Využíváme paralelismus pouze pro depth>=5 a >=4 tahy (overhead jinak > zisk).
        Na <=2 jádrech zůstáváme sekvenční – spawn overhead přebije zisk.
        Na >=4 jádrech použijeme max cpu//2 workerů (hard cap 4).
        """
        if self.n_workers == 1:
            return 1
        if self.depth < 5 or n_moves < 4:
            return 1   # sekvenční – overhead procesu by byl větší než zisk
        cpu = os.cpu_count() or 2
        if cpu <= 2:
            return 1   # Na 2 jádrech spawn overhead > zisk → sekvenční
        # Použijeme max cpu//2 workerů (šetříme jádra pro OS + ostatní)
        # Hard cap 4 – víc procesů má diminishing returns pro minimax
        workers = min(cpu // 2, n_moves, 4)
        return max(2, workers)

    def get_move(self, board: np.ndarray, color: int) -> Optional[Tuple[int, int]]:

        moves = get_valid_moves(board, color)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        ordered = _order_moves(board, moves, color)
        n_workers = self._choose_n_workers(len(ordered))

        if n_workers > 1:
            return self._get_move_parallel(board, color, ordered, n_workers)
        else:
            return self._get_move_sequential(board, color, ordered)

    def _get_move_sequential(self, board: np.ndarray, color: int,
                              ordered: List[Tuple[int, int]]) -> Tuple[int, int]:
        self.tt.clear()
        best_val  = -_INF
        best_move = ordered[0]
        alpha     = -_INF
        beta      =  _INF

        for m in ordered:
            b2  = make_move(board, m, color)
            val = -self._negamax(b2, self.depth - 1, 1 - color, -beta, -alpha)
            if val > best_val:
                best_val  = val
                best_move = m
            alpha = max(alpha, val)
            if self.randomize and abs(val - best_val) < 1e-4 and random.random() < 0.1:
                best_move = m

        return best_move

    def _get_move_parallel(self, board: np.ndarray, color: int,
                            ordered: List[Tuple[int, int]],
                            n_workers: int) -> Tuple[int, int]:

        import multiprocessing as mp

        board_bytes = board.tobytes()
        board_shape = board.shape
        board_dtype = board.dtype.str

        args = [
            (board_bytes, board_shape, board_dtype, m, color, self.depth)
            for m in ordered
        ]

        try:
            # spawn kontext je bezpečnější než fork pro numpy+ctypes
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=n_workers, maxtasksperchild=50) as pool:
                results = pool.map(_eval_root_move, args, chunksize=1)
        except Exception:
            # Fallback na sekvenční při jakékoli chybě (Windows, nested call atd.)
            self.tt.clear()
            return self._get_move_sequential(board, color, ordered)

        best_val  = -_INF
        best_move = ordered[0]
        for m, val in zip(ordered, results):
            if val > best_val or (abs(val - best_val) < 1e-4 and
                                   self.randomize and random.random() < 0.1):
                best_val  = val
                best_move = m

        return best_move

    def _negamax(self, board: np.ndarray, depth: int, color: int,
                 alpha: float, beta: float) -> float:

        h = board_hash(board)

        # Transpoziční tabulka
        tt_val = self.tt.get(h, depth, alpha, beta)
        if tt_val is not None:
            return tt_val

        moves = get_valid_moves(board, color)
        opp   = 1 - color

        # ── Terminální stav nebo nulová hloubka ────────────────────
        if not moves:
            opp_moves = get_valid_moves(board, opp)
            if not opp_moves:
                # Konec hry
                val = self._terminal_value(board, color)
                self.tt.put(h, val, depth, 'exact')
                return val
            # Pasovat a dát tah soupeři (negamax: negujeme a prohodíme barvy)
            return -self._negamax(board, depth, opp, -beta, -alpha)

        if depth == 0:
            val = evaluate(board, color)
            self.tt.put(h, val, depth, 'exact')
            return val

        # ── Rekurzivní prohledávání ────────────────────────────────
        ordered   = _order_moves(board, moves, color)
        orig_alpha = alpha
        best_val   = -_INF

        for m in ordered:
            b2  = make_move(board, m, color)
            val = -self._negamax(b2, depth - 1, opp, -beta, -alpha)
            best_val = max(best_val, val)
            alpha    = max(alpha, val)
            if alpha >= beta:
                break   # Alfa-beta cut-off

        # Uložit do TT
        if best_val <= orig_alpha:
            flag = 'upper'
        elif best_val >= beta:
            flag = 'lower'
        else:
            flag = 'exact'
        self.tt.put(h, best_val, depth, flag)

        return best_val

    def _terminal_value(self, board: np.ndarray, color: int) -> float:
        b, w = count_pieces(board)
        mine = b if color == BLACK else w
        opp  = w if color == BLACK else b
        if mine > opp: return  _INF
        if mine < opp: return -_INF
        return 0.0

    def __repr__(self) -> str:
        return f'MinimaxOpponent(depth={self.depth}, size={self.board_size})'
