"""
Konvence:
    EMPTY = -1
    BLACK =  0   (první hráč)
    WHITE =  1   (druhý hráč)

Funkce:
    init_board(size)              → np.ndarray (size, size)
    get_valid_moves(board, color) → list[(row, col)]
    make_move(board, move, color) → np.ndarray (kopie)
    make_move_inplace(board, move, color) → None  [modifikuje board]
    is_terminal(board)            → bool
    count_pieces(board)           → (black_count, white_count)
    game_result(board, color)     → float +1/0/-1
    board_hash(board)             → int   (Zobrist hash pro transpoziční tabulku)

Optimalizace:
    - get_valid_moves: early-exit hledání v každém směru
    - make_move: přímý scan bez allokace dočasných listů
    - Zobrist hash: předpočítané random hodnoty per (position, color)
"""

import numpy as np
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────
# Konstanty
# ─────────────────────────────────────────────────────────────

EMPTY = -1
BLACK =  0
WHITE =  1

DIRECTIONS: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

# ─────────────────────────────────────────────────────────────
# Zobrist hash (přepočítané jednou při importu)
# ─────────────────────────────────────────────────────────────

_MAX_BOARD = 10
_rng_z = np.random.default_rng(42)
_ZOBRIST = {
    color: _rng_z.integers(0, 2**62, size=(_MAX_BOARD, _MAX_BOARD), dtype=np.int64)
    for color in (BLACK, WHITE)
}


def board_hash(board: np.ndarray) -> int:
    n = board.shape[0]
    h = np.int64(0)
    for color in (BLACK, WHITE):
        mask = (board == color)
        rows, cols = np.nonzero(mask)
        for r, c in zip(rows, cols):
            h ^= _ZOBRIST[color][r, c]
    return int(h)


# ─────────────────────────────────────────────────────────────
# Inicializace desky
# ─────────────────────────────────────────────────────────────

def init_board(size: int) -> np.ndarray:
    board = np.full((size, size), EMPTY, dtype=np.int8)
    mid = size // 2
    board[mid - 1, mid - 1] = WHITE
    board[mid - 1, mid]     = BLACK
    board[mid,     mid - 1] = BLACK
    board[mid,     mid]     = WHITE
    return board


# ─────────────────────────────────────────────────────────────
# Validní tahy
# ─────────────────────────────────────────────────────────────

def get_valid_moves(board: np.ndarray, color: int) -> List[Tuple[int, int]]:
    size = board.shape[0]
    opp  = 1 - color
    valid: List[Tuple[int, int]] = []

    for r in range(size):
        for c in range(size):
            if board[r, c] != EMPTY:
                continue
            # Zkontrolujeme všechny 8 směry
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                # Potřebujeme alespoň jednoho sousedního oponenta
                if not (0 <= nr < size and 0 <= nc < size and board[nr, nc] == opp):
                    continue
                # Jdeme dál hledat náš kámen
                nr += dr
                nc += dc
                while 0 <= nr < size and 0 <= nc < size:
                    cell = board[nr, nc]
                    if cell == color:
                        valid.append((r, c))
                        goto_next_cell = True
                        break
                    elif cell == EMPTY:
                        break  # přerušeno prázdným polem
                    nr += dr
                    nc += dc
                else:
                    goto_next_cell = False
                    continue
                if 'goto_next_cell' in dir() and goto_next_cell:
                    break  # našli jsme platný tah v tomto poli, přejdeme na další
    return valid


def _is_valid_move(board: np.ndarray, r: int, c: int, color: int) -> bool:
    size = board.shape[0]
    if board[r, c] != EMPTY:
        return False
    opp = 1 - color
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < size and 0 <= nc < size and board[nr, nc] == opp):
            continue
        nr += dr
        nc += dc
        while 0 <= nr < size and 0 <= nc < size:
            cell = board[nr, nc]
            if cell == color:
                return True
            elif cell == EMPTY:
                break
            nr += dr
            nc += dc
    return False


# ─────────────────────────────────────────────────────────────
# Provedení tahu
# ─────────────────────────────────────────────────────────────

def make_move(board: np.ndarray, move: Tuple[int, int], color: int) -> np.ndarray:
    new_board = board.copy()
    make_move_inplace(new_board, move, color)
    return new_board


def make_move_inplace(board: np.ndarray, move: Tuple[int, int], color: int) -> None:
    size  = board.shape[0]
    opp   = 1 - color
    r, c  = move
    board[r, c] = color

    for dr, dc in DIRECTIONS:
        # Sbíráme oponentovy kameny v tomto směru
        to_flip = []
        nr, nc  = r + dr, c + dc
        while 0 <= nr < size and 0 <= nc < size and board[nr, nc] == opp:
            to_flip.append((nr, nc))
            nr += dr
            nc += dc
        # Přehodíme jen pokud na konci je náš kámen
        if (to_flip and
                0 <= nr < size and 0 <= nc < size and
                board[nr, nc] == color):
            for fr, fc in to_flip:
                board[fr, fc] = color


# ─────────────────────────────────────────────────────────────
# Stav hry
# ─────────────────────────────────────────────────────────────

def is_terminal(board: np.ndarray) -> bool:
    return (not get_valid_moves(board, BLACK) and
            not get_valid_moves(board, WHITE))


def count_pieces(board: np.ndarray) -> Tuple[int, int]:
    return int((board == BLACK).sum()), int((board == WHITE).sum())


def game_result(board: np.ndarray, color: int) -> float:
    b, w = count_pieces(board)
    mine = b if color == BLACK else w
    opp  = w if color == BLACK else b
    if mine > opp:
        return  1.0
    if mine < opp:
        return -1.0
    return  0.0


def board_to_str(board: np.ndarray) -> str:
    n = board.shape[0]
    symbols = {BLACK: '●', WHITE: '○', EMPTY: '·'}
    header = '  ' + ' '.join(str(c) for c in range(n))
    rows   = [header]
    for r in range(n):
        row = str(r) + ' ' + ' '.join(symbols[int(board[r, c])] for c in range(n))
        rows.append(row)
    return '\n'.join(rows)
