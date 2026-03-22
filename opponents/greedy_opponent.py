"""
opponents/greedy_opponent.py

Vybírá tah který okamžitě obrátí nejvíce kamenů.
Je silnější než Random, ale slabší než Minimax.
Vhodný pro 2. fázi kurikula.
"""

import numpy as np
from typing import Optional, Tuple, List

from core.board import get_valid_moves, make_move, count_pieces


class GreedyOpponent:

    def __init__(self, board_size: int = 8):
        self.board_size = board_size

    def get_move(self, board: np.ndarray, color: int) -> Optional[Tuple[int, int]]:
        moves = get_valid_moves(board, color)
        if not moves:
            return None

        best_move  = None
        best_gain  = -1
        my_count   = int((board == color).sum())

        for m in moves:
            b2        = make_move(board, m, color)
            new_count = int((b2 == color).sum())
            gain      = new_count - my_count
            if gain > best_gain:
                best_gain = gain
                best_move = m

        return best_move

    def __repr__(self) -> str:
        return f'GreedyOpponent(size={self.board_size})'
