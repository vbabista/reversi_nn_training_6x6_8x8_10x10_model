"""
opponents/random_opponent.py  –  Náhodný oponent.

Hraje vždy náhodný validní tah. Použití: první fáze kurikula (learning basics).
"""

import random
from typing import Optional, Tuple, List
import numpy as np

from core.board import get_valid_moves


class RandomOpponent:


    def __init__(self, board_size: int = 8):
        self.board_size = board_size

    def get_move(self, board: np.ndarray, color: int) -> Optional[Tuple[int, int]]:
        moves = get_valid_moves(board, color)
        if not moves:
            return None
        return random.choice(moves)

    def __repr__(self) -> str:
        return f'RandomOpponent(size={self.board_size})'
