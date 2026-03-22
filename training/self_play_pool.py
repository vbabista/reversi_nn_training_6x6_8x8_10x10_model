"""
training/self_play_pool.py  –  Pool historických snapshotů pro self-play.

Self-play s jedním modelem → model se učí porážet sám sebe (limitované).
Self-play s poolem historických verzí → diverzita, stabilní learning signal.

AlphaGo Zero přístup adaptovaný pro Reversi:
  - Pool obsahuje posledních K snapshots modelu
  - Pro každou self-play hru: soupeř = náhodná verze z poolu
  - Nový snapshot se přidá každých N epizod
  - Nejstarší snapshot se zahodí pokud pool přeplní kapacitu

Výhody oproti pure self-play:
  1. Diverzita oponentů → model nemůže overfit na aktuální verzi
  2. Stabilní learning signal (oponent se nemění každou hru)
  3. Uchování dovedností naučených v minulých fázích

Pool je sdílená instance per board_size (vytvořena v train.py).
"""

import random
from typing import Optional, List
import numpy as np


class SelfPlayPool:

    def __init__(self, max_size: int = 8):
        self.max_size  = max_size
        self._pool: List = []       # list ReversiNet kopií
        self._episode_stamps: List[int] = []   # epizoda přidání

    def add(self, nn, episode: int):

        snapshot = nn.copy()
        self._pool.append(snapshot)
        self._episode_stamps.append(episode)

        # Udržujeme max_size (smazat nejstarší)
        if len(self._pool) > self.max_size:
            self._pool.pop(0)
            self._episode_stamps.pop(0)

    def sample(self):

        if not self._pool:
            return None
        return random.choice(self._pool)

    def sample_weighted(self, current_nn):

        if not self._pool:
            return current_nn
        # Geometrické váhy: nejnovější má největší pravděpodobnost
        n      = len(self._pool)
        weights = np.array([(0.8 ** (n - 1 - i)) for i in range(n)], dtype=np.float32)
        weights /= weights.sum()
        idx = int(np.random.choice(n, p=weights))
        return self._pool[idx]

    def __len__(self) -> int:
        return len(self._pool)

    def __repr__(self) -> str:
        stamps = self._episode_stamps[-3:] if self._episode_stamps else []
        return f'SelfPlayPool(size={len(self._pool)}/{self.max_size}, recent_eps={stamps})'
