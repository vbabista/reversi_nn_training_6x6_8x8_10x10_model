"""
training/replay_buffer.py  –  Prioritizovaný Experience Replay Buffer.

Standardní RL replay buffer pro TD learning:
  - Circular buffer (přepíše nejstarší záznamy)
  - Uniform sampling (default) nebo Prioritized (PER)
  - Efektivní NumPy implementace bez závislostí

Formát záznamu: (state, target)
  state  : np.ndarray float32, shape (input_size,)
  target : float32, TD target v [-1, 1]

Poznámka k prioritizaci (PER):
  PER vzorkuje proporcionálně k |TD error|^α.
  Tato implementace používá zjednodušené PER s segment-tree-like
  sumtree implementovaným v NumPy.
"""

import numpy as np
from typing import Tuple, Optional


# ══════════════════════════════════════════════════════════════════════
# UNIFORM REPLAY BUFFER (základní)
# ══════════════════════════════════════════════════════════════════════

class ReplayBuffer:


    def __init__(self, capacity: int, input_size: int, use_per: bool = True):
        self.capacity   = capacity
        self.input_size = input_size
        self.use_per    = use_per

        # Circular buffer
        self.states  = np.zeros((capacity, input_size), dtype=np.float32)
        self.targets = np.zeros(capacity,               dtype=np.float32)
        self.ptr     = 0       # ukazatel na aktuální pozici
        self.size    = 0       # aktuální počet uložených přechodů

        # PER: priorities (sumtree alternativa – přímé pole)
        if use_per:
            self.priorities = np.zeros(capacity, dtype=np.float32)
            self.per_alpha  = 0.6     # exponent prioritizace
            self.per_beta   = 0.4     # exponent IS korekce (roste → 1.0)
            self.per_eps    = 1e-5    # min priorita (nemůže být 0)
            self.max_prio   = 1.0     # max priorita viděná

    def add(self, state: np.ndarray, target: float,
            priority: Optional[float] = None):
        """Přidá jeden přechod do bufferu."""
        self.states[self.ptr]  = state.astype(np.float32)
        self.targets[self.ptr] = float(target)

        if self.use_per:
            p = float(priority) if priority is not None else self.max_prio
            self.priorities[self.ptr] = p
            self.max_prio = max(self.max_prio, p)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states: np.ndarray, targets: np.ndarray):

        n = len(states)
        for i in range(n):
            self.add(states[i], targets[i])

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:

        n = min(self.size, self.capacity)

        if self.use_per:
            # PER sampling: proporcionálně k priorities^alpha
            prios = self.priorities[:n] ** self.per_alpha
            probs = prios / prios.sum()
            indices = np.random.choice(n, size=batch_size, replace=False
                                        if n >= batch_size else True, p=probs)
            # Importance-sampling korekce
            is_weights = (n * probs[indices]) ** (-self.per_beta)
            is_weights /= is_weights.max()
            is_weights = is_weights.astype(np.float32)
            # Postupně zvyšujeme beta → plná IS korekce
            self.per_beta = min(1.0, self.per_beta + 2e-6)
        else:
            indices    = np.random.choice(n, size=batch_size,
                                          replace=n < batch_size)
            is_weights = np.ones(batch_size, dtype=np.float32)

        return (self.states[indices],
                self.targets[indices],
                indices,
                is_weights)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):

        if not self.use_per:
            return
        prios = np.abs(td_errors) + self.per_eps
        self.priorities[indices] = prios.astype(np.float32)
        self.max_prio = max(self.max_prio, float(prios.max()))

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (f'ReplayBuffer(size={self.size}/{self.capacity}, '
                f'per={self.use_per}, input_size={self.input_size})')
