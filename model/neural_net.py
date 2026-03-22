"""
model/neural_net.py  –  Hluboká reziduální síť pro Reversi RL (pouze NumPy).

Architektura  (pro N×N desku):
──────────────────────────────────────────────────────────────────────────
    INPUT: 4 × N² (4 kanály: moje_kameny, opp_kameny, moje_tahy, opp_tahy)

    LinearBlock(input → H)          # vstupní projekce
    ResidualBlock × R               # R reziduálních bloků (H→H)
    LinearBlock(H → H//2)           # bottle-neck
    Linear(H//2 → 1) + tanh         # skórovaný výstup v (-1, 1)

    Kde H ∈ {512, 1024}, R ∈ {3, 4} dle velikosti desky (viz config.py)

Reziduální blok:
    x → LayerNorm → Linear → LeakyReLU → LayerNorm → Linear → LeakyReLU → + x

LayerNorm (implementace v NumPy):
    - Normalizuje přes feature dimenzi (axis=-1)
    - Learnable scale γ a offset β per vrstva

Optimalizátor: Adam (β₁=0.9, β₂=0.999)
    - Gradient clipping (per-tensor L2 norm)
    - L2 regularizace
    - Cyclické LR scheduling (spravováno zvenku)

Uložení: np.save(..., allow_pickle=True) jako dict {name: np.ndarray}
    Soubor: models/{size}x{size}/model.npy  nebo  checkpoint_N.npy

Veřejné API:
    ReversiNet(input_size, H, res_blocks, lr)
    .forward(x)                       → (batch, 1) tanh hodnoty
    .score_moves(board, moves, color) → (n_moves,) np.ndarray  [nejrychlejší cesta]
    .train_batch(states, targets)     → float loss
    .soft_update_from(other, tau)     → zkopíruje váhy s koeficientem tau
    .copy()                           → hluboká kopie
    .save(path)
    .load(path) [statická]
"""

import numpy as np
import os
from typing import List, Tuple, Optional

# Lazy import core.board – funguje po přidání root do sys.path v train.py
def _get_valid_moves(board, color):
    from core.board import get_valid_moves
    return get_valid_moves(board, color)

def _make_move(board, move, color):
    from core.board import make_move
    return make_move(board, move, color)


# ══════════════════════════════════════════════════════════════════════
# KÓDOVÁNÍ STAVU DESKY
# ══════════════════════════════════════════════════════════════════════

def encode_board(board: np.ndarray, color: int) -> np.ndarray:

    size  = board.shape[0]
    n_sq  = size * size
    opp   = 1 - color

    ch0 = (board == color).astype(np.float32).flatten()
    ch1 = (board == opp).astype(np.float32).flatten()

    ch2 = np.zeros(n_sq, dtype=np.float32)
    for r, c in _get_valid_moves(board, color):
        ch2[r * size + c] = 1.0

    ch3 = np.zeros(n_sq, dtype=np.float32)
    for r, c in _get_valid_moves(board, opp):
        ch3[r * size + c] = 1.0

    return np.concatenate([ch0, ch1, ch2, ch3]).reshape(1, -1)


def encode_moves_batch(board: np.ndarray, moves: List[Tuple[int, int]], color: int) -> np.ndarray:

    size  = board.shape[0]
    n_sq  = size * size
    opp   = 1 - color
    n     = len(moves)
    out   = np.zeros((n, 4 * n_sq), dtype=np.float32)

    for i, m in enumerate(moves):
        b2 = _make_move(board, m, color)

        out[i, :n_sq]          = (b2 == color).astype(np.float32).flatten()
        out[i, n_sq:2*n_sq]    = (b2 == opp).astype(np.float32).flatten()

        for r, c in _get_valid_moves(b2, color):
            out[i, 2*n_sq + r*size + c] = 1.0
        for r, c in _get_valid_moves(b2, opp):
            out[i, 3*n_sq + r*size + c] = 1.0

    return out


# ══════════════════════════════════════════════════════════════════════
# LAYER NORMALIZATION (NumPy)
# ══════════════════════════════════════════════════════════════════════

class LayerNorm:

    LN_EPS = 1e-5

    def __init__(self, dim: int, name: str = ''):
        self.dim   = dim
        self.name  = name
        self.gamma = np.ones(dim, dtype=np.float32)   # scale
        self.beta  = np.zeros(dim, dtype=np.float32)  # bias

        # Adam momenty
        self.m_gamma = np.zeros(dim, dtype=np.float32)
        self.v_gamma = np.zeros(dim, dtype=np.float32)
        self.m_beta  = np.zeros(dim, dtype=np.float32)
        self.v_beta  = np.zeros(dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        mu  = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + self.LN_EPS
        xn  = (x - mu) / std
        y   = self.gamma * xn + self.beta
        return y, xn, std

    def backward(self, dy: np.ndarray, xn: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        B, D = dy.shape
        d_beta  = dy.sum(axis=0)
        d_gamma = (dy * xn).sum(axis=0)

        dxn = dy * self.gamma
        dvar = (-0.5 * (dxn * xn / std).sum(axis=-1, keepdims=True))
        dmu  = (-dxn / std).sum(axis=-1, keepdims=True) + dvar * (-2.0 * xn / D)
        dx   = dxn / std + dvar * 2.0 * xn / D + dmu / D

        return dx, d_gamma, d_beta

    def state_dict(self) -> dict:
        return {
            f'{self.name}_gamma':   self.gamma,
            f'{self.name}_beta':    self.beta,
            f'{self.name}_m_gamma': self.m_gamma,
            f'{self.name}_v_gamma': self.v_gamma,
            f'{self.name}_m_beta':  self.m_beta,
            f'{self.name}_v_beta':  self.v_beta,
        }

    def load_state_dict(self, d: dict):
        self.gamma   = d[f'{self.name}_gamma'].astype(np.float32)
        self.beta    = d[f'{self.name}_beta'].astype(np.float32)
        self.m_gamma = d.get(f'{self.name}_m_gamma', np.zeros_like(self.gamma))
        self.v_gamma = d.get(f'{self.name}_v_gamma', np.zeros_like(self.gamma))
        self.m_beta  = d.get(f'{self.name}_m_beta',  np.zeros_like(self.beta))
        self.v_beta  = d.get(f'{self.name}_v_beta',  np.zeros_like(self.beta))


# ══════════════════════════════════════════════════════════════════════
# HLAVNÍ SÍŤOVÁ TŘÍDA
# ══════════════════════════════════════════════════════════════════════

class ReversiNet:
    # Adam hyperparametry
    BETA1 = 0.9
    BETA2 = 0.999
    EPS   = 1e-8
    L2    = 5e-6
    CLIP  = 2.0       # Max L2 norm gradientu per tensor
    LRELU = 0.05      # Koeficient Leaky ReLU záporné větve

    def __init__(self, input_size: int, H: int = 512, res_blocks: int = 3,
                 lr: float = 3e-4):

        self.input_size = input_size
        self.H          = H
        self.res_blocks = res_blocks
        self.lr         = lr
        self.t          = 0          # Adam krok

        rng = np.random.default_rng(1337)

        def he(fan_in, fan_out):
            return rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)).astype(np.float32)

        def zeros(shape):
            return np.zeros(shape, dtype=np.float32)

        # ── Vstupní projekce: input_size → H ──────────────────────────
        self.W_in = he(input_size, H);  self.b_in = zeros((1, H))
        self.ln_in = LayerNorm(H, 'ln_in')

        # ── Reziduální bloky: H → H ────────────────────────────────────
        # Každý blok: LN → Linear → LReLU → LN → Linear → LReLU → +x
        self.res_W1 = [he(H, H) for _ in range(res_blocks)]
        self.res_b1 = [zeros((1, H)) for _ in range(res_blocks)]
        self.res_W2 = [he(H, H) for _ in range(res_blocks)]
        self.res_b2 = [zeros((1, H)) for _ in range(res_blocks)]
        self.res_ln1 = [LayerNorm(H, f'res{i}_ln1') for i in range(res_blocks)]
        self.res_ln2 = [LayerNorm(H, f'res{i}_ln2') for i in range(res_blocks)]

        # ── Bottle-neck: H → H//2 ────────────────────────────────────
        H2 = H // 2
        self.W_bn = he(H, H2);  self.b_bn = zeros((1, H2))
        self.ln_bn = LayerNorm(H2, 'ln_bn')

        # ── Výstupní hlava: H//2 → 1 ──────────────────────────────────
        self.W_out = rng.normal(0, 0.01, (H2, 1)).astype(np.float32)
        self.b_out = zeros((1, 1))

        # ── Adam momenty (všechny lineární váhy) ──────────────────────
        self._param_names = self._collect_param_names()
        self._init_adam()

    # ─────────────────────────────────────────────────────────────
    # Aktivační funkce
    # ─────────────────────────────────────────────────────────────

    def _lrelu(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.LRELU * x)

    def _lrelu_d(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, self.LRELU).astype(np.float32)

    # ─────────────────────────────────────────────────────────────
    # Adam pomocné metody
    # ─────────────────────────────────────────────────────────────

    def _collect_param_names(self) -> List[str]:
        names = ['W_in', 'b_in', 'W_bn', 'b_bn', 'W_out', 'b_out']
        for i in range(self.res_blocks):
            names += [f'resW1_{i}', f'resb1_{i}', f'resW2_{i}', f'resb2_{i}']
        return names

    def _get_param(self, name: str) -> np.ndarray:
        # resW1_0  →  res_W1[0]
        if name.startswith('res') and '_' in name:
            key, idx_s = name.rsplit('_', 1)
            idx = int(idx_s)
            # key = 'resW1' → attr = 'res_W1'
            attr = 'res_' + key[3:]
            return getattr(self, attr)[idx]
        return getattr(self, name)

    def _set_param(self, name: str, val: np.ndarray):
        if name.startswith('res') and '_' in name:
            key, idx_s = name.rsplit('_', 1)
            idx  = int(idx_s)
            attr = 'res_' + key[3:]
            getattr(self, attr)[idx] = val
        else:
            setattr(self, name, val)

    def _init_adam(self):
        for name in self._param_names:
            p = self._get_param(name)
            setattr(self, f'm_{name}', np.zeros_like(p))
            setattr(self, f'v_{name}', np.zeros_like(p))

    # ─────────────────────────────────────────────────────────────
    # FORWARD PASS
    # ─────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:

        # Vstupní projekce
        h, _, _ = self.ln_in.forward(x @ self.W_in + self.b_in)
        h = self._lrelu(h)

        # Reziduální bloky
        for i in range(self.res_blocks):
            # První polovina bloku
            h1, _, _ = self.res_ln1[i].forward(h @ self.res_W1[i] + self.res_b1[i])
            h1 = self._lrelu(h1)
            # Druhá polovina bloku
            h2, _, _ = self.res_ln2[i].forward(h1 @ self.res_W2[i] + self.res_b2[i])
            h2 = self._lrelu(h2)
            # Reziduální spojení
            h = h + h2

        # Bottle-neck
        h, _, _ = self.ln_bn.forward(h @ self.W_bn + self.b_bn)
        h = self._lrelu(h)

        # Výstup
        return np.tanh(h @ self.W_out + self.b_out)

    # ─────────────────────────────────────────────────────────────
    # FORWARD S CACHE (pro backward)
    # ─────────────────────────────────────────────────────────────

    def _forward_with_cache(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:

        cache = {'x': x}

        # Vstupní projekce
        z_in = x @ self.W_in + self.b_in
        h_in_ln, xn_in, std_in = self.ln_in.forward(z_in)
        h = self._lrelu(h_in_ln)
        cache.update({'z_in': z_in, 'h_in_ln': h_in_ln,
                       'xn_in': xn_in, 'std_in': std_in, 'h_in': h})

        # Reziduální bloky
        for i in range(self.res_blocks):
            h_prev = h

            z1 = h @ self.res_W1[i] + self.res_b1[i]
            h1_ln, xn1, std1 = self.res_ln1[i].forward(z1)
            h1 = self._lrelu(h1_ln)

            z2 = h1 @ self.res_W2[i] + self.res_b2[i]
            h2_ln, xn2, std2 = self.res_ln2[i].forward(z2)
            h2 = self._lrelu(h2_ln)

            h = h_prev + h2

            cache[f'h_prev_{i}'] = h_prev
            cache[f'z1_{i}']     = z1
            cache[f'h1_ln_{i}']  = h1_ln
            cache[f'xn1_{i}']    = xn1
            cache[f'std1_{i}']   = std1
            cache[f'h1_{i}']     = h1
            cache[f'z2_{i}']     = z2
            cache[f'h2_ln_{i}']  = h2_ln
            cache[f'xn2_{i}']    = xn2
            cache[f'std2_{i}']   = std2
            cache[f'h2_{i}']     = h2
            cache[f'h_out_{i}']  = h

        # Bottle-neck
        z_bn = h @ self.W_bn + self.b_bn
        h_bn_ln, xn_bn, std_bn = self.ln_bn.forward(z_bn)
        h_bn = self._lrelu(h_bn_ln)
        cache.update({'h_res_final': h, 'z_bn': z_bn, 'h_bn_ln': h_bn_ln,
                       'xn_bn': xn_bn, 'std_bn': std_bn, 'h_bn': h_bn})

        # Výstup
        z_out = h_bn @ self.W_out + self.b_out
        out   = np.tanh(z_out)
        cache.update({'z_out': z_out, 'out': out})

        return out, cache

    # ─────────────────────────────────────────────────────────────
    # BACKWARD PASS + ADAM UPDATE
    # ─────────────────────────────────────────────────────────────

    def train_batch(self, states: np.ndarray, targets: np.ndarray) -> float:

        B = states.shape[0]
        X = states.astype(np.float32)
        T = targets.reshape(B, 1).astype(np.float32)

        # ── Forward s cache ──────────────────────────────────────
        out, cache = self._forward_with_cache(X)
        loss = float(np.mean((out - T) ** 2))

        # ── Backward: dL/dout ─────────────────────────────────────
        # MSE: dL/dout = 2*(out-T)/B
        # tanh': 1 - tanh(z)^2 = 1 - out^2
        dz_out = (2.0 / B) * (out - T) * (1.0 - out ** 2)

        dW_out  = cache['h_bn'].T @ dz_out
        db_out  = dz_out.sum(0, keepdims=True)
        dh_bn   = dz_out @ self.W_out.T

        # ── Bottle-neck backward ──────────────────────────────────
        dh_bn_lrelu = dh_bn * self._lrelu_d(cache['h_bn_ln'])
        dh_bn_ln, dg_bn, db_bn_ln = self.ln_bn.backward(
            dh_bn_lrelu, cache['xn_bn'], cache['std_bn'])

        dz_bn = dh_bn_ln
        dW_bn = cache['h_res_final'].T @ dz_bn
        db_bn = dz_bn.sum(0, keepdims=True)
        dh    = dz_bn @ self.W_bn.T

        # ── Reziduální bloky backward (v opačném pořadí) ──────────
        dres_W1, dres_b1, dres_W2, dres_b2 = [], [], [], []
        dln1_g, dln1_b, dln2_g, dln2_b = [], [], [], []

        for i in reversed(range(self.res_blocks)):
            # Reziduální přičtení: h = h_prev + h2 → gradient prochází oběma
            dh_prev_res = dh.copy()   # část přes skip
            dh2         = dh.copy()   # část přes blok

            # h2 = lrelu(h2_ln) → dh2_ln
            dh2_lrelu = dh2 * self._lrelu_d(cache[f'h2_ln_{i}'])
            dh2_ln, dg2, db2 = self.res_ln2[i].backward(
                dh2_lrelu, cache[f'xn2_{i}'], cache[f'std2_{i}'])

            dz2 = dh2_ln
            dW2 = cache[f'h1_{i}'].T @ dz2
            db2_ = dz2.sum(0, keepdims=True)
            dh1  = dz2 @ self.res_W2[i].T

            # h1 = lrelu(h1_ln) → dh1_ln
            dh1_lrelu = dh1 * self._lrelu_d(cache[f'h1_ln_{i}'])
            dh1_ln, dg1, db1 = self.res_ln1[i].backward(
                dh1_lrelu, cache[f'xn1_{i}'], cache[f'std1_{i}'])

            dz1 = dh1_ln
            dW1 = cache[f'h_prev_{i}'].T @ dz1
            db1_ = dz1.sum(0, keepdims=True)
            dh_from_blk = dz1 @ self.res_W1[i].T

            # Gradient do předchozí vrstvy = skip + blok
            dh = dh_prev_res + dh_from_blk

            # Accumulate (reversed → prepend)
            dres_W1.insert(0, dW1);  dres_b1.insert(0, db1_)
            dres_W2.insert(0, dW2);  dres_b2.insert(0, db2_)
            dln1_g.insert(0, dg1);   dln1_b.insert(0, db1)
            dln2_g.insert(0, dg2);   dln2_b.insert(0, db2)

        # ── Vstupní projekce backward ─────────────────────────────
        dh_in_lrelu = dh * self._lrelu_d(cache['h_in_ln'])
        dh_in_ln, dg_in, db_in_ln = self.ln_in.backward(
            dh_in_lrelu, cache['xn_in'], cache['std_in'])

        dz_in = dh_in_ln
        dW_in = X.T @ dz_in
        db_in = dz_in.sum(0, keepdims=True)

        # ── L2 regularizace (pouze na váhové matice, ne biasy/LN) ─
        dW_in  += self.L2 * self.W_in
        dW_bn  += self.L2 * self.W_bn
        dW_out += self.L2 * self.W_out
        for i in range(self.res_blocks):
            dres_W1[i] += self.L2 * self.res_W1[i]
            dres_W2[i] += self.L2 * self.res_W2[i]

        # ── Gradient clipping (per-tensor L2 norm) ─────────────────
        grads_main = [dW_in, db_in, dW_bn, db_bn, dW_out, db_out]
        for g in grads_main:
            self._clip_grad(g)
        for i in range(self.res_blocks):
            self._clip_grad(dres_W1[i]); self._clip_grad(dres_b1[i])
            self._clip_grad(dres_W2[i]); self._clip_grad(dres_b2[i])

        # ── Adam update: lineární váhy ─────────────────────────────
        self.t += 1
        named_grads = [
            ('W_in', dW_in), ('b_in', db_in),
            ('W_bn', dW_bn), ('b_bn', db_bn),
            ('W_out', dW_out), ('b_out', db_out),
        ]
        for i in range(self.res_blocks):
            named_grads += [
                (f'resW1_{i}', dres_W1[i]), (f'resb1_{i}', dres_b1[i]),
                (f'resW2_{i}', dres_W2[i]), (f'resb2_{i}', dres_b2[i]),
            ]
        for name, grad in named_grads:
            self._adam_step_named(name, grad)

        # ── Adam update: LayerNorm parametry ──────────────────────
        self._adam_ln(self.ln_in,  dg_in,  db_in_ln)
        self._adam_ln(self.ln_bn,  dg_bn,  db_bn_ln)
        for i in range(self.res_blocks):
            self._adam_ln(self.res_ln1[i], dln1_g[i], dln1_b[i])
            self._adam_ln(self.res_ln2[i], dln2_g[i], dln2_b[i])

        return loss

    def _clip_grad(self, g: np.ndarray):
        norm = float(np.sqrt(np.sum(g * g)))
        if norm > self.CLIP:
            g *= self.CLIP / norm

    def _adam_step_named(self, name: str, grad: np.ndarray):
        p = self._get_param(name)
        m = getattr(self, f'm_{name}')
        v = getattr(self, f'v_{name}')

        m = self.BETA1 * m + (1.0 - self.BETA1) * grad
        v = self.BETA2 * v + (1.0 - self.BETA2) * (grad ** 2)

        mh = m / (1.0 - self.BETA1 ** self.t)
        vh = v / (1.0 - self.BETA2 ** self.t)

        p -= self.lr * mh / (np.sqrt(vh) + self.EPS)

        self._set_param(name, p)
        setattr(self, f'm_{name}', m)
        setattr(self, f'v_{name}', v)

    def _adam_ln(self, ln: LayerNorm, dg: np.ndarray, db: np.ndarray):
        """Adam update pro LayerNorm parametry (gamma a beta)."""
        for attr, grad, m_attr, v_attr in [
            ('gamma', dg, 'm_gamma', 'v_gamma'),
            ('beta',  db, 'm_beta',  'v_beta'),
        ]:
            p = getattr(ln, attr)
            m = getattr(ln, m_attr)
            v = getattr(ln, v_attr)

            m = self.BETA1 * m + (1.0 - self.BETA1) * grad
            v = self.BETA2 * v + (1.0 - self.BETA2) * (grad ** 2)

            mh = m / (1.0 - self.BETA1 ** self.t)
            vh = v / (1.0 - self.BETA2 ** self.t)

            p -= self.lr * mh / (np.sqrt(vh) + self.EPS)

            setattr(ln, attr,   p)
            setattr(ln, m_attr, m)
            setattr(ln, v_attr, v)

    # ─────────────────────────────────────────────────────────────
    # VÝBĚR TAHU (inference)
    # ─────────────────────────────────────────────────────────────

    def score_moves(self, board: np.ndarray, moves: List[Tuple[int, int]],
                    color: int) -> np.ndarray:

        if not moves:
            return np.array([], dtype=np.float32)
        batch  = encode_moves_batch(board, moves, color)   # (n, 4·N²)
        scores = self.forward(batch)                        # (n, 1)
        return scores.flatten()

    def best_move(self, board: np.ndarray, moves: List[Tuple[int, int]],
                  color: int) -> Tuple[int, int]:
        """Vrátí tah s nejvyšším skóre. Moves nesmí být prázdné."""
        scores = self.score_moves(board, moves, color)
        return moves[int(np.argmax(scores))]

    # ─────────────────────────────────────────────────────────────
    # SETOVÁNÍ LR (pro LR scheduling)
    # ─────────────────────────────────────────────────────────────

    def set_lr(self, lr: float):
        self.lr = float(lr)

    # ─────────────────────────────────────────────────────────────
    # TARGET NETWORK: SOFT UPDATE
    # ─────────────────────────────────────────────────────────────

    def soft_update_from(self, source: 'ReversiNet', tau: float = 0.005):

        for name in self._param_names:
            p_self   = self._get_param(name)
            p_source = source._get_param(name)
            self._set_param(name, tau * p_source + (1.0 - tau) * p_self)

        def _soft_ln(ln_self, ln_src):
            ln_self.gamma = tau * ln_src.gamma + (1.0 - tau) * ln_self.gamma
            ln_self.beta  = tau * ln_src.beta  + (1.0 - tau) * ln_self.beta

        _soft_ln(self.ln_in, source.ln_in)
        _soft_ln(self.ln_bn, source.ln_bn)
        for i in range(self.res_blocks):
            _soft_ln(self.res_ln1[i], source.res_ln1[i])
            _soft_ln(self.res_ln2[i], source.res_ln2[i])

    # ─────────────────────────────────────────────────────────────
    # KOPÍROVÁNÍ (pro self-play pool a target net)
    # ─────────────────────────────────────────────────────────────

    def copy(self) -> 'ReversiNet':

        clone = ReversiNet(self.input_size, self.H, self.res_blocks, self.lr)
        clone.t = self.t

        for name in self._param_names:
            clone._set_param(name, self._get_param(name).copy())
            setattr(clone, f'm_{name}', getattr(self, f'm_{name}').copy())
            setattr(clone, f'v_{name}', getattr(self, f'v_{name}').copy())

        def _copy_ln(dst, src):
            dst.gamma   = src.gamma.copy()
            dst.beta    = src.beta.copy()
            dst.m_gamma = src.m_gamma.copy()
            dst.v_gamma = src.v_gamma.copy()
            dst.m_beta  = src.m_beta.copy()
            dst.v_beta  = src.v_beta.copy()

        _copy_ln(clone.ln_in, self.ln_in)
        _copy_ln(clone.ln_bn, self.ln_bn)
        for i in range(self.res_blocks):
            _copy_ln(clone.res_ln1[i], self.res_ln1[i])
            _copy_ln(clone.res_ln2[i], self.res_ln2[i])

        return clone

    # ─────────────────────────────────────────────────────────────
    # ULOŽENÍ A NAČTENÍ JAKO .npy
    # ─────────────────────────────────────────────────────────────

    def save(self, path: str):

        if not path.endswith('.npy'):
            path += '.npy'
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        data = {
            'input_size': np.int32(self.input_size),
            'H':          np.int32(self.H),
            'res_blocks': np.int32(self.res_blocks),
            'lr':         np.float32(self.lr),
            't':          np.int64(self.t),
        }

        # Lineární váhy + Adam
        for name in self._param_names:
            data[f'p_{name}'] = self._get_param(name)
            data[f'm_{name}'] = getattr(self, f'm_{name}')
            data[f'v_{name}'] = getattr(self, f'v_{name}')

        # LayerNorm parametry
        def _save_ln(ln, prefix):
            data.update({
                f'{prefix}_gamma':   ln.gamma,
                f'{prefix}_beta':    ln.beta,
                f'{prefix}_m_gamma': ln.m_gamma,
                f'{prefix}_v_gamma': ln.v_gamma,
                f'{prefix}_m_beta':  ln.m_beta,
                f'{prefix}_v_beta':  ln.v_beta,
            })

        _save_ln(self.ln_in, 'ln_in')
        _save_ln(self.ln_bn, 'ln_bn')
        for i in range(self.res_blocks):
            _save_ln(self.res_ln1[i], f'res{i}_ln1')
            _save_ln(self.res_ln2[i], f'res{i}_ln2')

        np.save(path, data, allow_pickle=True)

    @staticmethod
    def load(path: str) -> 'ReversiNet':

        if not path.endswith('.npy'):
            path += '.npy'
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Model nenalezen: {path}')

        d = np.load(path, allow_pickle=True).item()

        input_size = int(d['input_size'])
        H          = int(d['H'])
        res_blocks = int(d['res_blocks'])
        lr         = float(d['lr'])

        net = ReversiNet(input_size, H, res_blocks, lr)
        net.t = int(d['t'])

        # Lineární váhy + Adam
        for name in net._param_names:
            if f'p_{name}' in d:
                net._set_param(name, d[f'p_{name}'].astype(np.float32))
            if f'm_{name}' in d:
                setattr(net, f'm_{name}', d[f'm_{name}'].astype(np.float32))
            if f'v_{name}' in d:
                setattr(net, f'v_{name}', d[f'v_{name}'].astype(np.float32))

        # LayerNorm
        def _load_ln(ln, prefix):
            if f'{prefix}_gamma' in d:
                ln.gamma   = d[f'{prefix}_gamma'].astype(np.float32)
                ln.beta    = d[f'{prefix}_beta'].astype(np.float32)
                ln.m_gamma = d.get(f'{prefix}_m_gamma', np.zeros_like(ln.gamma))
                ln.v_gamma = d.get(f'{prefix}_v_gamma', np.zeros_like(ln.gamma))
                ln.m_beta  = d.get(f'{prefix}_m_beta',  np.zeros_like(ln.beta))
                ln.v_beta  = d.get(f'{prefix}_v_beta',  np.zeros_like(ln.beta))

        _load_ln(net.ln_in, 'ln_in')
        _load_ln(net.ln_bn, 'ln_bn')
        for i in range(net.res_blocks):
            _load_ln(net.res_ln1[i], f'res{i}_ln1')
            _load_ln(net.res_ln2[i], f'res{i}_ln2')

        return net

    def __repr__(self) -> str:
        H2 = self.H // 2
        return (f'ReversiNet(in={self.input_size}, '
                f'H={self.H}, res={self.res_blocks}, '
                f'H//2={H2}, out=1[tanh], '
                f'lr={self.lr:.2e}, t={self.t})')
