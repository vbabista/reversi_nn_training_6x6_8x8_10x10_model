"""
train.py  –  Hlavní trénovací skript pro Reversi RL v2.1 (NumPy only).

Spuštění:
    python train.py                    # 6×6, 8×8, 10×10  (paralelně + dashboard)
    python train.py --size 8           # jen 8×8
    python train.py --fast             # 8% epizod (test)
    python train.py --resume           # pokračuj z checkpointu
    python train.py --no-adaptive      # scheduled curriculum
    python train.py --sequential       # zakáže paralelní trénink (debug)

NOVINKY v2.1:
  - Paralelní trénink: každá deska ve vlastním procesu (spawn)
  - Live multi-panel dashboard: každá deska má svůj panel v terminálu
  - Výstup každého workeru logován do logs/train_NxN.log (čitelné kdykoli)
  - Python minimax depth>=5: paralelní root-move evaluace (>=4 jádra)
  - C minimax: depth-only mode – garantovaná hloubka nezávislá na HW
"""

import sys
import os
import random
import time
import argparse
import threading
import shutil
import re
import multiprocessing as mp
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    ARCH, INPUT_CHANNELS, TOTAL_EPISODES, FAST_MULTIPLIER,
    LR_START, LR_END, LR_WARMUP,
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
    REPLAY_CAPACITY, REPLAY_MIN_FILL, BATCH_SIZE, TRAIN_EVERY,
    TARGET_NET_UPDATE, TARGET_NET_TAU,
    GAMMA, TD_LAMBDA,
    CHECKPOINT_INTERVAL, LOG_INTERVAL, MODEL_DIR, LOG_DIR,
    SELFPLAY_POOL_SIZE, SELFPLAY_SNAPSHOT_EVERY,
)
from core.board import (
    init_board, get_valid_moves, make_move,
    count_pieces, BLACK, WHITE,
)
from model.neural_net    import ReversiNet, encode_board, encode_moves_batch
from training.replay_buffer   import ReplayBuffer
from training.reward_shaper   import RewardShaper, margin_game_result, margin_label
from training.metrics_tracker import MetricsTracker
from training.curriculum      import CurriculumManager
from training.self_play_pool  import SelfPlayPool


# ======================================================================
# LR / EPSILON
# ======================================================================

def compute_lr(episode: int, total_eps: int) -> float:
    if episode < LR_WARMUP:
        return LR_START * episode / max(1, LR_WARMUP)
    progress = (episode - LR_WARMUP) / max(1, total_eps - LR_WARMUP)
    cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
    return LR_END + (LR_START - LR_END) * cosine


def compute_epsilon(episode: int) -> float:
    return max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))


# ======================================================================
# SINGLE GAME
# ======================================================================

def play_one_game(nn, nn_target, opponent, size, ai_color, epsilon,
                  global_progress, replay, shaper, self_pool):
    board   = init_board(size)
    color   = BLACK
    consec  = 0
    n_ai_moves   = 0
    forced_count = 0
    traj_states  = []
    traj_rewards = []
    corners_list = []
    mob_list     = []

    shaper.reset(board, ai_color)

    if opponent is None:
        selfplay_opp = self_pool.sample_weighted(nn) if len(self_pool) > 0 else nn

    while True:
        moves = get_valid_moves(board, color)
        if not moves:
            consec += 1
            if consec >= 2:
                break
            color = 1 - color
            continue
        consec = 0

        if color == ai_color:
            board_before = board.copy()
            n_ai_moves  += 1
            if random.random() < epsilon:
                chosen = random.choice(moves)
            else:
                scores = nn.score_moves(board, moves, color)
                chosen = moves[int(np.argmax(scores))]
            board  = make_move(board, chosen, color)
            r_step = shaper.step_reward(board_before, board, color)
            enc    = encode_board(board, color).flatten()
            traj_states.append(enc)
            traj_rewards.append(r_step)
            from training.reward_shaper import _corners as _c, _mobility_ratio as _mr
            corners_list.append(_c(board, color))
            mob_list.append(_mr(board, color))
            if len(get_valid_moves(board, 1 - color)) == 0:
                forced_count += 1
        else:
            if opponent is None:
                if random.random() < epsilon * 0.5:
                    chosen = random.choice(moves)
                else:
                    scores = selfplay_opp.score_moves(board, moves, color)
                    chosen = moves[int(np.argmax(scores))]
            else:
                chosen = opponent.get_move(board, color)
                if chosen is None:
                    consec += 1
                    color = 1 - color
                    continue
            board = make_move(board, chosen, color)

        color = 1 - color

    final_result = margin_game_result(board, ai_color)
    b_cnt, w_cnt = count_pieces(board)
    ai_discs  = b_cnt if ai_color == BLACK else w_cnt
    opp_discs = w_cnt if ai_color == BLACK else b_cnt

    improv     = shaper.game_improvement_bonus(board, ai_color)
    adj_result = float(np.clip(final_result + improv.get('total', 0.0), -1.0, 1.0))

    if traj_rewards:
        final_enc  = encode_board(board, ai_color).flatten()
        all_states = traj_states + [final_enc]
        td_targets = shaper.compute_td_targets(
            step_rewards    = traj_rewards,
            states          = all_states,
            nn_target       = nn_target,
            final_result    = adj_result,
            global_progress = global_progress,
            gamma           = GAMMA,
            td_lambda       = TD_LAMBDA,
        )
        for enc, td_t in zip(traj_states, td_targets):
            replay.add(enc, td_t)

    return {
        'result':        final_result,
        'ai_discs':      ai_discs,
        'opp_discs':     opp_discs,
        'n_ai_moves':    n_ai_moves,
        'forced_passes': forced_count,
        'corners_avg':   float(np.mean(corners_list)) if corners_list else 0.0,
        'mob_avg':       float(np.mean(mob_list))     if mob_list     else 0.0,
        'improvement':   improv.get('total', 0.0),
        'margin_label':  margin_label(board, ai_color),
    }


# ======================================================================
# TRAIN ONE SIZE
# ======================================================================

def train_size(size: int, total_eps: int, resume: bool = False,
               adaptive: bool = True):
    print(f'\n{"="*72}')
    print(f'  TRÉNINK  {size}x{size}  ({total_eps} epizod)')
    print(f'{"="*72}\n')

    input_size = INPUT_CHANNELS * size * size
    arch       = ARCH[size]
    model_dir  = os.path.join(MODEL_DIR, f'{size}x{size}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    final_path    = os.path.join(model_dir, f'model_{size}x{size}.npy')
    start_episode = 0

    if resume:
        cps = sorted([f for f in os.listdir(model_dir)
                      if f.startswith('checkpoint_ep') and f.endswith('.npy')])
        if cps:
            last_cp = os.path.join(model_dir, cps[-1])
            print(f'  Nacitam checkpoint: {last_cp}')
            nn = ReversiNet.load(last_cp)
            start_episode = max(0, nn.t // max(1, total_eps // 10))
            print(f'  Adam krok t={nn.t}, start_ep~{start_episode}')
        else:
            nn = ReversiNet(input_size, arch['H'], arch['res_blocks'], LR_START)
    else:
        nn = ReversiNet(input_size, arch['H'], arch['res_blocks'], LR_START)

    nn_target = nn.copy()
    print(f'  Sit:    {nn}')
    print(f'  Buffer: capacity={REPLAY_CAPACITY}, PER=True')
    print()
    sys.stdout.flush()

    replay     = ReplayBuffer(REPLAY_CAPACITY, input_size, use_per=True)
    shaper     = RewardShaper()
    metrics    = MetricsTracker(size, LOG_DIR)
    curriculum = CurriculumManager(size, total_eps, adaptive=adaptive)
    self_pool  = SelfPlayPool(SELFPLAY_POOL_SIZE)
    self_pool.add(nn, 0)

    for ep in range(start_episode, total_eps):
        global_progress = ep / max(1, total_eps - 1)
        lr      = compute_lr(ep, total_eps)
        epsilon = compute_epsilon(ep - start_episode)
        nn.set_lr(lr)

        opponent, phase_name, opp_name = curriculum.get_opponent(nn)
        ai_color = BLACK if ep % 2 == 0 else WHITE

        game_info = play_one_game(
            nn=nn, nn_target=nn_target,
            opponent=opponent, size=size,
            ai_color=ai_color, epsilon=epsilon,
            global_progress=global_progress,
            replay=replay, shaper=shaper,
            self_pool=self_pool,
        )

        curriculum.record_result(game_info['result'])

        ep_loss = 0.0
        if len(replay) >= REPLAY_MIN_FILL and ep % TRAIN_EVERY == 0:
            states, targets, indices, is_weights = replay.sample(BATCH_SIZE)
            ep_loss = nn.train_batch(states, targets)
            preds   = nn.forward(states).flatten()
            errors  = np.abs(preds - targets)
            replay.update_priorities(indices, errors)

        if ep % TARGET_NET_UPDATE == 0 and ep > 0:
            nn_target.soft_update_from(nn, TARGET_NET_TAU)

        if ep % SELFPLAY_SNAPSHOT_EVERY == 0 and ep > 0:
            self_pool.add(nn, ep)

        metrics.record_episode(
            result            = game_info['result'],
            ai_discs          = game_info['ai_discs'],
            opp_discs         = game_info['opp_discs'],
            n_ai_moves        = game_info['n_ai_moves'],
            td_loss           = ep_loss,
            corners_avg       = game_info['corners_avg'],
            mob_avg           = game_info['mob_avg'],
            forced_passes     = game_info['forced_passes'],
            curriculum_phase  = phase_name,
            opponent_name     = opp_name,
            lr                = lr,
            epsilon           = epsilon,
            improvement_bonus = game_info['improvement'],
        )

        if (ep + 1) % LOG_INTERVAL == 0:
            print(metrics.console_summary(total_eps))
            sys.stdout.flush()

        if (ep + 1) % CHECKPOINT_INTERVAL == 0:
            cp_path = os.path.join(model_dir, f'checkpoint_ep{ep+1:06d}.npy')
            nn.save(cp_path)
            _cleanup_checkpoints(model_dir, keep=3)
            print(f'  [checkpoint] {cp_path}')
            sys.stdout.flush()

    nn.save(final_path)
    print(f'\n  DONE: Finalni model: {final_path}')
    print(f'  {metrics.total_summary}')
    sys.stdout.flush()
    metrics.close()
    return nn


def _cleanup_checkpoints(model_dir: str, keep: int = 3):
    cps = sorted([f for f in os.listdir(model_dir)
                  if f.startswith('checkpoint_ep') and f.endswith('.npy')])
    for old in cps[:-keep]:
        os.remove(os.path.join(model_dir, old))


# ======================================================================
# LIVE DASHBOARD
# ======================================================================

# ANSI kody (Windows 10+ CMD/WT, Linux, macOS)
_A_HOME  = '\033[H'
_A_CLR   = '\033[2J'
_A_CEOL  = '\033[K'
_A_CEOS  = '\033[J'
_A_BOLD  = '\033[1m'
_A_DIM   = '\033[2m'
_A_RST   = '\033[0m'
_A_GREEN = '\033[92m'
_A_YEL   = '\033[93m'
_A_CYAN  = '\033[96m'
_A_RED   = '\033[91m'
_A_WHITE = '\033[97m'

_SIZE_CLR = {6: '\033[96m', 8: '\033[93m', 10: '\033[92m'}


def _strip_ansi(s: str) -> str:
    return re.sub(r'\033\[[0-9;]*m', '', s)


def _ansi_enabled() -> bool:
    if not sys.stdout.isatty():
        return False
    if sys.platform == 'win32':
        try:
            import ctypes
            k32  = ctypes.windll.kernel32
            hdl  = k32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if k32.GetConsoleMode(hdl, ctypes.byref(mode)):
                k32.SetConsoleMode(hdl, mode.value | 0x0004)
            return True
        except Exception:
            return False
    return os.environ.get('TERM', '') not in ('dumb', '')


class LiveDashboard:
    """
    Multi-panel live dashboard pro paralelni trenink.

    Kazda deska ma vlastni panel. Dashboard cte log soubory
    a prekresluje terminal kazdych REFRESH_INTERVAL sekund.
    """

    REFRESH   = 3.0     # sekundy
    LOG_LINES = 8       # pocet radku logu na panel

    def __init__(self, sizes: list, eps_per_size: dict, log_dir: str):
        self.sizes        = sizes
        self.eps_per_size = eps_per_size
        self.log_dir      = log_dir
        self.log_paths    = {s: os.path.join(log_dir, f'train_{s}x{s}.log') for s in sizes}
        self.t0           = time.time()
        self._stop        = threading.Event()
        self._ansi        = _ansi_enabled()
        self._thread      = None

    # ------------------------------------------------------------------
    def _elapsed(self) -> str:
        s = int(time.time() - self.t0)
        h, rem = divmod(s, 3600)
        m, s   = divmod(rem, 60)
        return f'{h:02d}:{m:02d}:{s:02d}'

    def _tail(self, path: str, n: int) -> list:
        if not os.path.isfile(path):
            return ['  (ceka na start...)']
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                data = f.read()
            lines = [l for l in data.splitlines() if l.strip()]
            return lines[-n:] if lines else ['  (zatim prazdny log)']
        except Exception as e:
            return [f'  (chyba: {e})']

    def _parse(self, lines: list, size: int) -> dict:
        info = {
            'pct': 0.0, 'ep': 0, 'total': self.eps_per_size[size],
            'wr': 0.0, 'dr': 0.0, 'loss': 0.0, 'margin': 0.0,
            'phase': 'start', 'done': False,
        }
        for line in reversed(lines):
            stripped = _strip_ansi(line)
            # Progress: [XX.X%] Ep NNNN/TOTAL | W:XX.X% D:X.X% | Loss:X.XXXX
            if stripped.lstrip().startswith('[') and 'Ep ' in stripped and 'W:' in stripped:
                try:
                    pct = float(stripped.split(']')[0].split('[')[1].rstrip('%'))
                    info['pct'] = pct
                    ep_str = stripped.split('Ep ')[1].split('|')[0].strip()
                    ep, tot = ep_str.split('/')
                    info['ep']    = int(ep.strip())
                    info['total'] = int(tot.strip())
                    info['wr']    = float(stripped.split('W:')[1].split('%')[0])
                    if 'D:' in stripped:
                        info['dr'] = float(stripped.split('D:')[1].split('%')[0])
                    if 'Loss:' in stripped:
                        info['loss'] = float(stripped.split('Loss:')[1].split('|')[0].strip())
                    if 'Margin:' in stripped:
                        info['margin'] = float(stripped.split('Margin:')[1].split('|')[0].strip())
                except Exception:
                    pass
                break
        for line in reversed(lines):
            stripped = _strip_ansi(line)
            if 'Curriculum:' in stripped and '=>' in stripped.replace('->', '=>'):
                try:
                    arrow = '->' if '->' in stripped else '=>'
                    after = stripped.split(arrow)[1].split('[')[0].strip()
                    info['phase'] = after
                except Exception:
                    pass
                break
            if 'Curriculum:' in stripped and chr(8594) in stripped:  # unicode arrow
                try:
                    after = stripped.split(chr(8594))[1].split('[')[0].strip()
                    info['phase'] = after
                except Exception:
                    pass
                break
        # Kontrola dokonceni
        for line in reversed(lines[-5:]):
            if 'DONE:' in _strip_ansi(line) or 'Finalni model' in _strip_ansi(line):
                info['done'] = True
                info['pct']  = 100.0
                break
        return info

    def _bar(self, pct: float, w: int = 20) -> str:
        filled = int(w * min(pct, 100.0) / 100.0)
        return '|' * filled + '.' * (w - filled)

    # ------------------------------------------------------------------
    def _build_panels(self, cols: int) -> dict:
        """Vraci dict {size: [radky_panelu_jako_str]}"""
        n     = len(self.sizes)
        # Sirka jednoho panelu: rovnomerne, odecteme oddelovace
        p_w   = max(22, (cols - (n - 1)) // n)
        panels = {}

        for s in self.sizes:
            clr   = _SIZE_CLR.get(s, _A_WHITE)
            lines = self._tail(self.log_paths[s], self.LOG_LINES + 40)
            info  = self._parse(lines, s)
            log_l = self._tail(self.log_paths[s], self.LOG_LINES)
            pw    = p_w
            plines = []

            # Titulek
            status_sym = (f'{_A_GREEN}DONE{_A_RST}' if info['done']
                          else f'{clr}RUN {_A_RST}')
            plines.append(f' {clr}{_A_BOLD}[ {s}x{s} ]{_A_RST} {status_sym}')

            # Progress bar
            bar = self._bar(info['pct'], w=min(20, pw - 12))
            w_clr = (_A_GREEN if info['wr'] >= 60 else
                     _A_YEL   if info['wr'] >= 45 else _A_RED)
            plines.append(f' {clr}{bar}{_A_RST} {info["pct"]:5.1f}%')

            # Ep
            plines.append(f' Ep: {info["ep"]:>7}/{info["total"]}')

            # Win rate
            lr_ = max(0.0, 100.0 - info['wr'] - info['dr'])
            plines.append(f' W:{w_clr}{info["wr"]:5.1f}%{_A_RST}'
                          f' D:{info["dr"]:4.1f}%'
                          f' L:{lr_:5.1f}%')
            plines.append(f' Loss:   {info["loss"]:.5f}')
            plines.append(f' Margin: {info["margin"]:+.2f}')

            # Faze curricula
            ph = info['phase']
            max_ph = pw - 9
            if len(ph) > max_ph:
                ph = ph[:max_ph - 2] + '..'
            plines.append(f' Faze: {_A_YEL}{ph}{_A_RST}')

            # Oddelovac
            plines.append(f' {_A_DIM}' + chr(183) * max(0, pw - 3) + _A_RST)

            # Radky logu (oriznuty)
            for ll in log_l:
                plain = _strip_ansi(ll)
                if len(plain) > pw - 2:
                    plain = plain[:pw - 5] + '...'
                plines.append(' ' + plain)

            panels[s] = plines

        return panels, p_w

    # ------------------------------------------------------------------
    def _render(self):
        cols, rows = shutil.get_terminal_size(fallback=(120, 40))
        panels, p_w = self._build_panels(cols)
        buf = [_A_HOME]

        # Horni lista
        cpu    = os.cpu_count() or '?'
        desky  = '  '.join(f'{s}x{s}' for s in self.sizes)
        header = (f' {_A_BOLD}{_A_CYAN}REVERSI RL{_A_RST}'
                  f' | Paralelni trenink'
                  f' | CPU:{cpu}'
                  f' | {_A_CYAN}{desky}{_A_RST}'
                  f' | Cas:{self._elapsed()} ')
        plain_h = _strip_ansi(header)
        pad_h   = max(0, cols - len(plain_h) - 1)
        buf.append(header + ' ' * pad_h + _A_CEOL)

        sep = _A_DIM + '=' * cols + _A_RST
        buf.append(sep + _A_CEOL)

        # Panely bok po boku
        max_rows = max(len(panels[s]) for s in self.sizes)
        div      = _A_DIM + '|' + _A_RST

        for ri in range(max_rows):
            if ri >= rows - 4:
                break
            parts = []
            for s in self.sizes:
                plines = panels[s]
                cell   = plines[ri] if ri < len(plines) else ''
                plain  = _strip_ansi(cell)
                pad    = max(0, p_w - len(plain))
                parts.append(cell + ' ' * pad)
            row = div.join(parts)
            buf.append(row + _A_CEOL)

        # Spodni lista
        buf.append(sep + _A_CEOL)
        footer = (f' {_A_DIM}Logy: {self.log_dir}/train_NxN.log'
                  f'  |  Ctrl+C pro ukonceni'
                  f'  |  Refresh: {self.REFRESH:.0f}s{_A_RST}')
        buf.append(footer + _A_CEOL)
        buf.append(_A_CEOS)

        sys.stdout.write('\n'.join(buf))
        sys.stdout.flush()

    def _fallback(self):
        """Jednoduchy vypis bez ANSI."""
        ts = self._elapsed()
        out = [f'\n{"─"*60}', f'  Trenink bezi | {ts}']
        for s in self.sizes:
            lines = self._tail(self.log_paths[s], 3)
            out.append(f'  [{s}x{s}]')
            for l in lines:
                out.append(f'    {_strip_ansi(l)}')
        out.append('─' * 60)
        print('\n'.join(out))

    def _loop(self):
        if self._ansi:
            sys.stdout.write(_A_CLR)
            sys.stdout.flush()
        while not self._stop.is_set():
            try:
                if self._ansi:
                    self._render()
                else:
                    self._fallback()
            except Exception:
                pass
            self._stop.wait(timeout=self.REFRESH)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name='dashboard')
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._ansi:
            cols, rows = shutil.get_terminal_size(fallback=(120, 40))
            sys.stdout.write(f'\033[{rows};0H\n')
            sys.stdout.flush()

    def print_final(self, done_list: list):
        print(f'\n{"="*72}')
        print(f'  TRENINK DOKONCEN  |  Celkovy cas: {self._elapsed()}')
        print(f'{"="*72}')
        for size, ec in done_list:
            status = 'OK' if ec == 0 else f'CHYBA (exit {ec})'
            lp     = self.log_paths.get(size, '?')
            clr    = _SIZE_CLR.get(size, '')
            print(f'\n  {clr}{_A_BOLD}[ {size}x{size} ]{_A_RST}  ->  {status}'
                  f'  |  Log: {lp}')
            for l in self._tail(lp, 4):
                print(f'      {_strip_ansi(l)}')
        print()


# ======================================================================
# WORKER SUBPROCESS
# ======================================================================

def _train_size_worker(kwargs: dict):
    """
    Top-level wrapper pro multiprocessing.Process.

    Presmeruje stdout/stderr do log souboru.
    Dashboard v hlavnim procesu cte tyto soubory a zobrazuje je live.
    """
    root     = kwargs.pop('_root')
    log_path = kwargs.pop('log_path')

    import sys as _sys, os as _os, warnings
    if root not in _sys.path:
        _sys.path.insert(0, root)

    import random as _rand, numpy as _np
    warnings.filterwarnings('ignore')
    _np.random.seed(kwargs['seed'])
    _rand.seed(kwargs['seed'])

    _os.makedirs(_os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, 'w', buffering=1, encoding='utf-8')

    class _Tee:
        def __init__(self, f): self._f = f
        def write(self, s):
            self._f.write(s)
            self._f.flush()
        def flush(self):    self._f.flush()
        def isatty(self):   return False

    _sys.stdout = _Tee(log_file)
    _sys.stderr = _Tee(log_file)

    try:
        from train import train_size
        train_size(
            size      = kwargs['size'],
            total_eps = kwargs['total_eps'],
            resume    = kwargs['resume'],
            adaptive  = kwargs['adaptive'],
        )
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        log_file.close()


# ======================================================================
# PARALELNI TRENINK S DASHBOARDEM
# ======================================================================

def train_parallel(sizes: list, eps_per_size: dict,
                   resume: bool, adaptive: bool, seed: int):
    cpu = os.cpu_count() or 2

    if cpu <= 2:
        max_procs = 1
    elif cpu <= 4:
        max_procs = min(len(sizes), cpu - 1)
    else:
        max_procs = min(len(sizes), max(2, cpu // 2))

    os.makedirs(LOG_DIR, exist_ok=True)

    log_paths = {s: os.path.join(LOG_DIR, f'train_{s}x{s}.log') for s in sizes}
    for lp in log_paths.values():
        if os.path.isfile(lp):
            os.remove(lp)

    if max_procs <= 1 or len(sizes) == 1:
        print(f'\n  [Train] Sekvencni trenink ({cpu} CPU jader, {len(sizes)} deska/y)\n')
        for size in sizes:
            np.random.seed(seed)
            random.seed(seed)
            train_size(size, eps_per_size[size], resume=resume, adaptive=adaptive)
        return

    dashboard = LiveDashboard(sizes, eps_per_size, LOG_DIR)
    dashboard.start()

    ctx   = mp.get_context('spawn')
    procs = []
    done  = []

    try:
        for i, size in enumerate(sizes):
            kwargs = {
                '_root':    _ROOT,
                'size':     size,
                'total_eps': eps_per_size[size],
                'resume':   resume,
                'adaptive': adaptive,
                'seed':     seed + i,
                'log_path': log_paths[size],
            }
            p = ctx.Process(
                target = _train_size_worker,
                args   = (kwargs,),
                name   = f'train_{size}x{size}',
                daemon = False,
            )
            procs.append((size, p))

            alive = [pp for _, pp in procs if pp.is_alive()]
            while len(alive) >= max_procs:
                time.sleep(1.0)
                alive = [pp for _, pp in procs if pp.is_alive()]

            p.start()

        for size, p in procs:
            p.join()
            done.append((size, p.exitcode))

    except KeyboardInterrupt:
        for _, p in procs:
            if p.is_alive():
                p.terminate()
        for _, p in procs:
            p.join(timeout=5)
        done = [(s, -1) for s, _ in procs]

    finally:
        dashboard.stop()

    dashboard.print_final(done)


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Reversi RL Trainer v2.1 – live dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Priklady:
  python train.py                    # 6x6, 8x8, 10x10 paralelne + dashboard
  python train.py --size 8           # jen 8x8
  python train.py --fast             # 8%% epizod (test)
  python train.py --size 8 --eps 5000
  python train.py --resume
  python train.py --no-adaptive
  python train.py --sequential       # zakaze paralelismus + dashboard

Logy:
  logs/train_6x6.log    kompletni vypis procesu 6x6
  logs/train_8x8.log    kompletni vypis procesu 8x8
  logs/train_10x10.log  kompletni vypis procesu 10x10
        """)
    parser.add_argument('--size',        type=int, nargs='+', choices=[6, 8, 10],
                        default=[6, 8, 10])
    parser.add_argument('--fast',        action='store_true')
    parser.add_argument('--eps',         type=int, default=None)
    parser.add_argument('--resume',      action='store_true')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--no-adaptive', dest='adaptive', action='store_false')
    parser.add_argument('--sequential',  action='store_true',
                        help='Zakaze paralelni trenink a dashboard')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    sizes = sorted(set(args.size))

    print(f'\n{"="*72}')
    print(f'  Reversi RL Trainer v2.1  |  NumPy only  |  C depth-only mode')
    print(f'  Desky: {sizes}  |  Seed: {args.seed}  |  CPU jader: {os.cpu_count()}')
    print(f'{"="*72}')

    eps_per_size = {}
    for size in sizes:
        e = args.eps if args.eps else TOTAL_EPISODES[size]
        if args.fast:
            e = max(80, int(e * FAST_MULTIPLIER))
        eps_per_size[size] = e

    t0 = time.time()

    if args.sequential or len(sizes) == 1:
        print(f'\n  Rezim: sekvencni\n')
        for size in sizes:
            train_size(size, eps_per_size[size],
                       resume=args.resume, adaptive=args.adaptive)
    else:
        print(f'\n  Rezim: paralelni + live dashboard\n')
        train_parallel(
            sizes        = sizes,
            eps_per_size = eps_per_size,
            resume       = args.resume,
            adaptive     = args.adaptive,
            seed         = args.seed,
        )

    elapsed = time.time() - t0
    print(f'{"="*72}')
    print(f'  Celkovy cas: {elapsed/60:.1f} min  |  Modely: {MODEL_DIR}/')
    print(f'{"="*72}\n')


if __name__ == '__main__':
    mp.freeze_support()
    main()
