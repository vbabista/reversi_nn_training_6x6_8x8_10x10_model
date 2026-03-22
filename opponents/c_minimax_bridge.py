"""
opponents/c_minimax_bridge.py  --  Bridge pro C alfa-beta minimax engine.

PLATFORM SUPPORT:
  Linux / WSL (Python bezi uvnitr WSL):
    Pouziva ctypes primo -> nejrychlejsi, zadna overhead.

  Windows (Python bezi nativne na Windows, WSL nainstalovan):
    Pouziva WslBridge: spusti persistentni WSL subprocess s wsl_worker.py,
    komunikuje pres stdin/stdout (JSON lines).
    Pozadavky: wsl.exe v PATH, WSL distro s gcc a python3 nainstalovan.

  Fallback (zadny C engine, zadny WSL):
    Automaticky pouzije Python MinimaxOpponent (max depth 5).

Detekce platformy:
  - sys.platform == 'win32'    -> Windows
  - /proc/version obsahuje 'microsoft' -> WSL
  - jinak                      -> Linux nativni

Pouziti (stejne API jako MinimaxOpponent):
  opp = CMinimax(max_depth=18, time_limit_ms=100, board_size=8)
  move = opp.get_move(board, color)  # numpy int8, color=0/1
"""

import os
import sys
import ctypes
import subprocess
import json
import threading
import queue
import numpy as np
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────
# Detekce platformy
# ─────────────────────────────────────────────────────────────────────

def _detect_platform() -> str:
    if sys.platform == 'win32':
        return 'windows'
    try:
        proc = open('/proc/version').read().lower()
        if 'microsoft' in proc or 'wsl' in proc:
            return 'wsl'
    except OSError:
        pass
    return 'linux'

PLATFORM = _detect_platform()

# ─────────────────────────────────────────────────────────────────────
# Cesty
# ─────────────────────────────────────────────────────────────────────

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_ENGINE_DIR = os.path.normpath(os.path.join(_THIS_DIR, '..', 'engines', 'minimax_c'))
_SO_PATH    = os.path.join(_ENGINE_DIR, 'reversi_ab.so')
_C_PATH     = os.path.join(_ENGINE_DIR, 'reversi_ab.c')
_WORKER_PATH = os.path.join(_ENGINE_DIR, 'wsl_worker.py')

# ─────────────────────────────────────────────────────────────────────
# Kompilace (Linux / WSL)
# ─────────────────────────────────────────────────────────────────────

def _compile_linux() -> bool:
    if not os.path.isfile(_C_PATH):
        return False
    if (os.path.isfile(_SO_PATH) and
            os.path.getmtime(_SO_PATH) >= os.path.getmtime(_C_PATH)):
        return True
    print('[CMinimax] Kompilace C enginu...')
    try:
        r = subprocess.run(
            ['gcc', '-O3', '-march=native', '-ffast-math',
             '-shared', '-fPIC', '-o', _SO_PATH, _C_PATH, '-lm'],
            capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            print(f'[CMinimax] Kompilace OK ({_SO_PATH})')
            return True
        print(f'[CMinimax] Chyba kompilace:\n{r.stderr[:500]}')
    except Exception as e:
        print(f'[CMinimax] gcc selhal: {e}')
    return False

# ─────────────────────────────────────────────────────────────────────
# CTYPES BACKEND (Linux / WSL)
# ─────────────────────────────────────────────────────────────────────

_lib = None
_lib_ok = False

def _load_ctypes() -> bool:
    global _lib, _lib_ok
    if _lib_ok:
        return True
    if not _compile_linux():
        return False
    try:
        _lib = ctypes.CDLL(_SO_PATH)
        _lib.reversi_get_move.restype  = ctypes.c_int
        _lib.reversi_get_move.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        _lib.clear_tt.restype  = None
        _lib.clear_tt.argtypes = []
        _lib_ok = True
        return True
    except Exception as e:
        print(f'[CMinimax] ctypes load selhal: {e}')
        return False


class _CtypesBackend:

    def get_move(self, board: np.ndarray, size: int, color: int,
                  max_depth: int, time_ms: int) -> Optional[Tuple[int,int]]:
        flat = np.ascontiguousarray(board.flatten(), dtype=np.int8)
        ptr  = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        res  = _lib.reversi_get_move(ptr, size, color, max_depth, time_ms)
        if res < 0:
            return None
        return divmod(res, 100)

    def clear_tt(self):
        if _lib_ok:
            _lib.clear_tt()

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────
# WSL SUBPROCESS BACKEND (Windows native Python)
# ─────────────────────────────────────────────────────────────────────

def _wsl_path(windows_path: str) -> str:

    p = windows_path.replace('\\', '/')
    if len(p) >= 2 and p[1] == ':':
        drive = p[0].lower()
        rest  = p[2:]
        return f'/mnt/{drive}{rest}'
    return p


def _check_wsl_available() -> bool:
    try:
        r = subprocess.run(['wsl', '--status'],
                            capture_output=True, text=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class _WslBackend:

    def __init__(self):
        self._proc   = None
        self._lock   = threading.Lock()
        self._ok     = False
        self._start()

    def _start(self):

        # Prevod Windows cesty na WSL cestu
        worker_wsl = _wsl_path(_WORKER_PATH)

        try:
            # Spustit wsl_worker.py uvnitr WSL
            self._proc = subprocess.Popen(
                ['wsl', 'python3', worker_wsl],
                stdin  = subprocess.PIPE,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                text   = True,
                bufsize = 1,  # line-buffered
            )
            # Pockat na "ready" signal
            ready_line = self._proc.stdout.readline().strip()
            msg = json.loads(ready_line)
            if msg.get('status') == 'ready':
                self._ok = True
                print(f'[CMinimax] WSL backend ready (PID {self._proc.pid})')
            else:
                print(f'[CMinimax] WSL backend chyba: {msg}')
        except Exception as e:
            print(f'[CMinimax] Nelze spustit WSL worker: {e}')
            self._ok = False

    def _send(self, msg: dict) -> dict:

        if not self._ok or self._proc is None:
            return {'status': 'error', 'msg': 'worker not running'}
        try:
            line = json.dumps(msg) + '\n'
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
            resp_line = self._proc.stdout.readline().strip()
            return json.loads(resp_line)
        except Exception as e:
            self._ok = False
            return {'status': 'error', 'msg': str(e)}

    def get_move(self, board: np.ndarray, size: int, color: int,
                  max_depth: int, time_ms: int) -> Optional[Tuple[int,int]]:
        with self._lock:
            board_list = board.flatten().tolist()
            msg = {
                'cmd': 'move',
                'board': board_list,
                'size':  size,
                'color': color,
                'max_depth': max_depth,
                'time_ms':   time_ms,
            }
            resp = self._send(msg)

        if resp.get('status') == 'ok':
            return (int(resp['row']), int(resp['col']))
        elif resp.get('status') == 'none':
            return None
        else:
            # Error fallback: vrat prvni platny tah
            from core.board import get_valid_moves
            moves = get_valid_moves(board, color)
            return moves[0] if moves else None

    def clear_tt(self):
        with self._lock:
            self._send({'cmd': 'clear_tt'})

    def close(self):
        if self._proc and self._ok:
            try:
                self._send({'cmd': 'quit'})
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._ok = False

    def __del__(self):
        self.close()


# ─────────────────────────────────────────────────────────────────────
# Vybrani backendu + singleton
# ─────────────────────────────────────────────────────────────────────

_backend = None
_backend_type = 'none'   # 'ctypes', 'wsl', 'none'

def _init_backend():
    global _backend, _backend_type

    if PLATFORM in ('linux', 'wsl'):
        # Pokus o ctypes
        if _load_ctypes():
            _backend = _CtypesBackend()
            _backend_type = 'ctypes'
            return
        print('[CMinimax] ctypes selhal, zkousim Python fallback')

    elif PLATFORM == 'windows':
        # Pokus o WSL backend
        if _check_wsl_available():
            try:
                b = _WslBackend()
                if b._ok:
                    _backend = b
                    _backend_type = 'wsl'
                    return
            except Exception as e:
                print(f'[CMinimax] WSL backend selhal: {e}')
        else:
            print('[CMinimax] wsl.exe nenalezen nebo WSL neni dostupny')
        print('[CMinimax] Pouzivam Python minimax fallback')

    _backend_type = 'none'


_init_backend()
C_ENGINE_AVAILABLE = _backend_type in ('ctypes', 'wsl')


# ─────────────────────────────────────────────────────────────────────
# Verejne API: CMinimax
# ─────────────────────────────────────────────────────────────────────

class CMinimax:
    """
    C alfa-beta minimax oponent.
    REZIMY:
      target_depth > 0  ->  depth-only mode (doporucene pro trenink).
                            Engine prohledava presne do target_depth, BEZ casoveho limitu.
                            Sila je konzistentni na vsech pocitacich.
                            Aktivovano z curriculum pomoci {'target_depth': N}.

      target_depth <= 0 ->  time-limited mode (puvodni chovani).
                            Engine pouziva iterative deepening s casovym limitem.
                            Sila zavisi na rychlosti HW (NEDOPORUCENO pro trenink).

    API:
        opp = CMinimax(target_depth=8, board_size=8)       # depth-only (doporuceno)
        opp = CMinimax(max_depth=18, time_limit_ms=1000, board_size=8)  # time-limited
        move = opp.get_move(board, color)  # color=0/1
    """

    def __init__(self, max_depth: int = 18, time_limit_ms: int = 1000,
                 board_size: int = 8, target_depth: int = 0):
        self.board_size = board_size
        self._fallback  = None

        # FIX: Depth-only mode when target_depth > 0
        if target_depth > 0:
            # time_limit_ms=-1 activuje depth_only_mode v C enginu
            self.max_depth     = target_depth
            self.time_limit_ms = -1
            self._mode         = f'depth-only(d={target_depth})'
        else:
            self.max_depth     = max_depth
            self.time_limit_ms = time_limit_ms
            self._mode         = f'time-limited(t={time_limit_ms}ms)'

        if not C_ENGINE_AVAILABLE:
            from opponents.minimax_opponent import MinimaxOpponent
            fb_depth = min(self.max_depth, 5)
            self._fallback = MinimaxOpponent(fb_depth, board_size)

    def get_move(self, board: np.ndarray,
                  color: int) -> Optional[Tuple[int,int]]:
        if self._fallback is not None:
            return self._fallback.get_move(board, color)

        move = _backend.get_move(
            board, self.board_size, color,
            self.max_depth, self.time_limit_ms
        )

        if move is None:
            return None

        # Sanity check: validni souradnice
        r, c = move
        if not (0 <= r < self.board_size and 0 <= c < self.board_size):
            from core.board import get_valid_moves
            moves = get_valid_moves(board, color)
            return moves[0] if moves else None

        return (r, c)

    def clear_tt(self):
        if _backend is not None:
            _backend.clear_tt()

    @property
    def backend_type(self) -> str:
        return _backend_type

    def __repr__(self) -> str:
        bt = _backend_type if not self._fallback else 'py-fallback'
        return f'CMinimax({bt}, {self._mode}, size={self.board_size})'


def make_c_minimax(level: str, board_size: int) -> 'CMinimax':

    # target_depth referencni pro 8x8
    depth_configs = {
        'fast':    6,
        'medium':  8,
        'strong':  9,
        'vstrong': 10,
        'elite':   12,
        'max':     13,
    }
    td = depth_configs.get(level, 10)
    return CMinimax(target_depth=td, board_size=board_size)
