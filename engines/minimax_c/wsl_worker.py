"""
engines/minimax_c/wsl_worker.py  –  Worker process beziici uvnitr WSL.

Tento skript se spousti UVNITR WSL a komunikuje s Windows Pythonem
pres stdin/stdout. Nesmi obsahovat non-ASCII znaky v komentarich
(pro kompatibilitu s heredoc volanim z Windows).

Protokol (JSON lines, jedna zprava per radek):
  Windows -> WSL:
    {"cmd": "move",  "board": [flat int8 list], "size": N, "color": C,
                     "max_depth": D, "time_ms": T}
    {"cmd": "clear_tt"}
    {"cmd": "quit"}

  WSL -> Windows:
    {"status": "ok",   "row": R, "col": C}
    {"status": "none"}                        
    {"status": "error", "msg": "..."}
    {"status": "ready"}                      
"""

import sys
import os
import json
import ctypes

_DIR = os.path.dirname(os.path.abspath(__file__))
_SO  = os.path.join(_DIR, 'reversi_ab.so')

# Kompilace pokud .so neexistuje
if not os.path.isfile(_SO):
    import subprocess
    _C = os.path.join(_DIR, 'reversi_ab.c')
    result = subprocess.run(
        ['gcc', '-O3', '-march=native', '-ffast-math',
         '-shared', '-fPIC', '-o', _SO, _C, '-lm'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.stdout.write(json.dumps({'status': 'error',
                                      'msg': result.stderr}) + '\n')
        sys.stdout.flush()
        sys.exit(1)

try:
    lib = ctypes.CDLL(_SO)
    lib.reversi_get_move.restype  = ctypes.c_int
    lib.reversi_get_move.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.clear_tt.restype  = None
    lib.clear_tt.argtypes = []
except Exception as e:
    sys.stdout.write(json.dumps({'status': 'error', 'msg': str(e)}) + '\n')
    sys.stdout.flush()
    sys.exit(1)

# Signal ready
sys.stdout.write(json.dumps({'status': 'ready'}) + '\n')
sys.stdout.flush()

# Main loop
for raw_line in sys.stdin:
    raw_line = raw_line.strip()
    if not raw_line:
        continue
    try:
        msg = json.loads(raw_line)
        cmd = msg.get('cmd', '')

        if cmd == 'quit':
            break

        elif cmd == 'clear_tt':
            lib.clear_tt()
            sys.stdout.write(json.dumps({'status': 'ok'}) + '\n')

        elif cmd == 'move':
            board_list = msg['board']
            size       = int(msg['size'])
            color      = int(msg['color'])
            max_depth  = int(msg.get('max_depth', 18))
            time_ms    = int(msg.get('time_ms',   100))

            import numpy as np
            flat = (ctypes.c_int8 * len(board_list))(*board_list)
            ptr  = ctypes.cast(flat, ctypes.POINTER(ctypes.c_int8))
            result = lib.reversi_get_move(ptr, size, color, max_depth, time_ms)

            if result < 0:
                sys.stdout.write(json.dumps({'status': 'none'}) + '\n')
            else:
                row, col = divmod(result, 100)
                sys.stdout.write(json.dumps({'status': 'ok',
                                              'row': row, 'col': col}) + '\n')
        else:
            sys.stdout.write(json.dumps({'status': 'error',
                                          'msg': f'unknown cmd: {cmd}'}) + '\n')

    except Exception as e:
        sys.stdout.write(json.dumps({'status': 'error', 'msg': str(e)}) + '\n')

    sys.stdout.flush()
