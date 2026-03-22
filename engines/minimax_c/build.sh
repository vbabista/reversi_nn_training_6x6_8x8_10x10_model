#!/usr/bin/env bash
# build.sh  –  Kompilace C minimax enginu pro Reversi
#
# Použití:
#   chmod +x build.sh && ./build.sh
#   nebo automaticky voláno z opponents/c_minimax_bridge.py

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/reversi_ab.c"
OUT="$DIR/reversi_ab.so"

echo "[build] Kompilace C minimax enginu..."
echo "  Zdroj:  $SRC"
echo "  Výstup: $OUT"

# Detekce platformy
OS="$(uname -s)"

if [ "$OS" = "Darwin" ]; then
    # macOS
    gcc -O3 -march=native -ffast-math \
        -shared -fPIC \
        -o "$OUT" "$SRC" \
        -lm
    echo "[build] macOS: použit clang/gcc"
else
    # Linux (Ubuntu, Debian, atd.)
    gcc -O3 -march=native -ffast-math \
        -shared -fPIC \
        -o "$OUT" "$SRC" \
        -lm
    echo "[build] Linux: použit gcc"
fi

echo "[build] Hotovo: $OUT"
echo "[build] Velikost: $(du -h "$OUT" | cut -f1)"

# Rychlý test
python3 -c "
import ctypes, os, sys
lib = ctypes.CDLL('$OUT')
print('[build] ctypes load OK')

# Test: inicializuj board 8x8
import numpy as np
# Black=0, White=1, Empty=-1
b = np.full((8,8), -1, dtype=np.int8)
b[3,3]=1; b[3,4]=0; b[4,3]=0; b[4,4]=1  # standard start
flat = b.flatten()

lib.reversi_get_move.restype  = ctypes.c_int
lib.reversi_get_move.argtypes = [
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
]
result = lib.reversi_get_move(
    flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    8, 0, 6, 1000  # depth=6, 1000ms
)
row, col = divmod(result, 100)
print(f'[build] Test tah (BLACK, d=6): ({row},{col}) ✓')
" || echo "[build] Python test selhal (ctypes test)"
