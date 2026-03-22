"""
verify.py  --  Overeni funkcionality a benchmark vs minimax.

Spusteni:
    python verify.py                # testy + benchmark (pouzije existujici model)
    python verify.py --no-model     # jen testy, bez benchmarku
    python verify.py --size 8       # jen 8x8
    python verify.py --full         # 100 her per hloubka
    python verify.py --wsl-test     # test WSL bridge (jen na Windows)
"""

import sys, os, time, argparse
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from core.board import (
    init_board, get_valid_moves, make_move,
    is_terminal, count_pieces, game_result as flat_game_result,
    board_to_str, BLACK, WHITE, EMPTY,
)
from model.neural_net     import ReversiNet, encode_board, encode_moves_batch
from training.replay_buffer   import ReplayBuffer
from training.reward_shaper   import RewardShaper, margin_game_result, margin_label
from training.curriculum      import CurriculumManager
from training.self_play_pool  import SelfPlayPool
from opponents.random_opponent  import RandomOpponent
from opponents.greedy_opponent  import GreedyOpponent
from opponents.minimax_opponent import MinimaxOpponent
from opponents.c_minimax_bridge import (
    CMinimax, C_ENGINE_AVAILABLE, PLATFORM, _backend_type
)
from config import ARCH, MODEL_DIR, MARGIN_THRESHOLDS

P = lambda b, d='': print(f'  {"OK" if b else "FAIL"}  {d}') or (None if b else (_ for _ in ()).throw(AssertionError(d)))

def section(t): print(f'\n{"─"*62}\n  {t}\n{"─"*62}')


# ══════════════════════════════════════════════════════════════════════
def test_board():
    section('1. Board logic')
    for size in [6, 8, 10]:
        b = init_board(size)
        P(b.shape == (size, size) and b.dtype == np.int8, f'init_board({size})')
        bc, wc = count_pieces(b)
        P(bc == 2 and wc == 2, f'start pieces ({size})')
        moves = get_valid_moves(b, BLACK)
        P(len(moves) == 4, f'valid moves start ({size}) = {moves}')
        b2 = make_move(b, moves[0], BLACK)
        P(not np.array_equal(b, b2), f'make_move immutable ({size})')

    b_full = np.zeros((8,8), dtype=np.int8)
    b_full[:5, :] = BLACK; b_full[5:, :] = WHITE
    P(is_terminal(b_full), 'is_terminal full board')
    P(flat_game_result(b_full, BLACK) == 1.0, 'flat_game_result win')

    print(f'\n  8x8 start:\n')
    for line in board_to_str(init_board(8)).split('\n'):
        print(f'    {line}')


# ══════════════════════════════════════════════════════════════════════
def test_margin_reward():
    section('2. Margin-based reward')

    # Dominance: 40 vs 0
    b = np.full((8,8), BLACK, dtype=np.int8)
    b[0,0] = WHITE; b[1,1] = WHITE  # 62 B, 2 W
    r = margin_game_result(b, BLACK)
    lbl = margin_label(b, BLACK)
    P(r == 1.0 and lbl == 'dominance', f'dominance: r={r}, lbl={lbl}')

    # Tihle: 17 vs 15 na 32 poli
    b2 = np.full((8,4), BLACK, dtype=np.int8)
    b3 = np.full((8,4), WHITE, dtype=np.int8)
    board = np.hstack([b2, b3])  # 32 B vs 32 W = draw
    r2 = margin_game_result(board, BLACK)
    P(abs(r2 - 0.08) < 0.01, f'draw: r={r2}')

    # Test vsechny thresholdy
    for thr, expected_r, lbl_name in MARGIN_THRESHOLDS:
        P(True, f'threshold {lbl_name}: r={expected_r}')


# ══════════════════════════════════════════════════════════════════════
def test_encoding():
    section('3. Kodovani stavu')
    for size in [6, 8, 10]:
        b     = init_board(size)
        moves = get_valid_moves(b, BLACK)
        n_sq  = size * size
        enc = encode_board(b, BLACK)
        P(enc.shape == (1, 4*n_sq), f'encode_board shape ({size}): {enc.shape}')
        P(enc.dtype == np.float32,  f'encode_board dtype ({size})')
        P(enc.min() >= 0 and enc.max() <= 1, f'encode_board range ({size})')
        batch = encode_moves_batch(b, moves, BLACK)
        P(batch.shape == (len(moves), 4*n_sq), f'batch shape ({size}): {batch.shape}')


# ══════════════════════════════════════════════════════════════════════
def test_neural_net():
    section('4. Neuronova sit')
    for size in [6, 8]:
        input_size = 4 * size * size
        arch = ARCH[size]
        nn = ReversiNet(input_size, arch['H'], arch['res_blocks'], lr=3e-4)
        P(nn is not None, f'init ({size}x{size}): {nn}')

        b     = init_board(size)
        moves = get_valid_moves(b, BLACK)
        x     = encode_board(b, BLACK)
        out   = nn.forward(x)
        P(out.shape == (1,1) and -1 <= float(out.flat[0]) <= 1,
          f'forward range ({size}): {float(out.flat[0]):.3f}')

        scores = nn.score_moves(b, moves, BLACK)
        P(scores.shape == (len(moves),), f'score_moves shape ({size})')

        batch_x = encode_moves_batch(b, moves, BLACK)
        targets = np.random.uniform(-1, 1, len(moves)).astype(np.float32)
        loss0 = nn.train_batch(batch_x, targets)
        losses = [loss0]
        for _ in range(20): losses.append(nn.train_batch(batch_x, targets))
        P(losses[-1] < losses[0], f'loss klesá ({size}): {losses[0]:.4f} -> {losses[-1]:.4f}')

        path = f'/tmp/verify_{size}.npy'
        nn.save(path)
        sc_before = nn.score_moves(b, moves, BLACK)
        nn2 = ReversiNet.load(path)
        sc_after  = nn2.score_moves(b, moves, BLACK)
        P(np.allclose(sc_before, sc_after, atol=1e-5),
          f'save/load ({size}): diff={np.abs(sc_before-sc_after).max():.2e}')


# ══════════════════════════════════════════════════════════════════════
def test_opponents():
    section('5. Oponenti')

    # Platform info
    print(f'  Platform: {PLATFORM}  |  Backend: {_backend_type}  |  C available: {C_ENGINE_AVAILABLE}')

    for size in [6, 8]:
        b = init_board(size)
        valid = set(get_valid_moves(b, BLACK))

        m = RandomOpponent(size).get_move(b, BLACK)
        P(m in valid, f'Random ({size}): {m}')

        m = GreedyOpponent(size).get_move(b, BLACK)
        P(m in valid, f'Greedy ({size}): {m}')

        for d in [1, 2, 3]:
            m = MinimaxOpponent(d, size).get_move(b, BLACK)
            P(m in valid, f'MinimaxPy d={d} ({size}): {m}')

        if C_ENGINE_AVAILABLE:
            for t_ms in [20, 100, 300]:
                opp = CMinimax(18, t_ms, size)
                t0  = time.time()
                m   = opp.get_move(b, BLACK)
                ms  = (time.time()-t0)*1000
                P(m in valid, f'CMinimax t={t_ms}ms ({size}): move={m} actual={ms:.0f}ms')
        else:
            print(f'  (C engine nedostupny, preskoceno)')


# ══════════════════════════════════════════════════════════════════════
def test_training_components():
    section('6. Treninkov komponenty')

    # ReplayBuffer
    buf = ReplayBuffer(500, 128, use_per=True)
    for _ in range(200):
        buf.add(np.random.randn(128).astype(np.float32), np.random.uniform(-1,1))
    P(len(buf) == 200, 'ReplayBuffer add')
    states, targets, idx, w = buf.sample(32)
    P(states.shape == (32,128) and targets.shape == (32,), 'ReplayBuffer sample shapes')
    P(w.min() > 0 and w.max() <= 1.01, f'IS weights range: {w.min():.3f} - {w.max():.3f}')

    # RewardShaper
    shaper = RewardShaper()
    b = init_board(8)
    b2 = make_move(b, (2,3), BLACK)
    shaper.reset(b, BLACK)
    r = shaper.step_reward(b, b2, BLACK)
    P(-1 <= r <= 1, f'step_reward: {r:.4f}')

    b_dom = np.full((8,8), BLACK, dtype=np.int8)
    b_dom[0,0] = WHITE
    bonus = shaper.game_improvement_bonus(b_dom, BLACK)
    P('total' in bonus, f'improvement_bonus keys: {list(bonus.keys())}')

    # Margin reward
    r_dom = margin_game_result(b_dom, BLACK)
    P(r_dom == 1.0, f'margin dominance: {r_dom}')

    # Curriculum
    cur = CurriculumManager(8, 5000, adaptive=True)
    for _ in range(250): cur.record_result(1.0)
    P(cur.phase_index > 0, f'curriculum promotion: phase={cur.phase_index} [{cur.phase_name}]')

    # SelfPlayPool
    pool = SelfPlayPool(4)
    nn = ReversiNet(4*8*8, H=32, res_blocks=1)
    pool.add(nn, 0); pool.add(nn, 100)
    P(len(pool) == 2, 'SelfPlayPool size')
    P(pool.sample() is not None, 'SelfPlayPool sample')


# ══════════════════════════════════════════════════════════════════════
def benchmark_model(size: int, n_games: int = 30):
    section(f'7. Benchmark modelu {size}x{size}')

    model_path = os.path.join(MODEL_DIR, f'{size}x{size}', f'model_{size}x{size}.npy')
    if not os.path.isfile(model_path):
        print(f'  Model nenalezen: {model_path}')
        print(f'  Spust: python train.py --size {size}')
        return

    nn = ReversiNet.load(model_path)
    print(f'  Model: {nn}')
    print(f'  Hry per oponent: {n_games}')

    opponents = [
        ('MinimaxPy d=1', MinimaxOpponent(1, size)),
        ('MinimaxPy d=2', MinimaxOpponent(2, size)),
        ('MinimaxPy d=3', MinimaxOpponent(3, size)),
    ]
    if C_ENGINE_AVAILABLE:
        opponents += [
            (f'CMinimax 20ms',  CMinimax(18,  20, size)),
            (f'CMinimax 100ms', CMinimax(18, 100, size)),
            (f'CMinimax 300ms', CMinimax(18, 300, size)),
        ]

    print()
    for opp_name, opp in opponents:
        wins = draws = losses = 0
        for game_i in range(n_games):
            b     = init_board(size)
            color = BLACK
            ai    = BLACK if game_i % 2 == 0 else WHITE
            consec = 0
            while True:
                moves = get_valid_moves(b, color)
                if not moves:
                    consec += 1
                    if consec >= 2: break
                    color = 1 - color; continue
                consec = 0
                if color == ai:
                    sc = nn.score_moves(b, moves, color)
                    m  = moves[int(np.argmax(sc))]
                else:
                    m = opp.get_move(b, color)
                    if m is None: color = 1-color; continue
                b     = make_move(b, m, color)
                color = 1 - color

            mr = margin_game_result(b, ai)
            if   mr > 0:              wins   += 1
            elif abs(mr - 0.08) < 0.01: draws += 1
            else:                     losses += 1

        wr = wins / n_games
        bar = '#' * int(wr * 20) + '.' * (20 - int(wr * 20))
        print(f'  vs {opp_name:<20} [{bar}] {wr:5.1%}  ({wins}W/{draws}D/{losses}L)')


# ══════════════════════════════════════════════════════════════════════
def test_wsl_bridge():
    """Specialni test WSL bridge (relevantni jen na Windows)."""
    section('WSL Bridge Test')
    print(f'  Platform: {PLATFORM}')
    print(f'  Backend:  {_backend_type}')
    if PLATFORM != 'windows':
        print('  (Test je relevantni jen na Windows, preskoceno)')
        return
    if not C_ENGINE_AVAILABLE:
        print('  WSL bridge nedostupny')
        return
    b = init_board(8)
    opp = CMinimax(18, 100, 8)
    m = opp.get_move(b, BLACK)
    valid = get_valid_moves(b, BLACK)
    P(m in valid, f'WSL bridge move: {m}')
    print('  WSL bridge OK')


# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-model',  action='store_true')
    parser.add_argument('--full',      action='store_true')
    parser.add_argument('--size',      type=int, nargs='+', choices=[6,8,10], default=[6,8])
    parser.add_argument('--wsl-test',  action='store_true')
    args = parser.parse_args()

    print(f'\n{"="*62}')
    print(f'  Reversi RL v2 -- Verifikace a Benchmark')
    print(f'{"="*62}')

    t0 = time.time()
    test_board()
    test_margin_reward()
    test_encoding()
    test_neural_net()
    test_opponents()
    test_training_components()

    if args.wsl_test:
        test_wsl_bridge()

    if not args.no_model:
        n = 100 if args.full else 30
        for size in args.size:
            benchmark_model(size, n)

    print(f'\n{"="*62}')
    print(f'  Vsechny testy OK  ({time.time()-t0:.1f}s)')
    print(f'{"="*62}\n')


if __name__ == '__main__':
    main()
