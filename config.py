"""
config.py  –  Centrální konfigurace pro Reversi RL trénink (v2).

Klíčové novinky oproti v1:
  1. C minimax engine (hloubka 1-18+, time-limited iterative deepening)
  2. 20 fázový curriculum: od náhodného přes všechny síly po elite mix
  3. Margin-based reward: dominance / silná / normální / těsná výhra-prohra
"""

# ══════════════════════════════════════════════════════════════════════
# ARCHITEKTURA SÍTĚ
# ══════════════════════════════════════════════════════════════════════

ARCH = {
    6:  {'H': 512, 'res_blocks': 3},
    8:  {'H': 1024, 'res_blocks': 3},
    10: {'H': 1024, 'res_blocks': 4},
}

INPUT_CHANNELS = 4

# ══════════════════════════════════════════════════════════════════════
# POČTY EPIZOD
# ══════════════════════════════════════════════════════════════════════

TOTAL_EPISODES = {
    6:  60_000,
    8:  70_000,
    10: 50_000,
}

FAST_MULTIPLIER = 0.08

# ══════════════════════════════════════════════════════════════════════
# LEARNING RATE
# ══════════════════════════════════════════════════════════════════════

LR_START  = 3e-4
LR_END    = 5e-6
LR_WARMUP = 500

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS   = 1e-8
L2_REG     = 5e-6
GRAD_CLIP  = 2.0

# ══════════════════════════════════════════════════════════════════════
# REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════

REPLAY_CAPACITY   = 350_000
REPLAY_MIN_FILL   = 1_024
BATCH_SIZE        = 256
TRAIN_EVERY       = 4
TARGET_NET_UPDATE = 200
TARGET_NET_TAU    = 0.005

# ══════════════════════════════════════════════════════════════════════
# EPSILON
# ══════════════════════════════════════════════════════════════════════

EPSILON_START = 0.40
EPSILON_END   = 0.03
EPSILON_DECAY = 0.9998

# ══════════════════════════════════════════════════════════════════════
# TD LEARNING
# ══════════════════════════════════════════════════════════════════════

GAMMA     = 0.97
TD_LAMBDA = 0.80

# ══════════════════════════════════════════════════════════════════════
# REWARD SHAPING
# ══════════════════════════════════════════════════════════════════════

REWARD_CORNER_WEIGHT    = 0.40
REWARD_MOBILITY_WEIGHT  = 0.25
REWARD_STABILITY_WEIGHT = 0.20
REWARD_FORCE_PASS       = 0.60

# ── MARGIN-BASED FINAL REWARD ────────────────────────────────────────
# diff = (my_discs - opp_discs) / total  in [-1, 1]
#
# VÝHRA:
#   diff >= 0.50  → Dominance    (+1.00)   >=75% kamenů
#   diff >= 0.20  → Silná výhra  (+0.75)   60-75%
#   diff >= 0.08  → Normální     (+0.50)   54-60%
#   diff >  0.00  → Těsná výhra  (+0.25)   50-54%
# REMÍZA:
#   diff == 0.00  → Remíza       (+0.08)
# PROHRA:
#   diff >= -0.08 → Těsná prohra (-0.25)
#   diff >= -0.20 → Normální     (-0.50)
#   diff >= -0.50 → Silná prohra (-0.75)
#   diff <  -0.50 → Dominována   (-1.00)   <=25% kamenů
#
MARGIN_THRESHOLDS = [
    (  0.50,  1.00, 'dominance'),
    (  0.20,  0.75, 'strong_win'),
    (  0.08,  0.50, 'normal_win'),
    (  0.001, 0.25, 'tight_win'),
    (  0.0,   0.08, 'draw'),
    ( -0.08, -0.25, 'tight_loss'),
    ( -0.20, -0.50, 'normal_loss'),
    ( -0.50, -0.75, 'strong_loss'),
    ( -2.00, -1.00, 'dominated'),
]

FINAL_BLEND_START = 0.20
FINAL_BLEND_END   = 0.85

IMPROVEMENT_EMA_ALPHA = 0.05
IMPROVEMENT_MAX_BONUS = 0.30

# ══════════════════════════════════════════════════════════════════════
# CURRICULUM LEARNING  (20 fází)
# ══════════════════════════════════════════════════════════════════════
#
# Typy: random, greedy, minimax_py, minimax_c, self, mixed, elite

CURRICULUM = [
    # Fáze 1-2: Základ
    ('random',         0.02, 'random',    {}),
    ('greedy',         0.03, 'greedy',    {}),
    # Fáze 3-8: Python minimax (depth-limited, identické napříč deskami)
    ('mm_py_d1',       0.02, 'minimax_py', {'depth': 1}),
    ('mm_py_d2',       0.03, 'minimax_py', {'depth': 2}),
    ('mm_py_d3',       0.03, 'minimax_py', {'depth': 3}),
    ('mm_py_d4',       0.04, 'minimax_py', {'depth': 4}),
    ('mm_py_d5',       0.05, 'minimax_py', {'depth': 5}),
    ('mm_py_d6',       0.05, 'minimax_py', {'depth': 6}),

    # Fáze 9-10: C minimax, target_depth = garantovaná hloubka
    # (škálování dle velikosti desky probíhá v CurriculumManager)
    ('mm_c_fast',      0.03, 'minimax_c',  {'target_depth': 6}),  
    ('mm_c_light',     0.03, 'minimax_c',  {'target_depth': 7}),   

    ('self_play_1',    0.06, 'self',       {}),

    ('mm_c_medium',    0.05, 'minimax_c',  {'target_depth': 8}),   
    ('mixed_med_self', 0.05, 'mixed',      {'target_depth': 8,  'self_ratio': 0.4}),

    ('mm_c_strong',    0.06, 'minimax_c',  {'target_depth': 9}),  
    ('self_play_2',    0.05, 'self',       {}),

    ('mm_c_vstrong',   0.06, 'minimax_c',  {'target_depth': 10}), 
    ('mixed_vs_self',  0.05, 'mixed',      {'target_depth': 10, 'self_ratio': 0.35}),

    ('mm_c_elite_1',   0.06, 'minimax_c',  {'target_depth': 12}), 
    ('mm_c_elite_2',   0.06, 'minimax_c',  {'target_depth': 13}),  

    ('all_mix',        0.17, 'elite',      {
        'target_depths': [6, 7, 8, 9, 10, 12, 13],  
        'self_ratio':    0.40,
        'py_depths':     [1, 2, 3, 4, 5, 6],
        'py_ratio':      0.30,
    }),
]

_total = sum(p[1] for p in CURRICULUM)
assert abs(_total - 1.0) < 1e-6, f'Curriculum suma != 1.0 (got {_total:.6f})'

CURRICULUM_PROMOTE_THRESHOLD = 0.8
CURRICULUM_EVAL_WINDOW       = 100

# ══════════════════════════════════════════════════════════════════════
# CHECKPOINTING A LOGOVÁNÍ
# ══════════════════════════════════════════════════════════════════════

CHECKPOINT_INTERVAL    = 3_000
LOG_INTERVAL           = 200
MODEL_DIR              = 'models'
LOG_DIR                = 'logs'

# ══════════════════════════════════════════════════════════════════════
# SELF-PLAY POOL
# ══════════════════════════════════════════════════════════════════════

SELFPLAY_POOL_SIZE      = 8
SELFPLAY_SNAPSHOT_EVERY = 1_500
