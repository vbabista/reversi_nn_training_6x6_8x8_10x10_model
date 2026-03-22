# reversi-rl

Hluboká reziduální neuronová síť pro hru Reversi, trénovaná pomocí reinforcement learning.  
**Pouze NumPy** – bez PyTorch nebo TensorFlow. Modely uloženy jako `.npy`.

---

## Spuštění

```bash
# Zkompiluj C engine (jednou)
cd engines/minimax_c && bash build.sh && cd ../..

# Trénink
python train.py              # 6×6, 8×8, 10×10 paralelně s dashboardem
python train.py --size 8     # jen 8×8
python train.py --fast       # 8 % epizod (rychlý test, ~5 min)
python train.py --resume     # pokračuj z checkpointu
python train.py --sequential # bez paralelismu (debug)

# Ověření + benchmark
python verify.py
python verify.py --size 8 --full
```

### Windows (WSL)

```bat
wsl
cd /mnt/c/Users/<jméno>/reversi-rl
cd engines/minimax_c && bash build.sh && cd ../..
python3 train.py --size 8
```

---

## Struktura

```
reversi-rl/
├── train.py              ← hlavní vstupní bod
├── verify.py             ← ověření + benchmark vs minimax
├── config.py             ← veškerá konfigurace
├── requirements.txt
│
├── core/
│   └── deska.py          ← herní logika (6×6, 8×8, 10×10)
│
├── model/
│   └── sit.py            ← reziduální síť, Adam, save/load .npy
│
├── training/
│   ├── buffer.py         ← prioritizovaný replay buffer (PER)
│   ├── odmeny.py         ← delta reward + margin final reward
│   ├── metriky.py        ← CSV logování
│   ├── curriculum.py     ← 20fázový adaptivní curriculum
│   └── selfplay_pool.py  ← pool historických snapshotů
│
├── opponents/
│   ├── nahodny.py        ← náhodný oponent (1. fáze)
│   ├── greedy.py       ← greedy oponent (2. fáze)
│   ├── minimax.py        ← Python alfa-beta minimax (hloubka 1–6)
│   └── c_minimax_bridge.py ← bridge pro C engine (Linux/WSL/Windows)
│
├── engines/minimax_c/
│   ├── reversi_ab.c      ← C alfa-beta engine
│   ├── reversi_ab.so     ← zkompilovaná knihovna (po build.sh)
│   ├── build.sh          ← kompilační skript
│   └── wsl_worker.py     ← WSL subprocess worker (pro Windows)
│
├── models/               ← natrénované modely (.npy)
└── logs/                 ← logy a CSV statistiky
```

---

## Architektura sítě

| Deska | H    | Bloky | Parametry (přibližně) |
|-------|------|-------|----------------------|
| 6×6   | 512  | 3     | ~5 M                 |
| 8×8   | 1024 | 3     | ~20 M                |
| 10×10 | 1024 | 4     | ~27 M                |

Vstup: `4 × N²` (moje kameny, soupeřovy kameny, moje platné tahy, soupeřovy tahy)  
Výstup: tanh ∈ (−1, +1) – hodnota pozice pro hráče na tahu

---

## Curriculum (20 fází)

Od náhodného hráče přes Python minimax (hloubka 1–6) až po C minimax (hloubky 6–13)  
a self-play mix. Adaptive curriculum automaticky povyšuje fázi při win rate ≥ 80 %.

---

## Reward

- **Per-step delta reward**: změna rohů, mobility, stability + force-pass bonus
- **Margin-based final reward**: dominance (+1.0) → těsná výhra (+0.25) → remíza (+0.08) → prohra (symetricky záporná)
- **TD(λ) return**: blend margin výsledku a λ-returnu (lambda roste s progress)
- **Cross-game improvement bonus**: EMA srovnání aktuální hry s historickým průměrem

---

## Požadavky

```
numpy>=1.21.0
```

Pro C engine: `gcc` (Linux/WSL).
