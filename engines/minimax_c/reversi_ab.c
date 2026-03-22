/*
 * reversi_ab.c - Elite Alpha-Beta minimax engine for Reversi (6x6, 8x8, 10x10)
 * * Optimizations:
 * - 1D Bordered Board: Extremely fast move generation.
 * - PVS (Principal Variation Search) + Aspiration Windows.
 * - Transposition Table: Zobrist hashing with 4M entries.
 * - Dynamic Heuristics: Mobility, Frontier discs, Corner/Edge stability.
 * - Phase Evaluation: Midgame focuses on mobility, Endgame on disc count.
 * - Perfect Endgame Solver: Exact score calculation when empty squares <= 14.
 * * NOTE: No non-ASCII characters in comments to preserve WSL/Python bridge compatibility.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <limits.h>

/* --- Constants & Macros --- */
#define MAX_SIZE     10
#define MAX_WIDTH    (MAX_SIZE + 2)
#define MAX_BOARD    (MAX_WIDTH * MAX_WIDTH)
#define MAX_MOVES    64

#define EMPTY        (-1)
#define BLACK        0
#define WHITE        1
#define BORDER       (-2)

#define SCORE_INF    1000000
#define SCORE_WIN    900000
#define SCORE_DRAW   0

/* Transposition Table (22 bits = 4.19 million entries) */
#define TT_BITS      22
#define TT_SIZE      (1u << TT_BITS)
#define TT_MASK      (TT_SIZE - 1u)

#define TT_EXACT     0
#define TT_LOWER     1
#define TT_UPPER     2

/* --- Data Structures --- */
typedef struct {
    uint64_t hash;
    int      score;
    int16_t  depth;
    uint8_t  flag;
    uint8_t  best_move;
} TTEntry;

typedef struct {
    int sq;
    int priority;
} Move;

/* --- Globals --- */
static TTEntry tt[TT_SIZE];
static uint64_t zobrist[MAX_BOARD][2];
static int      zobrist_ready = 0;

static clock_t  search_start;
static clock_t  time_limit_clocks;
static int      time_exceeded;
/* FIX: depth-only mode - when 1, ignore time limit and search to full max_depth */
static int      depth_only_mode;
static long long nodes_searched;

static int history[2][MAX_BOARD];
#define MAX_DEPTH 64
static int killer_moves[MAX_DEPTH][2];

static int DIRS[8];
static int BOARD_W;

/* Static positional weights mapped dynamically based on board size */
static int pos_weights[MAX_BOARD];

/* --- Zobrist Hashing --- */
static uint64_t lcg64(uint64_t *state) {
    *state = (*state) * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state;
}

static void init_engine(int size) {
    if (!zobrist_ready) {
        uint64_t state = 0xDEADBEEFCAFEBABEULL;
        for (int i = 0; i < MAX_BOARD; i++) {
            zobrist[i][0] = lcg64(&state);
            zobrist[i][1] = lcg64(&state);
        }
        zobrist_ready = 1;
    }
    
    BOARD_W = size + 2;
    DIRS[0] = -BOARD_W - 1; DIRS[1] = -BOARD_W;     DIRS[2] = -BOARD_W + 1;
    DIRS[3] = -1;                                   DIRS[4] = 1;
    DIRS[5] =  BOARD_W - 1; DIRS[6] =  BOARD_W;     DIRS[7] =  BOARD_W + 1;

    /* Build dynamic positional weight map for 6x6, 8x8, or 10x10 */
    for (int r = 1; r <= size; r++) {
        for (int c = 1; c <= size; c++) {
            int sq = r * BOARD_W + c;
            int v = 1; // Default inner square
            
            int is_edge = (r == 1 || r == size || c == 1 || c == size);
            int is_corner = ((r == 1 || r == size) && (c == 1 || c == size));
            int is_x = ((r == 2 || r == size - 1) && (c == 2 || c == size - 1));
            int is_c = (((r == 1 || r == size) && (c == 2 || c == size - 1)) || 
                        ((r == 2 || r == size - 1) && (c == 1 || c == size)));
            
            if (is_corner) v = 150;
            else if (is_x) v = -40;
            else if (is_c) v = -20;
            else if (is_edge) v = 10;
            
            pos_weights[sq] = v;
        }
    }
}

static uint64_t compute_hash(const int8_t *b) {
    uint64_t h = 0ULL;
    for (int i = 0; i < MAX_BOARD; i++) {
        if (b[i] == BLACK) h ^= zobrist[i][BLACK];
        else if (b[i] == WHITE) h ^= zobrist[i][WHITE];
    }
    return h;
}

void clear_tt(void) {
    memset(tt, 0, sizeof(tt));
}

static inline void tt_store(uint64_t hash, int score, int depth, int flag, int best_sq) {
    TTEntry *e = &tt[hash & TT_MASK];
    if (e->hash == 0 || e->depth <= depth || flag == TT_EXACT) {
        e->hash      = hash;
        e->score     = score;
        e->depth     = (int16_t)depth;
        e->flag      = (uint8_t)flag;
        e->best_move = (uint8_t)best_sq;
    }
}

static inline int tt_lookup(uint64_t hash, int depth, int alpha, int beta, int *score, int *best_sq) {
    TTEntry *e = &tt[hash & TT_MASK];
    if (e->hash != hash) return 0;
    *best_sq = e->best_move;
    if (e->depth < depth) return 0;
    *score = e->score;
    if (e->flag == TT_EXACT) return 1;
    if (e->flag == TT_LOWER && *score >= beta) return 1;
    if (e->flag == TT_UPPER && *score <= alpha) return 1;
    return 0;
}

/* --- Game Mechanics --- */
static int get_valid_moves(const int8_t *b, int color, int moves[MAX_MOVES]) {
    int opp = 1 - color;
    int n = 0;
    for (int r = 1; r <= BOARD_W - 2; r++) {
        for (int c = 1; c <= BOARD_W - 2; c++) {
            int sq = r * BOARD_W + c;
            if (b[sq] != EMPTY) continue;
            for (int d = 0; d < 8; d++) {
                int step = DIRS[d];
                int p = sq + step;
                if (b[p] == opp) {
                    p += step;
                    while (b[p] == opp) p += step;
                    if (b[p] == color) {
                        moves[n++] = sq;
                        break;
                    }
                }
            }
        }
    }
    return n;
}

static inline int make_move(int8_t *b, int sq, int color, uint64_t *hash) {
    int opp = 1 - color;
    int flipped = 0;
    b[sq] = (int8_t)color;
    *hash ^= zobrist[sq][color];
    
    for (int d = 0; d < 8; d++) {
        int step = DIRS[d];
        int p = sq + step;
        if (b[p] == opp) {
            int p2 = p + step;
            while (b[p2] == opp) p2 += step;
            if (b[p2] == color) {
                for (int p3 = sq + step; p3 != p2; p3 += step) {
                    b[p3] = (int8_t)color;
                    *hash ^= zobrist[p3][opp];
                    *hash ^= zobrist[p3][color];
                    flipped++;
                }
            }
        }
    }
    return flipped;
}

/* --- Advanced Heuristics --- */
static int evaluate_pos(const int8_t *b, int size, int color, int empty_count) {
    int opp = 1 - color;
    int my_discs = 0, op_discs = 0;
    int my_front = 0, op_front = 0;
    int my_pos = 0, op_pos = 0;

    /* Corner squares indices */
    int c1 = 1 * BOARD_W + 1;
    int c2 = 1 * BOARD_W + size;
    int c3 = size * BOARD_W + 1;
    int c4 = size * BOARD_W + size;

    for (int r = 1; r <= size; r++) {
        for (int c = 1; c <= size; c++) {
            int sq = r * BOARD_W + c;
            int piece = b[sq];
            if (piece == EMPTY) continue;
            
            if (piece == color) my_discs++;
            else op_discs++;
            
            /* Positional score with Corner Override */
            int p_weight = pos_weights[sq];
            if (p_weight < 0) {
                // If it's an X or C square, check if the adjacent corner is ours
                int corner_sq = 0;
                if (r <= 2 && c <= 2) corner_sq = c1;
                else if (r <= 2 && c >= size - 1) corner_sq = c2;
                else if (r >= size - 1 && c <= 2) corner_sq = c3;
                else if (r >= size - 1 && c >= size - 1) corner_sq = c4;
                
                if (corner_sq != 0 && b[corner_sq] == piece) {
                    p_weight = 20; // Safe X/C square is actually good!
                }
            }
            
            if (piece == color) my_pos += p_weight;
            else op_pos += p_weight;
            
            /* Frontier detection (adjacent to empty) */
            int is_frontier = 0;
            for (int d = 0; d < 8; d++) {
                if (b[sq + DIRS[d]] == EMPTY) {
                    is_frontier = 1;
                    break;
                }
            }
            if (is_frontier) {
                if (piece == color) my_front++;
                else op_front++;
            }
        }
    }

    /* Exact Win/Loss in Perfect Endgame */
    if (empty_count == 0 || my_discs == 0 || op_discs == 0) {
        if (my_discs > op_discs) return SCORE_WIN + my_discs - op_discs;
        if (my_discs < op_discs) return -(SCORE_WIN + op_discs - my_discs);
        return SCORE_DRAW;
    }

    /* Mobility calculation */
    int tmp_moves[MAX_MOVES];
    int my_mob = get_valid_moves(b, color, tmp_moves);
    int op_mob = get_valid_moves(b, opp, tmp_moves);

    /* Parity */
    int parity = (empty_count & 1) ? 30 : -30;

    /* Dynamic Weights based on game phase */
    float phase = 1.0f - (float)empty_count / (size * size);
    
    int score = 0;
    int pos_diff = my_pos - op_pos;
    int mob_diff = my_mob - op_mob;
    int front_diff = op_front - my_front; /* We want FEWER frontier discs */
    int disc_diff = my_discs - op_discs;

    if (phase < 0.3f) {
        score = pos_diff * 10 + mob_diff * 15 + front_diff * 5;
    } else if (phase < 0.7f) {
        score = pos_diff * 12 + mob_diff * 10 + front_diff * 10 + parity;
    } else {
        /* Nearing endgame, transition to absolute disc count */
        score = pos_diff * 5 + mob_diff * 5 + disc_diff * 15 + parity * 2;
    }

    return score;
}

/* --- Move Ordering --- */
static void sort_moves(Move moves[], int n, int tt_best_sq, int depth, int color) {
    for (int i = 0; i < n; i++) {
        int sq = moves[i].sq;
        int pri = 0;

        if (sq == tt_best_sq) {
            pri = 1000000;
        } else if (sq == killer_moves[depth][0] || sq == killer_moves[depth][1]) {
            pri = 500000;
        } else {
            pri = history[color][sq] + pos_weights[sq] * 10;
        }
        moves[i].priority = pri;
    }

    /* Insertion sort */
    for (int i = 1; i < n; i++) {
        Move key = moves[i];
        int j = i - 1;
        while (j >= 0 && moves[j].priority < key.priority) {
            moves[j+1] = moves[j];
            j--;
        }
        moves[j+1] = key;
    }
}

/* --- PVS Search --- */
static int pvs(int8_t *b, uint64_t hash, int size, int color, int depth, int alpha, int beta, int empty, int pass) {
    nodes_searched++;
    /* FIX: Only check time when in time-limited mode (depth_only_mode == 0) */
    if (!depth_only_mode && (nodes_searched & 2047) == 0) {
        if (clock() - search_start >= time_limit_clocks) time_exceeded = 1;
    }
    if (time_exceeded) return 0;

    int opp = 1 - color;

    if (empty == 0) {
        int m = 0, o = 0;
        for(int i=0; i<MAX_BOARD; i++) { if(b[i]==color) m++; else if(b[i]==opp) o++; }
        if (m > o) return SCORE_WIN + m - o;
        if (m < o) return -(SCORE_WIN + o - m);
        return SCORE_DRAW;
    }

    int tt_score, tt_best_sq = 0;
    if (tt_lookup(hash, depth, alpha, beta, &tt_score, &tt_best_sq)) {
        return tt_score;
    }

    if (depth <= 0) {
        return evaluate_pos(b, size, color, empty);
    }

    int valid_sqs[MAX_MOVES];
    int n_moves = get_valid_moves(b, color, valid_sqs);

    if (n_moves == 0) {
        if (pass) {
            int m = 0, o = 0;
            for(int i=0; i<MAX_BOARD; i++) { if(b[i]==color) m++; else if(b[i]==opp) o++; }
            if (m > o) return SCORE_WIN + m - o;
            if (m < o) return -(SCORE_WIN + o - m);
            return SCORE_DRAW;
        }
        return -pvs(b, hash, size, opp, depth, -beta, -alpha, empty, 1);
    }

    Move moves[MAX_MOVES];
    for (int i = 0; i < n_moves; i++) moves[i].sq = valid_sqs[i];
    sort_moves(moves, n_moves, tt_best_sq, depth < MAX_DEPTH ? depth : MAX_DEPTH-1, color);

    int best_score = -SCORE_INF;
    int best_sq = moves[0].sq;
    int orig_alpha = alpha;

    for (int i = 0; i < n_moves; i++) {
        if (time_exceeded) break;

        int sq = moves[i].sq;
        int8_t nb[MAX_BOARD];
        memcpy(nb, b, MAX_BOARD * sizeof(int8_t));
        uint64_t nh = hash;
        
        make_move(nb, sq, color, &nh);

        int score;
        if (i == 0) {
            score = -pvs(nb, nh, size, opp, depth - 1, -beta, -alpha, empty - 1, 0);
        } else {
            score = -pvs(nb, nh, size, opp, depth - 1, -alpha - 1, -alpha, empty - 1, 0);
            if (score > alpha && score < beta) {
                score = -pvs(nb, nh, size, opp, depth - 1, -beta, -score, empty - 1, 0);
            }
        }

        if (score > best_score) {
            best_score = score;
            best_sq = sq;
        }
        if (score > alpha) alpha = score;
        
        if (alpha >= beta) {
            int d = depth < MAX_DEPTH ? depth : MAX_DEPTH - 1;
            if (killer_moves[d][0] != sq) {
                killer_moves[d][1] = killer_moves[d][0];
                killer_moves[d][0] = sq;
            }
            history[color][sq] += depth * depth;
            break;
        }
    }

    if (!time_exceeded) {
        int flag = (best_score <= orig_alpha) ? TT_UPPER :
                   (best_score >= beta)       ? TT_LOWER : TT_EXACT;
        tt_store(hash, best_score, depth, flag, best_sq);
    }

    return best_score;
}

/* --- API --- */
int count_valid_moves(const int8_t *board_in, int size, int color) {
    init_engine(size);
    int8_t b[MAX_BOARD];
    for (int i = 0; i < MAX_BOARD; i++) b[i] = BORDER;
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            b[(r + 1) * BOARD_W + (c + 1)] = board_in[r * size + c];
        }
    }
    int moves[MAX_MOVES];
    return get_valid_moves(b, color, moves);
}

int reversi_get_move(const int8_t *board_in, int size, int color, int max_depth, int time_limit_ms) {
    init_engine(size);
    if (max_depth < 1) max_depth = 1;
    if (max_depth > 60) max_depth = 60;

    int8_t b[MAX_BOARD];
    int empty_count = 0;
    for (int i = 0; i < MAX_BOARD; i++) b[i] = BORDER;
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            int8_t val = board_in[r * size + c];
            b[(r + 1) * BOARD_W + (c + 1)] = val;
            if (val == EMPTY) empty_count++;
        }
    }

    int moves[MAX_MOVES];
    int n = get_valid_moves(b, color, moves);
    if (n == 0) return -1;
    if (n == 1) {
        int r = moves[0] / BOARD_W - 1;
        int c = moves[0] % BOARD_W - 1;
        return r * 100 + c;
    }

    /* Perfect endgame solver kick-in */
    if (empty_count <= 14 && max_depth < empty_count) {
        max_depth = empty_count;
    }

    search_start = clock();

    /* FIX: Activate depth-only mode when time_limit_ms <= 0.
     * Engine searches to full max_depth with NO time cutoff.
     * Guarantees consistent strength independent of hardware speed. */
    if (time_limit_ms <= 0) {
        depth_only_mode = 1;
        time_limit_clocks = 0; /* unused in depth-only mode */
    } else {
        depth_only_mode = 0;
        time_limit_clocks = (clock_t)((double)time_limit_ms * CLOCKS_PER_SEC / 1000.0);
    }
    time_exceeded = 0;
    nodes_searched = 0;

    /* Aging the history slightly instead of full wipe could be better, but full wipe is safe */
    memset(history, 0, sizeof(history));
    memset(killer_moves, 0, sizeof(killer_moves));

    uint64_t hash = compute_hash(b);
    int best_sq = moves[0];
    int best_score = -SCORE_INF;

    for (int depth = 1; depth <= max_depth; depth++) {
        if (time_exceeded) break;

        int alpha = -SCORE_INF, beta = SCORE_INF;
        if (depth >= 4 && best_score > -SCORE_WIN) {
            int window = 200 + (depth < 8 ? 200 : 0);
            alpha = best_score - window;
            beta  = best_score + window;
        }

        int score;
        int tries = 0;
        while (1) {
            score = pvs(b, hash, size, color, depth, alpha, beta, empty_count, 0);

            if (time_exceeded) break;

            if (score <= alpha) {
                alpha = alpha - 300 - tries * 500;
                if (alpha < -SCORE_INF) alpha = -SCORE_INF;
            } else if (score >= beta) {
                beta = beta + 300 + tries * 500;
                if (beta > SCORE_INF) beta = SCORE_INF;
            } else {
                break;
            }
            tries++;
            if (tries >= 3) {
                alpha = -SCORE_INF; beta = SCORE_INF;
            }
        }

        if (!time_exceeded || depth == 1) {
            best_score = score;
            TTEntry *e = &tt[hash & TT_MASK];
            if (e->hash == hash && e->best_move != 0) {
                best_sq = e->best_move;
            }
        }

        if (best_score >= SCORE_WIN || best_score <= -SCORE_WIN) break;

        /* FIX: Only apply time-based early abort in time-limited mode */
        if (!depth_only_mode) {
            double elapsed = (double)(clock() - search_start) / CLOCKS_PER_SEC * 1000.0;
            if ((time_limit_ms - elapsed) < (elapsed * 0.5) && depth >= 4) {
                break;
            }
        }
    }

    int final_r = best_sq / BOARD_W - 1;
    int final_c = best_sq % BOARD_W - 1;
    return final_r * 100 + final_c;
}