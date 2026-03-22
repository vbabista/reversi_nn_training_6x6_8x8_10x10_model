# opponents/__init__.py
from opponents.random_opponent      import RandomOpponent
from opponents.greedy_opponent      import GreedyOpponent
from opponents.minimax_opponent     import MinimaxOpponent
from opponents.c_minimax_bridge     import CMinimax, make_c_minimax, C_ENGINE_AVAILABLE
