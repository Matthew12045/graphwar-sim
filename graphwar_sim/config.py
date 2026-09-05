"""Constants for the Graphwar NORMAL_FUNC simulator.

Every numeric constant is copied from the reference Java source and cited to a
specific ``file:line`` in ``ref/graphwar/src/``. Nothing here is invented.

The reference is GPL-licensed; per the project's clean-room rule we *cite* the
source rather than copy its code. These are bare numeric literals (facts), not
code.
"""

from __future__ import annotations

import math

# --- Plane / board geometry -------------------------------------------------
# Constants.java:62
PLANE_LENGTH: int = 770
# Constants.java:63
PLANE_HEIGHT: int = 450
# Constants.java:64  (horizontal game span; the vertical transform reuses this)
PLANE_GAME_LENGTH: int = 50

# --- Terrain / circle generation (server-authoritative) ---------------------
# Constants.java:66
CIRCLE_MEAN_RADIUS: int = 40
# Constants.java:67
CIRCLE_STANDARD_DEVIATION: int = 25
# Constants.java:68
NUM_CIRCLES_MEAN_VALUE: int = 15
# Constants.java:69
NUM_CIRCLES_STANDARD_DEVIATION: int = 7

# --- Soldiers ---------------------------------------------------------------
# Constants.java:71
SOLDIER_RADIUS: int = 7
# Constants.java:72
SOLDIER_SELECTION_RADIUS: int = 15
# Chebyshev minimum spacing between soldiers, in both x and y.
# GraphServer.java:687 (the literal 20 in testSoldier).
SOLDIER_MIN_CHEBYSHEV: int = 20

# --- Explosion marker (drives the last-point marker; not part of hit test) --
# Constants.java:79
EXPLOSION_RADIUS: int = 12

# --- Trajectory integration -------------------------------------------------
# Constants.java:85
FUNC_MAX_STEPS: int = 20000
# Constants.java:86
FUNC_MAX_STEP_DISTANCE_SQUARED: float = 0.001
# Constants.java:87
FUNC_MIN_X_STEP_DISTANCE: float = 0.00001
# Constants.java:88
STEP_SIZE: float = 0.01

# --- Fire-angle fixed-point iteration ---------------------------------------
# Constants.java:92
ANGLE_ERROR: float = math.pi / 360
# Constants.java:93
MAX_ANGLE_LOOPS: int = 100

# --- Teams / game modes -----------------------------------------------------
# Constants.java:55
TEAM1: int = 1
# Constants.java:56
TEAM2: int = 2
# Constants.java:58
NORMAL_FUNC: int = 0
# Constants.java:59
FST_ODE: int = 1
# Constants.java:60
SND_ODE: int = 2

# --- Match shape ------------------------------------------------------------
# Constants.java:48
MAX_PLAYERS: int = 10
# Constants.java:49
MAX_SOLDIERS_PER_PLAYER: int = 4
# Constants.java:51
INITIAL_NUM_SOLDIERS: int = 2
# Constants.java:46  (milliseconds; informational for a headless sim)
TURN_TIME: int = 60000

# --- Token type codes (drive operator precedence) ---------------------------
# FunctionToken.java:22-39. The integer *value* is the precedence key: a lower
# number is pulled out of a nest level first (see parser.reorder_rec).
ADD: int = 1
SUBTRACT: int = 2
MULTIPLY: int = 3
DIVIDE: int = 4
POW: int = 5
SQRT: int = 6
LOG: int = 7
ABS: int = 8
SIN: int = 9
COS: int = 10
TAN: int = 11
LN: int = 12
VARIABLE1: int = 13
VARIABLE2: int = 14
VARIABLE3: int = 15
VALUE: int = 16
LEFT_BRACKET: int = 17
RIGHT_BRACKET: int = 18

# The set of token types that are "operations" (isOperation,
# PolishNotationFunction.java:720-728: type in [1, 12]).
_OPERATION_TYPES: frozenset[int] = frozenset(range(ADD, LN + 1))

# Sentinel for "no operator found" in reorder_rec (mirrors Java's next = -1).
_NO_OPERATOR: int = -1


def is_operation(token_type: int) -> bool:
    """True if ``token_type`` is an operator (``FunctionToken`` type in [1,12]).

    Faithful to ``PolishNotationFunction.isOperation``
    (PolishNotationFunction.java:720-728).
    """
    return token_type in _OPERATION_TYPES


def get_num_param(token_type: int) -> int:
    """Number of operands an operator consumes.

    Faithful to ``PolishNotationFunction.getNumParam``
    (PolishNotationFunction.java:730-748):

    - SUBTRACT (2) is unary (1) — the ``-``→``+-`` rewrite makes every minus a
      negation.
    - ADD/MULTIPLY/DIVIDE/POW (1..5, except SUBTRACT) are binary (2).
    - SQRT..LN (6..12) are unary (1).
    - Variables / values / brackets are 0.
    """
    if token_type == SUBTRACT:
        return 1
    if ADD <= token_type <= POW:
        return 2
    if SQRT <= token_type <= LN:
        return 1
    return 0
