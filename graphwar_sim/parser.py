"""Faithful Python port of ``Graphwar.PolishNotationFunction``.

This is a clean-room *port*: the algorithm is reproduced from the reference Java
source (cited per step) but not copied verbatim. It is **not** ``eval``/``exec``/
``sympify`` — the expression is tokenized, reordered, and evaluated by an
explicit recursive-descent read over the token list. LLM-supplied strings
therefore never reach any Python code-evaluation primitive (see the project's
security ground rule).

Reference: ``ref/graphwar/src/Graphwar/PolishNotationFunction.java`` and
``FunctionToken.java``.

The emitted form is **prefix (Polish) notation** — operator first, then its
operands. ``reorderRec`` (PolishNotationFunction.java:78-149) appends the chosen
operator *before* recursing into its operand ranges, and ``evaluateRec``
(PolishNotationFunction.java:968-1127) reads the operator first. (The class name
"PolishNotation" refers to this prefix order, not Reverse Polish.)
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from . import config

# Java Math edge-case sentinels.
_NAN: float = float("nan")
_INF: float = float("inf")


class MalformedFunction(Exception):
    """Raised when a function string does not form a well-formed expression.

    Faithful to the reference: ``MalformedFunction`` is an *empty* exception with
    no message (``MalformedFunction.java``). We keep it message-free so the port
    does not invent a diagnostic taxonomy that the source never had.
    """


# --- Java Math semantics (no exceptions; propagate NaN/Inf like the JVM) -----
# The integrator relies on NaN/Infinity propagating from these operations and
# terminating on them (Function.java:293-297). Python raises instead, so we
# replicate the JVM's IEEE-754 behaviour for the operations the parser uses.


def _java_div(a: float, b: float) -> float:
    """Java double division: ``0/0 -> NaN``, ``x/0 -> ±Inf`` (no exception)."""
    if b == 0.0:
        if a == 0.0 or math.isnan(a):
            return _NAN
        return _INF if a > 0 else -_INF
    return a / b


def _java_sqrt(x: float) -> float:
    """Java ``Math.sqrt``: negative -> NaN (no exception)."""
    if x < 0:
        return _NAN
    return math.sqrt(x)


def _java_log(x: float) -> float:
    """Java ``Math.log`` (natural): 0 -> -Inf, negative -> NaN."""
    if x == 0.0:
        return -_INF
    if x < 0 or math.isnan(x):
        return _NAN
    return math.log(x)


def _java_log10(x: float) -> float:
    """Java ``Math.log10``: 0 -> -Inf, negative -> NaN."""
    if x == 0.0:
        return -_INF
    if x < 0 or math.isnan(x):
        return _NAN
    return math.log10(x)


def _java_pow(base: float, exp: float) -> float:
    """Java ``Math.pow``: NaN/Inf per IEEE, never raises.

    Covers the cases the game actually hits (integer powers, ``x^0.5``-style
    fractional powers, negative bases) and matches the JDK's documented
    semantics for the non-finite edge cases.
    """
    if math.isnan(base) or math.isnan(exp):
        return _NAN
    if exp == 0.0:
        return 1.0
    if base == 1.0:
        return 1.0
    if base == 0.0:
        return 0.0 if exp > 0 else _INF
    if math.isinf(exp):
        if base > 1:
            return _INF if exp > 0 else 0.0
        if 0 < base < 1:
            return 0.0 if exp > 0 else _INF
        return _NAN
    if math.isinf(base):
        if exp > 0:
            if base < 0 and exp == int(exp) and (int(exp) % 2 != 0):
                return -_INF
            return _INF
        return 0.0  # exp < 0
    if base < 0:
        if exp != int(exp):
            return _NAN  # negative base, non-integer exponent
        n = int(exp)
        try:
            mag = math.pow(-base, exp)
        except OverflowError:
            mag = _INF
        return mag if n % 2 == 0 else -mag
    try:
        return math.pow(base, exp)
    except OverflowError:
        return _INF


def _java_sin(x: float) -> float:
    """Java ``Math.sin``: non-finite input -> NaN (no exception)."""
    if not math.isfinite(x):
        return _NAN
    return math.sin(x)


def _java_cos(x: float) -> float:
    if not math.isfinite(x):
        return _NAN
    return math.cos(x)


def _java_tan(x: float) -> float:
    if not math.isfinite(x):
        return _NAN
    return math.tan(x)


# --- Token representation ----------------------------------------------------


class _Token:
    """A single token: a type code plus, for VALUE tokens, a numeric payload.

    Mirrors ``FunctionToken`` (type) + ``ValueToken`` (type + value).
    """

    __slots__ = ("type", "value")

    def __init__(self, type: int, value: float | None = None) -> None:
        self.type = type
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.type == config.VALUE:
            return f"ValueToken({self.value!r})"
        return f"FunctionToken({self.type})"


# The numeric-literal alternative from the reference tokenizer
# (PolishNotationFunction.java:217). Used to decide "is this a number" exactly
# the way Java's ``Double.parseDouble`` succeeds on the regex-produced tokens.
_NUMBER_RE = re.compile(r"[0-9]*\.?[0-9]+$")

# The full tokenizer pattern (PolishNotationFunction.java:217). ``re.finditer``
# has the same "skip non-matching characters" semantics as Java's
# ``Matcher.find()`` loop.
_TOKEN_RE = re.compile(
    r"[0-9]*\.?[0-9]+|\(|\)|x|y'|y|\+|\*|/|\^|sqrt|log|abs|sin|sen|cos|tan|tg|-|ln|e|pi"
)


def _make_value_token(token: str) -> _Token:
    return _Token(config.VALUE, float(token))


def _keyword_token(token: str) -> _Token | None:
    """Map a non-numeric token to its type, mirroring the Java if/else chain
    (PolishNotationFunction.java:236-311). Returns ``None`` if unrecognized
    (cannot happen for regex-produced tokens)."""
    if token == "x":
        return _Token(config.VARIABLE1)
    if token == "y":
        return _Token(config.VARIABLE2)
    if token == "y'":
        return _Token(config.VARIABLE3)
    if token == "+":
        return _Token(config.ADD)
    if token == "-":
        return _Token(config.SUBTRACT)
    if token == "*":
        return _Token(config.MULTIPLY)
    if token == "/":
        return _Token(config.DIVIDE)
    if token == "sqrt":
        return _Token(config.SQRT)
    if token == "log":
        return _Token(config.LOG)
    if token == "abs":
        return _Token(config.ABS)
    if token == "sin" or token == "sen":
        return _Token(config.SIN)
    if token == "cos":
        return _Token(config.COS)
    if token == "tan" or token == "tg":
        return _Token(config.TAN)
    if token == "^":
        return _Token(config.POW)
    if token == "ln":
        return _Token(config.LN)
    if token == "e":
        return _Token(config.VALUE, math.e)
    if token == "pi":
        return _Token(config.VALUE, math.pi)
    if token == "(":
        return _Token(config.LEFT_BRACKET)
    if token == ")":
        return _Token(config.RIGHT_BRACKET)
    return None


def _is_implicit(type1: int, type2: int) -> bool:
    """``isImplicit`` (PolishNotationFunction.java:194-207)."""
    left = type1 in (
        config.VALUE,
        config.VARIABLE1,
        config.VARIABLE2,
        config.VARIABLE3,
        config.RIGHT_BRACKET,
    )
    right = type2 in (
        config.VALUE,
        config.VARIABLE1,
        config.VARIABLE2,
        config.VARIABLE3,
        config.LEFT_BRACKET,
    ) or config.get_num_param(type2) == 1
    return left and right


def _adjust_implicit_multiplications(tokens: list[_Token]) -> list[_Token]:
    """``adjustImplicitMultiplications`` (PolishNotationFunction.java:162-192).

    Insert a MULTIPLY between adjacent tokens where ``isImplicit`` holds.
    """
    if not tokens:
        return tokens
    out: list[_Token] = [tokens[0]]
    for nxt in tokens[1:]:
        if _is_implicit(out[-1].type, nxt.type):
            out.append(_Token(config.MULTIPLY))
        out.append(nxt)
    return out


def _create_regular_notation_tokens(arg_str: str) -> list[_Token]:
    """``createRegularNotationTokens`` (PolishNotationFunction.java:209-318)."""
    func_str = arg_str.lower()
    # Every '-' becomes '+-' (unary negation); 'exp' -> 'e^'; ',' -> '.'.
    func_str = func_str.replace("-", "+-")
    func_str = func_str.replace("exp", "e^")
    func_str = func_str.replace(",", ".")

    normal: list[_Token] = []
    for m in _TOKEN_RE.finditer(func_str):
        token = m.group(0)
        if _NUMBER_RE.fullmatch(token):
            normal.append(_make_value_token(token))
            continue
        kw = _keyword_token(token)
        if kw is not None:
            normal.append(kw)
        # Unknown characters are silently dropped (find() semantics).

    return _adjust_implicit_multiplications(normal)


def _precedes(t0: int, t1: int) -> bool:
    """``precedes`` (PolishNotationFunction.java:151-159): lower type wins."""
    return t0 < t1


def _reorder_rec(
    polish: list[_Token], tokens: Sequence[_Token], start: int, end: int
) -> bool:
    """``reorderRec`` (PolishNotationFunction.java:78-149).

    Recursively pull out the lowest-nest operator (tie-broken by ``precedes``)
    and emit it in prefix order (operator before its operands). Returns whether
    any token was emitted.
    """
    if start > end or start >= len(tokens):
        return False

    next_idx = config._NO_OPERATOR
    next_nest = math.inf

    nest = 0
    for i in range(start, end + 1):
        t = tokens[i].type
        if t == config.LEFT_BRACKET:
            nest += 1
        elif t == config.RIGHT_BRACKET:
            nest -= 1
        elif nest < next_nest or (
            nest == next_nest
            and (next_idx == config._NO_OPERATOR or _precedes(t, tokens[next_idx].type))
        ):
            next_idx = i
            next_nest = nest

    if next_idx == config._NO_OPERATOR:
        return False

    nparam = config.get_num_param(tokens[next_idx].type)
    if nparam == 0:
        polish.append(tokens[next_idx])
    elif nparam == 1:
        polish.append(tokens[next_idx])
        _reorder_rec(polish, tokens, next_idx + 1, end)
    else:  # nparam == 2
        polish.append(tokens[next_idx])
        left_exists = _reorder_rec(polish, tokens, start, next_idx - 1)
        # ADD may have a single operand: insert a 0 if the left side is empty
        # (PolishNotationFunction.java:131-138).
        if tokens[next_idx].type == config.ADD and not left_exists:
            polish.append(_Token(config.VALUE, 0.0))
        _reorder_rec(polish, tokens, next_idx + 1, end)

    return True


def _reorder_tokens_to_polish(tokens: Sequence[_Token]) -> list[_Token]:
    """``reorderTokensToPolishNotation`` (PolishNotationFunction.java:67-76)."""
    polish: list[_Token] = []
    _reorder_rec(polish, tokens, 0, len(tokens) - 1)
    return polish


def _get_values_needed(function: Sequence[_Token]) -> int:
    """``getValuesNeeded`` (PolishNotationFunction.java:640-664)."""
    values_needed = 1
    n = len(function)
    for i, tok in enumerate(function):
        if config.is_operation(tok.type):
            values_needed += config.get_num_param(tok.type) - 1
        else:
            values_needed -= 1
        if values_needed == 0 and i + 1 < n:
            return -1
    return values_needed


# --- The public function object ---------------------------------------------


class PolishNotationFunction:
    """A parsed, evaluable ``y = f(x)`` expression.

    Construct from a string; malformed input raises :class:`MalformedFunction`.
    """

    __slots__ = ("_function",)

    def __init__(self, func_str: str) -> None:
        normal = _create_regular_notation_tokens(func_str)
        self._function: list[_Token] = _reorder_tokens_to_polish(normal)
        if _get_values_needed(self._function) != 0:
            raise MalformedFunction()

    def tokens(self) -> tuple[tuple[int, float | None], ...]:
        """The prefix (Polish) token list as ``(type, value)`` pairs (for
        tests/debug)."""
        return tuple((t.type, t.value) for t in self._function)

    def evaluate(self, var1: float, var2: float = 0.0, var3: float = 0.0) -> float:
        """Evaluate the function at ``(var1, var2, var3)``.

        Equivalent to ``evaluateFunction``/``evaluateRec``
        (PolishNotationFunction.java:968-1127). The token list is **prefix**
        (Polish) notation — operator first, then its operands — so this is a
        recursive-descent read over the list, exactly mirroring ``evaluateRec``.
        ``SUBTRACT`` is unary (negation); ``LOG`` is base-10, ``LN`` is natural.
        """
        tokens = self._function
        pos = 0  # readLocation

        def rec() -> float:
            nonlocal pos
            tok = tokens[pos]
            pos += 1
            t = tok.type
            if t == config.VARIABLE1:
                return var1
            if t == config.VARIABLE2:
                return var2
            if t == config.VARIABLE3:
                return var3
            if t == config.VALUE:
                return tok.value  # type: ignore[return-value]
            if t == config.ADD:
                return rec() + rec()
            if t == config.SUBTRACT:
                return -rec()
            if t == config.MULTIPLY:
                return rec() * rec()
            if t == config.DIVIDE:
                return _java_div(rec(), rec())
            if t == config.SQRT:
                return _java_sqrt(rec())
            if t == config.LOG:
                return _java_log10(rec())
            if t == config.ABS:
                return abs(rec())
            if t == config.SIN:
                return _java_sin(rec())
            if t == config.COS:
                return _java_cos(rec())
            if t == config.TAN:
                return _java_tan(rec())
            if t == config.POW:
                return _java_pow(rec(), rec())
            if t == config.LN:
                return _java_log(rec())
            # Brackets are consumed during reordering and never reach the
            # prefix list; reaching here would be a port bug.
            raise MalformedFunction()  # pragma: no cover

        return rec()
