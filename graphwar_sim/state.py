"""Game state, turn order, and win rule for the headless Graphwar simulator.

A clean-room port of the turn/turn-advancement/win logic in the reference.
The reference is network-driven (the server broadcasts turn changes), so the
*logic* lives in the client; we port the client-side rules:

- Turn advancement: ``GameData.nextTurnMessage`` (GameData.java:873-892) —
  advance ``currentTurn`` one player at a time, skipping players with no living
  soldier, until one is found.
- Per-player soldier cycling: ``Player.nextTurn`` (Player.java:172-185) —
  advance the player's ``currentTurnSoldier`` to the next living soldier.
- Win rule: ``GameData.checkGameFinished`` (GameData.java:512-545) — the game
  ends when either team has no living soldier.
- Hit application: ``GameData.processFunction`` (GameData.java:1102-1111) —
  each hit soldier is marked dead.

The reference is GPL-licensed; per the project's clean-room rule we cite the
source rather than copy its code.

Map generation (terrain circles + soldier placement) is **not** in the source —
the reference places soldiers interactively and the server generates circles.
The generator here is a deterministic, seeded stand-in so matches are
reproducible; its *distributions* reuse the reference constants (labeled), but
the specific positions are ``# TUNABLE`` and are not from the source.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from . import config
from .parser import PolishNotationFunction
from .physics import Obstacle, ShotResult, Soldier, process_function_range


@dataclass
class Team:
    """One player on a given team (the reference's ``Player``).

    A ``Team`` holds that player's soldiers and which one is the current-turn
    soldier. ``team`` is the side (``config.TEAM1`` / ``config.TEAM2``); several
    ``Team`` objects may share a side, and the win rule is evaluated per side.
    """

    name: str
    team: int
    soldiers: list[Soldier] = field(default_factory=list)
    current_turn_soldier: int = 0

    def current_soldier(self) -> Soldier:
        """The soldier that fires on this player's turn."""
        return self.soldiers[self.current_turn_soldier]

    def num_alive(self) -> int:
        return sum(1 for s in self.soldiers if s.alive)

    def next_turn(self) -> bool:
        """Port of ``Player.nextTurn`` (Player.java:172-185).

        Advance ``current_turn_soldier`` to the next living soldier (cycling).
        Returns ``True`` if a living soldier was found (it is now the
        current-turn soldier), ``False`` if the player has none left.
        """
        n = len(self.soldiers)
        for _ in range(n):
            self.current_turn_soldier = (self.current_turn_soldier + 1) % n
            if self.soldiers[self.current_turn_soldier].alive:
                return True
        return False


@dataclass
class GameState:
    """The roster of players, the current turn, and the turn/win rules.

    Faithful to ``GameData``'s client-side turn bookkeeping. ``teams`` is the
    ``players`` list; ``current_turn`` indexes into it (GameData.java:49, 208).
    """

    teams: list[Team] = field(default_factory=list)
    current_turn: int = -1

    def current_team(self) -> Team:
        """``getCurrentTurnPlayer`` (GameData.java:206-208)."""
        return self.teams[self.current_turn]

    def _team_alive(self, team_id: int) -> bool:
        return any(
            s.alive for t in self.teams if t.team == team_id for s in t.soldiers
        )

    def check_game_finished(self) -> bool:
        """Port of ``checkGameFinished`` (GameData.java:512-545).

        The game is finished when either side has no living soldier.
        """
        team1_alive = self._team_alive(config.TEAM1)
        team2_alive = self._team_alive(config.TEAM2)
        return not (team1_alive and team2_alive)

    def advance_turn(self) -> None:
        """Port of the advancement loop in ``nextTurnMessage``
        (GameData.java:881-892).

        If the game just finished, do nothing. Otherwise step ``current_turn``
        forward one player at a time, calling each candidate's ``next_turn``
        (which advances its current-turn soldier), until a player with a living
        soldier is found.
        """
        if self.check_game_finished():
            return
        n = len(self.teams)
        for _ in range(n):
            self.current_turn = (self.current_turn + 1) % n
            if self.teams[self.current_turn].next_turn():
                break

    def winner(self) -> int | None:
        """The winning side (``TEAM1``/``TEAM2``), or ``None`` if not over (or
        both sides dead simultaneously)."""
        if not self.check_game_finished():
            return None
        team1_alive = self._team_alive(config.TEAM1)
        team2_alive = self._team_alive(config.TEAM2)
        if team1_alive and not team2_alive:
            return config.TEAM1
        if team2_alive and not team1_alive:
            return config.TEAM2
        return None


def make_circle_obstacle(
    circles: Sequence[tuple[int, int, int]],
) -> Obstacle:
    """Build an :class:`Obstacle` from filled circles ``(cx, cy, radius)``.

    A pixel is terrain if it lies within ``radius`` of a circle centre
    (Euclidean, inclusive). Out-of-bounds pixels collide (``Obstacle.collidePoint``
    returns ``true`` for OOB, Obstacle.java:99-103).

    Note: the reference fills *anti-aliased* ovals (Obstacle.java:39-56), so its
    exact edge pixels differ from this crisp model. Golden tests therefore use
    the byte-identical grid dumped by ``GoldenShot`` (see ``tests/golden``); this
    model is for the simulator's own self-consistent maps.
    """
    height = config.PLANE_HEIGHT
    length = config.PLANE_LENGTH
    grid = [[False] * length for _ in range(height)]
    for cx, cy, r in circles:
        r2 = r * r
        y0, y1 = max(0, cy - r), min(height - 1, cy + r)
        x0, x1 = max(0, cx - r), min(length - 1, cx + r)
        for y in range(y0, y1 + 1):
            dy = y - cy
            for x in range(x0, x1 + 1):
                dx = x - cx
                if dx * dx + dy * dy <= r2:
                    grid[y][x] = True

    def collide_point(x: int, y: int) -> bool:
        if x < 0 or x >= length or y < 0 or y >= height:
            return True
        return grid[y][x]

    return Obstacle(collide_point=collide_point)


class Game:
    """A headless Graphwar match: seeded map, terrain, turn order, win rule.

    The physics of each shot is delegated to :func:`process_function_range`
    (the faithful port of ``Function.processFunctionRange``). ``Game`` adds the
    match layer: who fires, in what order, and when the game ends.
    """

    def __init__(self, state: GameState, terrain: Obstacle) -> None:
        self.state = state
        self.terrain = terrain

    # -- construction -------------------------------------------------------

    @classmethod
    def create(
        cls,
        seed: int,
        num_teams: int = 2,
        num_soldiers: int = config.INITIAL_NUM_SOLDIERS,
        num_circles: int | None = None,
    ) -> Game:
        """Create a deterministic match from ``seed``.

        Terrain and soldier positions are generated from ``seed`` (see
        :func:`generate_map`). ``num_circles`` defaults to a draw from the
        reference's circle-count distribution (Obstacle.java:70-77).
        """
        rng = random.Random(seed)
        circles, soldiers_by_team = generate_map(rng, num_teams, num_soldiers, num_circles)
        terrain = make_circle_obstacle(circles)
        return cls(state=GameState(teams=soldiers_by_team, current_turn=0), terrain=terrain)

    # -- queries ------------------------------------------------------------

    def all_soldiers(self) -> list[Soldier]:
        """Flat list of every soldier with correct ``player_index`` /
        ``soldier_index`` (the (j, k) hit identity used by the physics)."""
        out: list[Soldier] = []
        for j, team in enumerate(self.state.teams):
            for k, s in enumerate(team.soldiers):
                s.player_index = j
                s.soldier_index = k
                out.append(s)
        return out

    def finished(self) -> bool:
        return self.state.check_game_finished()

    def winner(self) -> int | None:
        return self.state.winner()

    # -- actions ------------------------------------------------------------

    def fire(self, func_str: str) -> ShotResult:
        """Fire the current-turn soldier with function ``y = f(x)``.

        Port of ``GameData.processFunction`` (GameData.java:1063-1114) for
        NORMAL_FUNC: the TEAM2 shooter fires ``inverted`` (mirrored); each hit
        soldier is marked dead. Does **not** advance the turn — call
        :meth:`end_turn` (or :meth:`play_turn`) to move on.
        """
        team = self.state.current_team()
        shooter = team.current_soldier()
        f = PolishNotationFunction(func_str)
        inverted = team.team == config.TEAM2
        result = process_function_range(
            f, shooter, self.all_soldiers(), self.terrain, inverted
        )
        # Apply kills (GameData.java:1102-1111).
        for player_index, soldier_index, _pos in result.hits:
            self.state.teams[player_index].soldiers[soldier_index].alive = False
        return result

    def play_turn(self, func_str: str) -> ShotResult:
        """Fire and then advance to the next turn (one full turn)."""
        result = self.fire(func_str)
        self.state.advance_turn()
        return result


def generate_map(
    rng: random.Random,
    num_teams: int,
    num_soldiers: int,
    num_circles: int | None = None,
) -> tuple[list[tuple[int, int, int]], list[Team]]:
    """Generate terrain circles and one ``Team`` per side with placed soldiers.

    Returns ``(circles, teams)``. The circle *count* and *radius* distributions
    reuse the reference constants (Obstacle.java:70-77, Constants.java:66-69);
    circle *centres* and soldier *positions* are ``# TUNABLE`` (not in source).
    Soldiers are placed on their side's half of the plane, avoiding terrain and
    each other (Chebyshev spacing, Constants.java / GraphServer.java:687).
    """
    length = config.PLANE_LENGTH
    height = config.PLANE_HEIGHT

    # --- Terrain circles ----------------------------------------------------
    if num_circles is None:
        num_circles = _draw_num_circles(rng)
    circles: list[tuple[int, int, int]] = []
    for _ in range(num_circles):
        radius = _draw_circle_radius(rng)
        cx = rng.randint(0, length - 1)
        cy = rng.randint(0, height - 1)
        circles.append((cx, cy, radius))

    terrain = make_circle_obstacle(circles)

    # --- Soldiers -----------------------------------------------------------
    # Team1 on the left third, Team2 on the right third. # TUNABLE layout.
    teams: list[Team] = []
    for t in range(num_teams):
        team_id = config.TEAM1 if t % 2 == 0 else config.TEAM2
        soldiers: list[Soldier] = []
        for _ in range(num_soldiers):
            x, y = _place_soldier(rng, terrain, team_id, soldiers)
            soldiers.append(Soldier(x=float(x), y=float(y), alive=True))
        teams.append(Team(name=f"team{t}", team=team_id, soldiers=soldiers))

    return circles, teams


def _draw_num_circles(rng: random.Random) -> int:
    """Circle count from the reference distribution (Obstacle.java:70-77)."""
    n = int(
        rng.gauss(config.NUM_CIRCLES_MEAN_VALUE, config.NUM_CIRCLES_STANDARD_DEVIATION)
    )
    while n < 0:
        n = int(
            rng.gauss(config.NUM_CIRCLES_MEAN_VALUE, config.NUM_CIRCLES_STANDARD_DEVIATION)
        )
    return n


def _draw_circle_radius(rng: random.Random) -> int:
    """Circle radius from the reference distribution (Constants.java:66-67)."""
    r = int(
        rng.gauss(config.CIRCLE_MEAN_RADIUS, config.CIRCLE_STANDARD_DEVIATION)
    )
    while r < 1:
        r = int(rng.gauss(config.CIRCLE_MEAN_RADIUS, config.CIRCLE_STANDARD_DEVIATION))
    return r


def _place_soldier(
    rng: random.Random,
    terrain: Obstacle,
    team_id: int,
    placed: Sequence[Soldier],
) -> tuple[int, int]:
    """Pick a legal soldier position: on the team's side, off-terrain, spaced.

    ``# TUNABLE`` — the reference places soldiers by player click; this is a
    deterministic stand-in. Uses the reference's Chebyshev spacing
    (GraphServer.java:687) and the plane bounds.
    """
    length = config.PLANE_LENGTH
    height = config.PLANE_HEIGHT
    radius = config.SOLDIER_RADIUS
    min_gap = config.SOLDIER_MIN_CHEBYSHEV

    # Left side for TEAM1, right side for TEAM2. # TUNABLE thirds.
    if team_id == config.TEAM1:
        x_lo, x_hi = radius, length // 3 - radius
    else:
        x_lo, x_hi = 2 * length // 3 + radius, length - radius - 1

    for _ in range(10000):
        x = rng.randint(x_lo, x_hi)
        y = rng.randint(radius, height - radius - 1)
        if terrain.collide_point(x, y):
            continue
        # 5-point clearance like Obstacle.soldierCollides (Obstacle.java:123-161).
        if any(
            terrain.collide_point(x + dx, y + dy)
            for dx, dy in ((0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius))
        ):
            continue
        if any(
            abs(x - p.x) < min_gap or abs(y - p.y) < min_gap for p in placed
        ):
            continue
        return x, y

    # Fallback: relax spacing if the side is crowded (shouldn't happen).
    for _ in range(10000):
        x = rng.randint(x_lo, x_hi)
        y = rng.randint(radius, height - radius - 1)
        if not terrain.collide_point(x, y):
            return x, y
    raise RuntimeError("could not place a soldier")
