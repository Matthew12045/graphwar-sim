"""CLI for the M4 evaluation harness.

Examples
--------
Run a fresh leaderboard from a root seed:

    python3 -m eval --root 1000 --matches 5 --out eval/results

Reproduce a committed leaderboard **from its seed file alone**:

    python3 -m eval --from-seeds eval/results/seeds.json --out eval/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import DEFAULT_ROSTER, MatchConfig, run_from_seed_file, run_leaderboard

_DEFAULT_OUT: Path = Path(__file__).resolve().parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, default=1000, help="root seed")
    parser.add_argument("--matches", type=int, default=5, help="matches per pair (x2 sides)")
    parser.add_argument("--soldiers", type=int, default=2, help="soldiers per team")
    parser.add_argument("--turns", type=int, default=100, help="max turns per match (draw cap)")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="output directory")
    parser.add_argument(
        "--from-seeds",
        type=Path,
        default=None,
        help="reproduce a leaderboard from an existing seeds.json instead of deriving a plan",
    )
    args = parser.parse_args()

    if args.from_seeds is not None:
        leaderboard = run_from_seed_file(args.from_seeds, args.out)
        print(f"reproduced {len(leaderboard.matches)} matches -> {leaderboard.leaderboard_file}")
        return

    leaderboard = run_leaderboard(
        root_seed=args.root,
        roster=DEFAULT_ROSTER,
        n_matches=args.matches,
        config=MatchConfig(num_soldiers=args.soldiers, max_turns=args.turns),
        out_dir=args.out,
    )
    print(f"played {len(leaderboard.matches)} matches -> {leaderboard.leaderboard_file}")


if __name__ == "__main__":
    main()
