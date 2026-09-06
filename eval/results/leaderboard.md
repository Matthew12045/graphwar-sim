# Graphwar agent leaderboard

- config: root_seed=1000, matches/pair=5 x 2 sides (30 matches), num_soldiers=2, max_turns=100
- roster: solver, random, straight
- seeds file: `seeds.json` — rerun via `python3 -m eval --from-seeds <file>`
- metrics reported: win rate + hit rate only (see IMPLEMENTATION_PLAN.md Phase 4 minimal slice); counters below are the raw inputs to those rates plus the M3 parse-failure logging.

## Win rate & hit rate

| Agent | Matches | Wins | Win rate | Shots | Hit rate | Kills | Friendly fire | Parse failures | Retries |
|---|---|---|---|---|---|---|---|---|---|
| solver | 20 | 14 | 0.700 | 322 | 0.078 | 31 | 0 | 0 | 0 |
| random | 20 | 0 | 0.000 | 466 | 0.002 | 1 | 2 | 0 | 0 |
| straight | 20 | 4 | 0.200 | 468 | 0.032 | 15 | 0 | 0 | 0 |

## Head-to-head

| Pair | A wins | B wins | Draws |
|---|---|---|---|
| solver vs random | 7 | 0 | 3 |
| solver vs straight | 7 | 0 | 3 |
| random vs straight | 0 | 4 | 6 |

## Solver degradation rungs (informational)

- solver: dud=297, per_target_gaussian=7, parabola=2, arc=12, line=3, fixed_grid_gaussian=1

## Per-match log

| # | Seed | TEAM1 | TEAM2 | Winner | Turns | TEAM1 shots | TEAM1 hits | TEAM2 shots | TEAM2 hits | Plot |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | solver | random | draw | 100 | 50 | 0 | 50 | 0 | [plot](plots/match_000_seed1000.png) |
| 1 | 1001 | random | solver | TEAM2 | 2 | 1 | 0 | 1 | 1 | [plot](plots/match_001_seed1001.png) |
| 2 | 1002 | solver | random | TEAM1 | 3 | 2 | 2 | 1 | 0 | [plot](plots/match_002_seed1002.png) |
| 3 | 1003 | random | solver | TEAM2 | 4 | 2 | 0 | 2 | 2 | [plot](plots/match_003_seed1003.png) |
| 4 | 1004 | solver | random | draw | 100 | 50 | 0 | 50 | 0 | [plot](plots/match_004_seed1004.png) |
| 5 | 1005 | random | solver | TEAM2 | 2 | 1 | 0 | 1 | 1 | [plot](plots/match_005_seed1005.png) |
| 6 | 1006 | solver | random | TEAM1 | 3 | 2 | 2 | 1 | 0 | [plot](plots/match_006_seed1006.png) |
| 7 | 1007 | random | solver | draw | 100 | 50 | 0 | 50 | 1 | [plot](plots/match_007_seed1007.png) |
| 8 | 1008 | solver | random | TEAM1 | 1 | 1 | 1 | 0 | 0 | [plot](plots/match_008_seed1008.png) |
| 9 | 1009 | random | solver | TEAM2 | 4 | 2 | 0 | 2 | 2 | [plot](plots/match_009_seed1009.png) |
| 10 | 2000 | solver | straight | draw | 100 | 50 | 0 | 50 | 1 | [plot](plots/match_010_seed2000.png) |
| 11 | 2001 | straight | solver | TEAM2 | 2 | 1 | 1 | 1 | 1 | [plot](plots/match_011_seed2001.png) |
| 12 | 2002 | solver | straight | TEAM1 | 3 | 2 | 2 | 1 | 0 | [plot](plots/match_012_seed2002.png) |
| 13 | 2003 | straight | solver | TEAM2 | 4 | 2 | 0 | 2 | 2 | [plot](plots/match_013_seed2003.png) |
| 14 | 2004 | solver | straight | TEAM1 | 3 | 2 | 2 | 1 | 1 | [plot](plots/match_014_seed2004.png) |
| 15 | 2005 | straight | solver | TEAM2 | 2 | 1 | 0 | 1 | 1 | [plot](plots/match_015_seed2005.png) |
| 16 | 2006 | solver | straight | TEAM1 | 1 | 1 | 1 | 0 | 0 | [plot](plots/match_016_seed2006.png) |
| 17 | 2007 | straight | solver | TEAM2 | 4 | 2 | 1 | 2 | 2 | [plot](plots/match_017_seed2007.png) |
| 18 | 2008 | solver | straight | draw | 100 | 50 | 1 | 50 | 0 | [plot](plots/match_018_seed2008.png) |
| 19 | 2009 | straight | solver | draw | 100 | 50 | 1 | 50 | 1 | [plot](plots/match_019_seed2009.png) |
| 20 | 3000 | random | straight | TEAM2 | 4 | 2 | 0 | 2 | 2 | [plot](plots/match_020_seed3000.png) |
| 21 | 3001 | straight | random | draw | 100 | 50 | 0 | 50 | 0 | [plot](plots/match_021_seed3001.png) |
| 22 | 3002 | random | straight | draw | 100 | 50 | 0 | 50 | 0 | [plot](plots/match_022_seed3002.png) |
| 23 | 3003 | straight | random | TEAM1 | 3 | 2 | 2 | 1 | 0 | [plot](plots/match_023_seed3003.png) |
| 24 | 3004 | random | straight | TEAM2 | 6 | 3 | 0 | 3 | 2 | [plot](plots/match_024_seed3004.png) |
| 25 | 3005 | straight | random | draw | 100 | 50 | 1 | 50 | 1 | [plot](plots/match_025_seed3005.png) |
| 26 | 3006 | random | straight | draw | 100 | 50 | 0 | 50 | 0 | [plot](plots/match_026_seed3006.png) |
| 27 | 3007 | straight | random | TEAM1 | 5 | 3 | 2 | 2 | 0 | [plot](plots/match_027_seed3007.png) |
| 28 | 3008 | random | straight | draw | 100 | 50 | 0 | 50 | 0 | [plot](plots/match_028_seed3008.png) |
| 29 | 3009 | straight | random | draw | 100 | 50 | 1 | 50 | 0 | [plot](plots/match_029_seed3009.png) |
