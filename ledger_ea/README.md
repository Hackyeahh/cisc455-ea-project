# Maze Ledger EA

A Rust EA for the **Maze Ledger** points game. The maze spends points on walls
and speed traps to slow an exploring robot. Coins give the maze more points, but
only when the robot picks them up, and each coin cuts the robot's time.

The robot is a step-for-step port of `simulate()` in the Maze Ledger web page, so
a layout gets the same score here and in the browser. The page is
`maze-ledger.html` in this folder: open it in a browser to play.

```
cargo build --release
target/release/ledger_ea evolve 10 900 ../out/ledger/ea_best.txt   # threads, seconds, output
target/release/ledger_ea eval layout.txt                            # score one layout (page text format)
```

Standard library only, no crates.

## Game rules (defaults)

| Item | Cost to the maze | Effect on the robot |
|---|---|---|
| Wall | 1 pt | Blocks the tile |
| Speed trap | 3 pt | +5 s **each time** the robot steps on it |
| Coin | +4 pt, only if the robot takes it | −6 s from the robot's escape time |

- Base budget 120 pt. At most 6 coins. No traps within 2 tiles of S or G.
- A path from S (top-left) to G (bottom-right) must exist.
- Valid maze: points spent ≤ 120 + 4 × coins the robot took.

## 1. Representation

| | What it is |
|---|---|
| **Genotype** | Flat array of 441 cells (21×21, row-major). Each cell: 0 floor, 1 wall, 2 trap, 3 coin |
| **Phenotype** | The maze as the robot meets it, plus the robot's run through it. Fitness comes from that run |
| **Fixed cells** | S (index 0) and G (index 440) are always floor |

Each cell maps directly to one tile, and the page uses the same text format
(`. # T $ S G`). This is an integer (categorical) representation. The earlier
mazes in this repo use a binary or ternary grid.

## 2. Initialization

Each individual starts as a random grid:
- One wall probability per individual, uniform in 0.15–0.35, so the population starts with a range of densities.
- About 1% traps and 1% coins.
- Then repair (section 4).

## 3. Fitness: the robot simulation

Fitness = robot **escape time = steps + trap seconds − 6 × coins taken**, from a full run.

| Robot part | Algorithm |
|---|---|
| Map | Its own map: unknown / floor / wall / trap. **Free-space assumption**: unknown tiles count as floor |
| Sensing | Each step it reads its own tile and its 4 neighbours |
| Planning | **Dijkstra** on its map. Enter a tile: 1 s. Known trap: 1 + 5 s. Known wall: blocked. Replans every step |
| Coin choice | **Reverse Dijkstra** from G. Detour = d(robot, coin) + d(coin, G) − d(robot, G). Takes the coin with the smallest detour if it is under 6 s, otherwise heads for G |
| Ties | Broken in insertion order, the same as the page, so scores match exactly |

Cost: two Dijkstras per step on 441 tiles, over 300–450 steps. That is why this
runs in Rust: about 1,200 evaluations per second on 10 threads.

## 4. Constraint handling: repair, then penalty

**Repair** (every new child, in order):
1. S and G become floor.
2. Remove traps within 2 tiles of S or G.
3. Remove random coins until at most 6.
4. Remove random walls or traps until spent ≤ 120 + 4 × coins placed (the budget if every coin is taken).

The repaired genome replaces the original (Lamarckian repair).

**Penalty ranking** (every feasible maze ranks above every infeasible one):

| Case | Fitness |
|---|---|
| Feasible | Escape time (positive) |
| Over budget (robot skipped coins) | −overspend |
| Robot does not finish (5,000-step cap) | −2000 |
| No path S → G | −1000 − 10 × walls to remove (0-1 BFS) |

Graded penalties give the search a direction: 2 points over budget is closer to
feasible than 20 points over.

## 5. Variation operators

| Operator | What it does | Why |
|---|---|---|
| **2-point crossover** | 2 cut points in the 441-cell array. Child = parent A, with parent B's segment between the cuts | Valid for any symbol. Children inherit horizontal bands |
| **Symbol mutation** | Each cell changes with p = 4/441. New symbol weights: floor 0.4, wall 0.4, trap 0.1, coin 0.1 (never the current symbol) | Traps and coins are rare and expensive. Random flips should not flood the maze with them |
| **Move mutation** | With p = 0.3, one coin or trap swaps with a random tile that is not a coin or trap | Moves an item without changing what was bought, so the budget stays the same |

## 6. Selection and population management

| Part | Choice |
|---|---|
| **Parent selection** | Tournament, k = 3. Works with negative penalty fitness (roulette wheel cannot), and beat roulette wheel in the original experiments |
| **Survivor selection** | Steady state: 30 children replace the 30 worst of 60 parents. The best 30 always survive (elitist) |
| **Population model** | Island model: 10 islands, one per thread, 60 individuals each |
| **Migration** | Every 50 generations, each island sends its best to the next island in a ring and replaces its worst with the migrant it receives |
| **Termination** | Wall-clock time |

Islands search in parallel and keep their own diversity. Migration spreads good
building blocks without letting one design take over all islands at once.

## 7. Parameters

| Parameter | Value |
|---|---|
| Islands × population | 10 × 60 |
| Children per generation | 30 |
| Tournament size | 3 |
| Mutation | 4/441 per cell, plus a move with p = 0.3 |
| Migration | Every 50 generations, ring |
| Run length | 900 s, about 1.09 million evaluations |

First choices, not tuned.

## Verification

- **Parity with the page:** 116 random layouts (0–5 coins taken, a quarter with hidden traps), scored by the page's JavaScript in Node and by `ledger_ea eval`. All 116 results are identical.
- The page also scores the champion at 632 s.

## Results (one 900 s run, 10 threads)

| Time | Evaluations | Best escape time |
|---|---|---|
| 90 s | 128k | 460 s |
| 270 s | 363k | 547 s |
| 540 s | 681k | 602 s |
| 900 s | 1.09M | **632 s** |

A human design (a comb of dead-end corridors with a toll row of traps) scored
325 s. Champion: `out/ledger/ea_best.txt`. Log: `out/ledger/ea_run_log.txt`.

**What the champion does:**
- 42 trap hits (+210 s) from only 13 traps, about 3 hits per trap. Most traps form a cluster in the middle of the maze.
- 3 coins sit in or next to the trap cluster, so each one pulls the robot through the traps again.
- 3 coins are spread far apart and drag the robot across the whole maze.
- The robot takes all 6 coins, so the maze can spend 144 points. The coin bonus costs the maze 36 s.

**Finding:** traps trigger every time, so the dominant strategy is a "toll plaza":
traps on tiles the robot crosses many times, with coins as bait to make it cross
again. A trap crossed 4 times buys 20 s for 3 points.

## Weak points (for Q&A)

- **1-D crossover on a 2-D maze.** Bands across rows keep horizontal structure but cut vertical structure. A 2-D block crossover is the natural alternative.
- **Rugged fitness landscape.** One changed tile can make the robot skip a coin. The maze then goes over budget, and its fitness drops from about 600 to a negative value.
- **Overfitting to one robot.** The fitness is exact (the robot is deterministic), but the maze exploits this robot's rules and tie-breaks.
- **Open rule question.** Should a trap trigger only once, or should the robot avoid traps it has already hit? Either change removes the toll-plaza exploit.
