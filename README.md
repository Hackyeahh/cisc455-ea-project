# Maze Designer EA

A Rust EA for the **Maze Designer** points game. The maze spends points on walls
and speed traps to slow an exploring robot. Coins give the maze more points, but
only when the robot picks them up, and each coin cuts the robot's time.

The robot is a step-for-step port of `simulate()` in the Maze Designer web page, so
a layout gets the same score here and in the browser. The page is
`maze-designer.html` in this folder: open it in a browser to play.

```
cargo build --release
target/release/maze_designer evolve 10 900 out/ea_best.txt         # threads, seconds, output
target/release/maze_designer eval layout.txt                        # score layouts (page text format)
```

Standard library only, no crates.

**How to read the justifications.** Each number and choice below has a reason
with one of three labels:

- **[measured]**: backed by a number from this project.
- **[reasoned]**: follows from the rules or from standard EA theory, but not tested here.
- **[first guess]**: chosen without evidence. A candidate for tuning.

## Game rules

| Rule | Value | Reason |
|---|---|---|
| Grid | 21 × 21 | [reasoned] Same as the original maze project. The shortest path is 40 steps, which leaves room for long detours. Small enough to see on one screen and to simulate about 120 times per second per thread |
| Start, goal | Opposite corners | [reasoned] This gives the longest possible shortest path. Side effect: on an empty map, every coin has a 0 s detour, so every coin is bait at first |
| Base budget | 120 pt | [first guess] At 1 pt per wall, this allows 27% of the tiles as walls. That is enough to build a hard maze but not enough to fill one |
| Wall | 1 pt | [reasoned] The unit of account. Every other price is relative to one wall |
| Trap | 3 pt, +5 s per hit | [first guess] Meant to cost about as much as 3 walls of delay. [measured] In the champion, traps give 4.4 s per point, and walls about 4.8 s per point if all steps are credited to walls. So the prices are close. But traps trigger on **every** hit, which makes them the strongest item in the right place (see Results) |
| Coin | +4 pt if taken, −6 s for the robot | [first guess] [measured] The coins look underpriced for the maze: a walls-only human design reaches 4.1 s per point on average. If that average also holds for the last few points, 4 pt buy about 16 s of delay against a 6 s bonus. This fits the EA's behaviour: its best maze places and collects all 6 coins |
| Most coins | 6 | [reasoned] A cap stops "coin farming": lining the route with coins the robot takes at no detour |
| No-trap radius | 2 tiles around S and G | [reasoned] Every route passes through one of the 2 tiles next to G, so 2 traps there would give a free guaranteed hit in every maze. With radius 2, the closest ring that every route must cross is at distance 3 and has 4 tiles, so this trick costs at least 4 traps (12 pt) for 5 s. Walls can still build a 1-tile corridor elsewhere and put one trap in it, but that choke point costs walls |
| Traps visible | When adjacent | [reasoned] A visible trap is a priced soft obstacle that the robot can route around. Hidden traps would bring back guaranteed hits at choke points |
| Step cap | 5,000 steps | [reasoned] A safety limit only. About 11 × the number of tiles, and about 10 × the longest run seen (490 steps) |

## Why an EA: tangled causality

Changing **one tile** changes what the robot sees. That changes its plan, which
changes which coins it takes and which traps it crosses. That in turn decides
whether the maze stays inside its budget. So the effect of a tile depends on
almost every other tile.

**[measured]** On the 632 s champion:

| Single-tile edit | Count | Neutral | Better | Change ≥ 50 s | Now infeasible |
|---|---|---|---|---|---|
| Remove one wall or trap | 128 | 0% | 0% | 24% | **38%** |
| Move one wall to an adjacent floor tile (same budget) | 299 | 4% | 1% | 23% | **33%** |

- **Removing a wall can break the budget.** The maze then spends 1 point *less*, but the robot's new route skips a coin, so the maze loses 4 points of budget and is over by 3. This happened in 38% of removals.
- After one removal, the robot's route differs for a median of **223 of its 459 steps**.
- **Non-additive effects:** for 572 pairs of wall moves, where each move and the pair are all feasible, 12% of pairs differ from the sum of the two single effects by 10 s or more. Most pairs do add up (median difference 0 s), and no pair flipped the sign. So the interactions are real but affect a minority of pairs.

**What this means for the choice of algorithm:**

| Problem property | Value | Consequence |
|---|---|---|
| Search space | 4^441 ≈ 10^265 layouts | Exhaustive search is impossible |
| Fitness | A simulation, no formula | No gradient. Black-box search only |
| Constraint | The budget depends on the robot's behaviour | Known only after the simulation, so a linear or integer program cannot state it directly |
| Interactions | One tile changes half the route; a third of edits break feasibility | A tile has no fixed value. Greedy tile-by-tile construction does not work |
| Landscape | Rugged: one edit can drop the score from 600 to infeasible | A single local search gets stuck. A population keeps several starting points |

A population with recombination can also combine parts from different parents,
such as a dead-end comb and a trap cluster. **[reasoned]** We have not yet compared
the EA with random search or a hill climber on the same number of evaluations.

## 1. Representation

| | Choice | Reason |
|---|---|---|
| Genotype | Flat array of 441 cells (21×21, row-major), each 0 floor, 1 wall, 2 trap, 3 coin | [reasoned] One cell = one tile, so every genotype maps to exactly one maze, and the page uses the same text format (`. # T $ S G`). An integer (categorical) representation, because the 4 item types have no order |
| Phenotype | The maze as the robot meets it, plus the robot's run | [reasoned] The fitness depends on the run, not on the layout alone |
| Fixed cells | S (index 0) and G (index 440) are always floor | [reasoned] The rules require them, so they are not searched |

The alternative, a list of item positions, makes "same tile twice" possible and
needs extra repair. The grid avoids that.

## 2. Initialization

| Choice | Value | Reason |
|---|---|---|
| Wall probability | Uniform 0.15–0.35, one value per individual | [reasoned] 120 pt allows 27% walls, so the range starts below and above the budget limit. Repair trims the dense ones. Different densities per individual give a diverse start |
| Traps | ~1% of tiles (~4 traps) | [first guess] Enough that traps exist from generation 0, few enough to stay inside the budget |
| Coins | ~1% of tiles (~4 coins) | [first guess] Same reason. Repair removes coins above 6 |

## 3. Fitness: the robot simulation

Fitness = robot **escape time = steps + trap seconds − 6 × coins taken**, from one full run.

| Robot part | Algorithm | Reason |
|---|---|---|
| Map | Unknown / floor / wall / trap. **Free-space assumption**: unknown tiles count as floor | [reasoned] The standard method for robots in unknown terrain, and the same robot as the original project |
| Sensing | Own tile + 4 neighbours each step | [reasoned] The smallest useful sensor. A larger range would make dead ends visible from outside |
| Planning | **Dijkstra**, replanned every step. Tile 1 s, known trap 1 + 5 s, known wall blocked | [reasoned] Traps make tile costs unequal, so plain BFS is not enough. Dijkstra finds the cheapest route on the robot's map |
| Coin choice | **Reverse Dijkstra** from G. Detour = d(robot, coin) + d(coin, G) − d(robot, G). Takes the smallest detour if it is under 6 s | [reasoned] This is the rational rule: a coin is worth exactly 6 s to the robot, so it adds no free parameter |
| Ties | Insertion order, the same as the page | [measured] Needed for exact parity: 116 of 116 test layouts score the same in the page and here |

Every evaluation runs two Dijkstras per step for 300–450 steps. **[measured]** In
Rust this gives about 1,200 evaluations per second on 10 threads. That speed is
why the EA is written in Rust.

## 4. Constraint handling: repair, then penalty

**Repair** runs on every new child, in this order:

| Step | Reason |
|---|---|
| 1. S and G become floor | [reasoned] Required by the rules |
| 2. Remove traps within 2 tiles of S or G | [reasoned] Required by the rules |
| 3. Remove random coins until at most 6 | [reasoned] Required by the rules. Random choice adds no bias |
| 4. Remove random walls or traps until spent ≤ 120 + 4 × coins placed | [reasoned] This is the largest budget the maze could ever earn, so any maze above it is certainly infeasible. It can be checked without a simulation |

The repaired genome replaces the original (**Lamarckian** repair). [reasoned] The
child then carries what was evaluated, and later children do not repeat the same
violation.

Whether the robot takes enough coins is only known after the simulation, so
repair cannot handle it. A **penalty** handles it instead. Every feasible maze
ranks above every infeasible one:

| Case | Fitness | Reason |
|---|---|---|
| Feasible | Escape time (always > 0) | [reasoned] The objective itself |
| Over budget | −overspend (−1 to −24) | [reasoned] Graded: 2 pt over is closer to feasible than 20 pt over, so the search gets a direction. −24 is the most possible overspend (144 − 120) |
| No path S → G | −1000 − 10 × walls to remove (0-1 BFS) | [reasoned] Far below any overspend, so a blocked maze never beats an over-budget one. Still graded by how many walls block the path |
| Robot does not finish | −2000 | [reasoned] A safety case. It did not occur in the 116 parity layouts |

The constants −1000, −10 and −2000 only set the order of the cases. Their sizes
have no other effect, because tournament selection uses ranks, not values.

## 5. Variation operators

| Operator | Value | Reason |
|---|---|---|
| **2-point crossover** | Always applied. 2 uniform cut points in the 441-cell array; the child is parent A with parent B's segment between the cuts | [reasoned] Works for any symbol and keeps runs of neighbouring cells together, so a row of corridor or a cluster of traps can pass to a child intact. Limitation: on a row-major array, the child inherits **horizontal bands**, so vertical structure is cut |
| **Symbol mutation** | Each cell with p = 4/441 | [reasoned] About 4 changes per child. Only 1.8% of children get no symbol mutation. The textbook default 1/441 (one change per child) is likely too weak here, because repair can undo a change when the budget is full. [first guess] The value 4 itself is not tuned |
| Symbol weights | Floor 0.4, wall 0.4, trap 0.1, coin 0.1 (never the current symbol) | [reasoned] Traps and coins are rare in good mazes (the champion: 96 walls, 16 traps, 6 coins), and a trap costs 3 times as much as a wall. Equal weights would add too many traps and coins, and repair would delete them again |
| **Move mutation** | With p = 0.3, pick one coin or trap and a random tile. If that tile is not a coin or trap, the two swap. Otherwise nothing changes | See the next section. [first guess] The value 0.3 is not tuned |

### Symbol mutation vs. move mutation

The two operators make different kinds of change, and the EA needs both.

| | Symbol mutation | Move mutation |
|---|---|---|
| What it changes | **What** is bought: the mix of walls, traps and coins | **Where** an item is: the same items in new places |
| Effect on the budget | Usually changes it | None, except when the item lands where repair removes it (see below) |
| Scale | ~4 random tiles anywhere | 1 coin or trap, which can jump across the whole maze |
| Main use | Exploring the item mix early in the run | Fine placement once the budget is full |

**Why move mutation matters when the budget is full.** The champion spends 144 of
144 points, and late in a run most good mazes do. In a full maze, a symbol
mutation that adds a wall puts the maze over budget. Repair then deletes a
**random** wall or trap somewhere else, which may be an important one. So one
"small" symbol mutation becomes two changes, and the second one is random.

A move mutation avoids this. It relocates one item without changing what was
bought, so repair has nothing to undo. It works on coins and traps because those
are the items whose **placement** decides the score: a trap is worth 0 s if the
robot never crosses it, and about 2.6 hits in the champion's trap cluster.

Two side effects:
- When a trap swaps with a wall, the wall moves too, which is also budget-neutral.
- An item that lands on S or G, or a trap that lands in the no-trap zone, is removed by repair. The maze then has spare points. A coin removed this way also lowers the budget, so repair may then delete a random wall or trap.

**[reasoned]** We have not run the EA without move mutation. An ablation (p = 0
vs. 0.3) would test whether it matters as much as the argument above says.

## 6. Selection and population management

| Part | Value | Reason |
|---|---|---|
| **Parent selection** | Tournament, k = 3, contestants drawn **with replacement** | [reasoned] Uses only ranks, so negative penalty fitness works, which roulette wheel cannot handle. [measured] In the original project, tournament (k = 3) beat roulette wheel on every seed |
| Tournaments | 2 per child: **60 per generation per island**, 600 per generation in total, about 218,000 per island in the 900 s run | Follows from 30 children × 2 parents |
| Selection pressure | P(the best individual wins a given tournament) = 1 − (59/60)³ = 4.9%, so it is a parent about **2.95 times** per generation | [reasoned] Moderate. k = 2 gives about 2.0, and k = 5 about 4.8, which takes over a 60-individual island faster |
| **Survivor selection** | Steady state: 30 children replace the 30 worst of 60 parents, always | [reasoned] The best 30 always survive (elitist), so the best maze is never lost. Children enter even when they are worse than the parent they replace, which keeps some diversity |
| Generation gap | 30 / 60 = 50% | [first guess] Between a classic steady state (1–2 children per generation, fast takeover) and a generational GA (100%, no elitism) |
| Island size | 60 | [reasoned] Large enough for diversity, small enough that each island runs about 3,600 generations in 15 minutes |
| **Islands** | 10, one per thread | [reasoned] 75% of the 14 logical cores. One island per thread means the threads share nothing in the inner loop, so they never wait for each other |
| **Migration** | Every 50 generations, best individual to the next island in a ring; it replaces the receiver's worst | [reasoned] Each island gets about 72 migrations in a run. In a 10-island ring, a good maze needs at least 9 hops (450 generations, about 12% of the run) to reach every island. This spreads good parts slowly, so the islands do not all converge to one design early. Locking only every 50 generations also keeps thread contention near zero |
| **Termination** | 900 s of wall-clock time | [first guess] Fits a working session. Limitation: the run is **not reproducible**, because migration timing depends on thread speed. A fixed evaluation count would fix this |

## Verification

- **Parity with the page [measured]:** 116 random layouts (0–5 coins taken, a quarter with hidden traps), scored by the page's JavaScript in Node and by `maze_designer eval`. All 116 results are identical.
- The page also scores the champion at 632 s.
- `eval` accepts files with Windows (CRLF) line endings.

## Results (one 900 s run, 10 threads)

| Time | Evaluations | Best escape time |
|---|---|---|
| 90 s | 128k | 460 s |
| 270 s | 363k | 547 s |
| 540 s | 681k | 602 s |
| 900 s | 1.09M | **632 s** |

The score was still rising slowly at the end. Champion: `out/ea_best.txt`.
Log: `out/ea_run_log.txt`.

| Design | Escape time | Steps | Trap seconds | Coins |
|---|---|---|---|---|
| Human: comb of dead ends + toll row | 325 s | 286 | +75 | 6 / 6 |
| Human: diagonal lattice, walls only (`diagonal.txt`) | 490 s | 490 | 0 | 0 |
| **EA champion** | **632 s** | 458 | +210 | 6 / 6 |

**What the champion does:**
- 42 trap hits (+210 s) from 16 traps, about 2.6 hits per trap. Most traps form a cluster in the middle of the maze.
- 3 coins sit in or next to the trap cluster, so each one pulls the robot through the traps again.
- 3 coins are spread far apart and drag the robot across the whole maze.
- The robot takes all 6 coins, so the maze can spend 144 points. The coin bonus costs the maze 36 s.

**Findings:**
- **Toll plaza.** Traps trigger every time, so the strongest use of a trap is a tile the robot crosses many times, with coins as bait to make it cross again.
- **The human wins on walls.** The diagonal design makes the robot walk 490 steps with walls only. The champion's walls give only 458 steps. The EA wins through the trap rule, not through better wall structure. A possible cause is the 1-D crossover, which cuts diagonal and vertical structure.

## Weak points (for Q&A)

- **No baseline algorithm yet.** Random search and a hill climber with the same evaluation budget are missing.
- **One run, not reproducible.** Wall-clock termination and asynchronous migration. Needs a fixed evaluation count and several seeds.
- **1-D crossover on a 2-D maze.** Bands across rows keep horizontal structure but cut vertical structure. A 2-D block crossover is the natural alternative.
- **Untuned parameters.** Marked [first guess] above.
- **Overfitting to one robot.** The fitness is exact (the robot is deterministic), but the maze exploits this robot's rules and tie-breaks.
- **Open rule question.** Should a trap trigger only once, or should the robot avoid traps it has already hit? Either change removes the toll-plaza exploit.
