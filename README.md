# Maze evolution demo: slow down an exploring robot

A prototype for the CISC455 group project. An EA designs mazes. A fixed robot
tries to cross each maze from the top-left to the bottom-right corner. The EA
wants mazes that make the robot do as much physical work as possible.

**The robot** does not know the maze. It assumes unseen tiles are open, plans a
route to the goal with A*, walks one step, and inspects the 4 tiles next to it.
When it sees a wall on its planned route, it replans. This is the standard
"free-space assumption" approach for robots in unknown terrain.

**Framing:** Robot navigation in unknown terrain, or tower defense against a
pathfinding enemy. Either way, the maze must keep a path open.

## Run

Needs Python 3.11 or newer. Dependencies: numpy and matplotlib.

**With uv** (recommended, uses the exact versions in `uv.lock`):

```
uv run python experiments.py           # full run, ~15 min on one core
uv run python experiments.py --quick   # smoke test, ~75 s
```

The first `uv run` creates `.venv/` and installs everything.

**With plain Python and pip:**

```
python -m venv .venv
.venv\Scripts\activate                 # Windows
source .venv/bin/activate              # macOS / Linux
pip install -r requirements.txt
python experiments.py --quick
```

If you add a dependency, run `uv add <package>`, then
`uv export --no-hashes --no-header --no-annotate --no-emit-project -o requirements.txt`
so both files stay in sync.

Output goes to `out/`: five PNG figures, `evolution.gif`, `results.json`, `run_log.txt`.

## EA design

| Part | Choice |
|---|---|
| Genotype | 21×21 binary grid: 0 = floor, 1 = wall. Start and goal are always floor |
| Phenotype | The maze the robot explores |
| Fitness | **Robot cost = steps walked + distinct tiles inspected.** Only physical work counts. The robot's planning computation is free |
| Constraint | A path must exist. Blocked mazes get a graded penalty: `1 / (1 + walls to remove)`, computed with a 0-1 BFS |
| Mutation | Bit-flip, rate 3/441 per cell |
| Recombination | 2-point crossover on the row-major genome (children inherit horizontal bands) |
| **GA** parent selection | Roulette wheel (FPS), the group's plan, compared against tournament (k = 3) |
| **GA** survivor selection | Steady state: 20 children replace the worst 20 of 50 parents (elitist) |
| **GA** budget | At most 160 walls. Repair removes random walls |
| **NSGA-II** | Objectives: maximize robot cost, minimize wall count. Constraint domination on "walls to remove". μ = λ = 60. Duplicate objective vectors go to the back in survivor selection |
| Generations | 150 |

## Experiments

1. **Random vs. evolved** (`fig1`). Same wall budget.
2. **Parent selection** (`fig2`). FPS vs. tournament over 5 seeds.
3. **NSGA-II front** (`fig3`). Robot cost vs. number of walls, with four mazes from the front.
4. **Wall decay** (`fig4`). Remove k random walls and run the robot again. Compares normal evolved mazes, mazes evolved with decay in the fitness (mean over 3 samples of 3 removed walls, shared seeds per generation, survivors re-evaluated), and random mazes.
5. **Robot pool** (`fig5`). Steps walked on the same mazes by the trained explorer, a wall follower (right-hand rule), and a robot that knows the map.

## Results (seeds 0–4, 150 generations)

| Maze | Robot cost | Steps walked | Tiles inspected | Shortest path |
|---|---|---|---|---|
| Random, ≤160 walls (mean of 30) | 182 | 60 | 122 | 40 |
| GA, roulette wheel (mean best of 5) | 720 | 321 | 399 | 54 |
| GA, tournament (mean best of 5) | 800 | 379 | 421 | 55 |

The evolved robot walks about 7× the shortest path. Each wrong turn sends it into
a dead end that it can only see from inside.

### Finding 1: the robot fitness has no cheap exploit

The NSGA-II front rises smoothly from 10 walls (cost 124) to about 100 walls
(cost ~650), on all 3 seeds (`fig3`). Each extra wall buys more robot work. The
earlier version, which counted A* node expansions, had a cliff: 2–3 tiles around
the goal forced A* to expand the whole grid. Counting physical work removes
that exploit, because the robot never searches cells it cannot reach.

### Finding 2: evolved mazes use deceptive dead ends

The robot plans through unseen space as if it were open. The evolved mazes are
full of short dead-end pockets that the robot must walk into before it can rule
them out (`fig1`, `evolution.gif`). This demo does not measure whether the
pockets point toward the goal. That is a hypothesis to test.

### Finding 3: robustness is a trade-off

- **Wall decay** (`fig4`): removing 2 random walls cuts the evolved mazes from 800 to 687. Removing 32 walls cuts them to 238. They still beat random mazes (164).
- **Decay in the fitness** (`fig4`) gives mazes that start lower (698) but lose less: they are ahead at 16 removed walls (405 vs. 355) and at 32 (262 vs. 238). The two curves cross between 2 and 4 removed walls.

### Finding 4: overfitting to one robot

The slowdown **transfers** to a robot the EA never saw (`fig5`). A wall follower
(right-hand rule) walks 362 steps on the evolved mazes and 80 on random mazes.
The trained explorer walks 379 and 75. A robot that knows the map walks only
55 vs. 41, so the evolved mazes are hard because of hidden information, not because
the shortest path is long.

The decay-robust mazes slow the wall follower more than the trained explorer
(317 vs. 297 steps). This comes from only 3 mazes, so treat it as a hint.

The earlier A* expansion version did not transfer: greedy best-first search was
not slowed at all.

### Finding 5: tournament beats roulette wheel

Tournament selection reaches higher fitness on every seed (`fig2`). When fitness
values are close, FPS gives almost no selection pressure (Lecture 9).

### Limitation found during testing

Changing A*'s tie-break rule (larger g vs. smaller g first) does not change the
robot's route at all. The route depends on the order in which A* checks neighbours.
So "a different tie-break" is not a different robot, and the pool uses a wall
follower instead.

## Extension: coins, upgrades, and coevolution

`coin_game.py` adds 10 coins to each maze. The robot sees coins inside its sensor
range and decides if a coin is worth the detour. It spends coins in a shop:

| Item | Price | Effect |
|---|---|---|
| Sensor | 3 coins | Sensor radius +1 (max 3). Sees walls and coins earlier |
| Hammer | 2 coins | One charge. Smashing an adjacent wall costs 1 extra action |

The robot is now a genome of 3 genes in [0, 1]: **greed** (largest detour it
accepts for a coin, 0–20 steps), **sensors** (sensor upgrades to buy before
hammers, 0–2), and **smash** (planner cost of a known wall while it holds a
hammer, 1–21). Robot cost = steps walked + walls smashed. Inspected tiles no
longer count, so a bigger sensor is not a penalty. With no coins, the robot
walks exactly the same route as the original robot (`test_coin_game.py` checks this).

```
uv run python coin_experiments.py          # full run, ~28 min on one core
uv run python coin_experiments.py --quick  # smoke test, ~2 min
uv run python test_coin_game.py            # invariant checks
```

Output goes to `out/coins/`.

| Experiment | What evolves |
|---|---|
| E1 (`c1`) | Robot genes only, against fixed mazes. Tested on unseen coin layouts |
| E2 (`c2`, `c3`, `c5`) | **Competitive coevolution**: 24 mazes vs 16 robots. Each plays 4 random opponents + 2 from the other side's hall of fame, on 2 coin layouts. Baseline: mazes against one fixed robot |
| E3 (`c4`) | **Bait**: the maze genome also holds the 10 coin positions (coin mutation moves a coin with p = 0.15) |

### Results (seeds 0–2, 80 coevolution generations)

**E1. Upgrades break the anti-explorer mazes.** On the mazes evolved against the
original robot (150 GA generations), robot cost per robot:

| Robot | Cost |
|---|---|
| Original robot, no coins | 379 |
| Greedy (chases every coin, smashes freely) | 296 |
| Evolved robot (greed 11.9, sensors 0, smash 5.3) | **187** |
| Same evolved robot, shop closed | 352 |

The shop-closed row is the ablation. Chasing coins only changes the route a
little (379 → 352). The upgrades give most of the gain. On random mazes,
coins do almost nothing (72 → 68–69), because those mazes are already easy.
The evolved robot buys hammers, not sensors, and smashes only when the detour
is about 5 steps or more. The greedy robot smashes too freely and wastes coins.

**E2. Coevolution gives mazes that resist upgrades, and robots that generalize.**
Mean cost on held-out coin layouts (lower = better robot):

| Robot \ mazes | Coevolved | Evolved vs E1 robot | Anti-explorer | Random |
|---|---|---|---|---|
| Original (never detours) | 255 | 261 | 220 | 73 |
| Greedy | 244 | 224 | 296 | 71 |
| E1 robot | 214 | **244** | 187 | 68 |
| Coevolved robot, seed 0 | 212 | 242 | 191 | 71 |
| Coevolved robot, seed 1 | 216 | 188 | 202 | 70 |
| Coevolved robot, seed 2 | 209 | 197 | 238 | 69 |

- Mazes evolved against one fixed robot overfit to it. They raise the E1 robot to 244, but two of the three coevolved robots cost only 188 and 197 on them. This is the same overfitting as Finding 4, now on the maze side.
- No maze set is the hardest for every robot. The coevolved mazes are the only set on which every evolved or coevolved robot costs more than 200.
- The coevolved robots do not agree on one strategy (final genes: greed 9.6 / 17.1 / 2.6, sensors 0 / 1 / 1). Several strategies are about equally good.
- The CIAO plot (`c3`, current champion vs. champions of earlier generations) shows progress on the maze side: later maze champions beat almost all robot champions. It is not a clean arms race. Some maze champions (gen 42, 57) are weak, and the last robot champion is weak. With only 4 + 2 opponents per evaluation, the champion choice is noisy.

**E3. Bait coins: weak evidence.** Mazes that place their own coins against the
greedy robot put more coins on long detours (30–60 steps) than random placement
does (`c4`). The median detour changes only a little (8 vs. 6 steps). On these
mazes, the coins cost the greedy robot 55 actions net: 301 with the coins, 246
with the coins removed. The bait mazes did not beat mazes with random coins
(301 vs. 312). These two numbers are not measured the same way: 312 is the best
in-generation fitness, which selection on noise inflates. In coevolution with
bait, the robots' mean greed was lower than with random coins from about
generation 30 to 75 (`c4`). The final greed values overlap, so treat this as a hint.

**Hammer price sweep** (side study, 3 anti-explorer mazes from 80 GA
generations, 15 robot generations, one seed). The original robot costs 337 on these mazes:

| Hammer price | 1 | 2 | 3 | 5 | none for sale |
|---|---|---|---|---|---|
| Evolved robot cost | 136 | 238 | 246 | 264 | 269 |

Cheap hammers make the mazes almost useless. With no hammers for sale, sensors alone still cut the cost by 20%.

### Ideas not tried

- Let the maze also evolve the shop prices, or give it a budget of "unbreakable" walls.
- More opponents per evaluation, to reduce noise in the coevolution champions.
- A moving guard that the maze places, as a second adversary inside the maze.

## Extension: breakable walls (one EA, one representation)

When the maze placed coins or hammers itself, it hid them (E3). So in this
version, the maze does not place items. It decides **which walls the robot's
hammer can break**.

| Part | Choice |
|---|---|
| Genotype | 21×21 ternary grid: 0 = floor, 1 = breakable wall, 2 = solid wall |
| Robot | The original explorer, plus a hammer with no charge limit. Smashing a breakable wall costs 4 extra actions. It sees the wall type when it inspects a tile |
| Fitness | Robot cost = steps + smash actions + tiles inspected |
| Constraint | Solid walls must not cut the start from the goal (graded penalty, as before) |
| Budgets | ≤ 160 walls, ≤ `solid` of them solid. Repair removes random walls, then makes random solid walls breakable |
| Mutation | Each cell changes with p = 3/441 to one of the other two symbols |
| GA | Same as before: steady state, tournament (k = 3), 2-point crossover, 150 generations |

```
uv run python breakable.py          # ~3 min, 5 seeds
uv run python test_breakable.py
```

### Results (seeds 0–4, `out/breakable/b1_breakable.png`)

| Solid wall budget | 0 | 10 | 20 | 40 | 160 |
|---|---|---|---|---|---|
| Robot cost, best maze (mean of 5) | 443 | 435 | 488 | 536 | 594 |

- **The hammer breaks the old mazes.** Mazes evolved for the robot without a hammer cost 800. With every wall breakable, they cost 185.
- **Evolving against the hammer recovers most of that.** With no solid walls at all, the GA reaches 443. The best maze with 40 solid walls makes the robot smash 38 walls. The maze cannot stop the hammer, so it makes the robot pay for it many times.
- **Solid walls carry somewhat more load.** At a budget of 40, removing one solid wall drops the cost by 39 on average, and removing one breakable wall drops it by 34. Removing the wall drops the cost by 20 or more for 39% of solid walls and 26% of breakable walls. Most walls of both types barely matter.
- **Limitation: the GA does not find the old optimum.** With a budget of 160, the search space contains the old mazes (all walls solid, cost about 800), but the GA reaches only 594. A possible cause: mutation turns a solid wall into a breakable wall half the time. I did not test this.

## Files

- `maze_ea.py`: robot, planner, operators, GA, NSGA-II
- `experiments.py`: runs all experiments and draws the figures
- `coin_game.py`: coin game, upgradable robot, robot GA, coevolution, bait coins
- `coin_experiments.py`: runs E1–E3 and draws the figures in `out/coins/`
- `test_coin_game.py`: invariant checks for the coin game
- `breakable.py`: breakable-wall variant (ternary grid), its GA and experiment
- `test_breakable.py`: invariant checks for the breakable-wall variant
- `ASSIGNMENT.md`: the project brief, word for word
