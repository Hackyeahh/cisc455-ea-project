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

```
uv run python experiments.py           # full run, ~15 min on one core
uv run python experiments.py --quick   # smoke test, ~75 s
```

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

## Files

- `maze_ea.py`: robot, planner, operators, GA, NSGA-II
- `experiments.py`: runs all experiments and draws the figures
