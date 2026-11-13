# Maze evolution demo: slow down A*

A prototype for the CISC455 group project. An EA designs mazes. A fixed A* solver
tries to solve each one. The EA wants mazes that make A* do as much work as possible.

**Framing:** Tower defense. You buy walls and mud tiles to slow a pathfinding enemy,
and the game does not allow you to block the path completely.

## Run

```
uv run python experiments.py           # full run, ~3 min on one core
uv run python experiments.py --quick   # smoke test, ~20 s
```

Output goes to `out/`: five PNG figures, `evolution.gif`, `results.json`, `run_log.txt`.

## EA design

| Part | Choice |
|---|---|
| Genotype | 21×21 integer grid: 0 = floor, 1 = wall, 2 = mud. Start (top-left) and goal (bottom-right) are always floor |
| Phenotype | The maze that A* searches |
| Solver | A*, Manhattan heuristic, 4-connected. Floor step = 1, mud step = 5. Ties go to the larger g |
| Fitness | Number of nodes A* expands. Deterministic, unlike wall-clock time |
| Constraint | A path must exist. Blocked mazes get a graded penalty: `1 / (1 + walls to remove)`, computed with a 0-1 BFS |
| Build cost | Wall = 1, mud = 2 |
| Mutation | Random resetting, rate 3/441 per cell |
| Recombination | 2-point crossover on the row-major genome (children inherit horizontal bands) |
| **GA** parent selection | Roulette wheel (FPS), the group's plan, compared against tournament (k = 3) |
| **GA** survivor selection | Steady state: 20 children replace the worst 20 of 50 parents (elitist) |
| **GA** budget | Build cost ≤ 160. Repair turns random obstacles back into floor |
| **NSGA-II** | Objectives: maximize expansions, minimize build cost. No budget. Constraint domination on "walls to remove". μ = λ = 60 |
| Generations | 150 |

## Experiments

1. **Random vs. evolved** (`fig1`). Same budget. Shows what the EA learned.
2. **Parent selection** (`fig2`). FPS vs. tournament over 5 seeds.
3. **NSGA-II front** (`fig3`). The trade-off between build cost and slowdown, with four mazes from the front.
4. **Wall decay** (`fig4`). Remove k random walls and measure A* again. Compares normal evolved mazes, mazes evolved with decay in the fitness (mean over 5 decay samples of 3 walls, shared seeds per generation, survivors re-evaluated), and random mazes.
5. **Solver pool** (`fig5`). The same mazes, solved by A* with the other tie-break and by greedy best-first search. Tests overfitting to one solver.

6. **Alternative fitness: path cost** (`fig6`, `fig7`). The same GA and NSGA-II, with fitness = cost of the path A* returns (the enemy's travel time).

## Results (seeds 0–4, 150 generations, ~3 min on one core)

| Measure | Value |
|---|---|
| Random maze, budget 160 | 130 nodes expanded (mean of 30) |
| GA, roulette wheel (FPS) | 423 nodes expanded (mean best of 5 seeds) |
| GA, tournament | 436 nodes expanded (mean best of 5 seeds). The grid has 441 cells |
| GA on path cost | path cost 154–196, against 43 for random mazes and ~52 for the mazes evolved on expansions |

### Finding 1: the expansion fitness has a cheap exploit (the "moat")

On an open grid, every cell has the same A* value f = g + h = 40. A* normally avoids
expanding all of them through tie-breaking. If the EA puts 2–3 mud or wall tiles
around the goal, the optimal cost C* rises above 40. A* must then expand **every**
cell with f < C*, which is almost the whole grid. NSGA-II shows this as a cliff:
build cost 16 → 154 expansions, build cost 17 → 427 expansions (`fig3`).

So "maximize expansions" saturates near 441 and does not need maze-like
structure. This is a property of A* (it must expand every node with f < C*),
not a bug in the code.

### Finding 2: path cost gives a real trade-off

With fitness = path cost, the mazes have winding corridors (`fig6`), and NSGA-II
finds a smooth front between build cost and travel time (`fig7`). The NSGA-II
fronts are not converged at 150 generations: the highest path cost differs a lot
between seeds (82, 154, 232).

### Finding 3: the expansion mazes are fragile and specific to A*

- **Wall decay** (`fig4`): removing 2 random walls cuts the evolved mazes from 436 to 241 expansions. They rely on a few moat walls. The best one has only 8 walls in total.
- **Decay in the fitness** (`fig4`) fixes this: the mazes keep ~400 expansions with 32 walls removed. They do it by building the moat from mud, because the decay rule removes only walls. The EA exploited the test rule.
- **Solver pool** (`fig5`): greedy best-first search expands only 45 nodes on the evolved mazes, fewer than on random mazes (54). The slowdown does not transfer.

### Finding 4: tournament beats roulette wheel

Tournament selection reaches higher fitness on every seed (`fig2`). Late in the
run, fitness values like 420 vs. 430 give FPS almost no selection pressure.

### Finding 5: NSGA-II needs duplicate handling

Without it, copies of one maze fill the first front and the population collapses
to a single point. The code now pushes duplicate objective vectors to the back
during survivor selection.

## Files

- `maze_ea.py`: tiles, solvers, operators, GA, NSGA-II
- `experiments.py`: runs all experiments and draws the figures
