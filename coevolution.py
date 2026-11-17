"""Co-evolve mazes and the robot that explores them.

Two populations evolve at the same time:

  Mazes  : 21 x 21 binary grids (same genome and operators as maze_ea.run_ga).
           Fitness = mean robot cost against a panel of robots. Higher is better.
  Robots : 8 real genes that parametrize the planner, including the strength of a
           dead-end memory (see maze_ea.ROBOT_NAMES).
           Fitness = mean *normalized* cost (cost / shortest path) against a panel
           of mazes. Lower is better, so the stored fitness is its negative.

Panels. In every generation each side is scored against a small panel of opponents
from the other side: the current top individuals, a few random ones, and some
members of the hall of fame (the best individual of every earlier generation).
The hall of fame stops the populations from cycling: a maze cannot "forget" how to
beat an old robot, and a robot cannot forget an old maze.

One maze x robot cost cache serves both populations, so a run is cheaper than
two separate evaluations.

Raw fitness curves wobble in co-evolution because both sides move. Use
`master_tournament` after the run for a fair measure of progress: it plays every
generation's best maze against every generation's best robot.
"""
from dataclasses import dataclass
from itertools import count

import numpy as np

from maze_ea import (ROBOT_HIGH, ROBOT_LOW, ROBOT_NAMES, STANDARD_ROBOT, astar, crossover, explore,
                     mutate, random_maze, repair_budget, walls_to_open)


# ----------------------------------------------------------------------------- individuals
@dataclass(eq=False)
class Maze:
    uid: int
    grid: np.ndarray
    blocked: int            # walls that must be removed to open a path (0 = solvable)
    shortest: int | None    # path length of a robot that knows the map (None if blocked)
    fit: float = 0.0        # higher is better


@dataclass(eq=False)
class Bot:
    uid: int
    genes: np.ndarray
    fit: float = 0.0        # higher is better (= minus mean normalized cost)


class Arena:
    """Makes individuals and caches robot cost for every (maze, robot) pair played."""

    def __init__(self):
        self._ids = count()
        self.cache = {}
        self.evals = 0

    def maze(self, grid):
        blocked = walls_to_open(grid)
        shortest = None if blocked else len(astar(grid.ravel().tolist(), grid.shape[0], 0, grid.size - 1)) - 1
        return Maze(next(self._ids), grid, blocked, shortest)

    def bot(self, genes):
        return Bot(next(self._ids), np.asarray(genes, float))

    def cost(self, maze, bot):
        key = (maze.uid, bot.uid)
        if key not in self.cache:
            self.cache[key] = explore(maze.grid, bot.genes)["cost"]
            self.evals += 1
        return self.cache[key]


def score_mazes(arena, mazes, panel):
    """Maze fitness = mean robot cost over the robot panel. Blocked mazes get a small graded score."""
    for m in mazes:
        m.fit = (1.0 / (1 + m.blocked) if m.blocked
                 else float(np.mean([arena.cost(m, b) for b in panel])))


def score_bots(arena, bots, panel):
    """Robot fitness = minus mean (cost / shortest path) over the maze panel."""
    for b in bots:
        b.fit = -float(np.mean([arena.cost(m, b) / m.shortest for m in panel]))


# ----------------------------------------------------------------------------- operators
def tournament(rng, pop, k):
    idx = rng.choice(len(pop), min(k, len(pop)), replace=False)
    return pop[max(idx, key=lambda i: pop[i].fit)]


def bot_crossover(rng, a, b, alpha=0.25, low=ROBOT_LOW, high=ROBOT_HIGH):
    """BLX-alpha blend crossover: each gene is drawn from the parents' range, widened by alpha."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    d = hi - lo
    return np.clip(rng.uniform(lo - alpha * d, hi + alpha * d), low, high)


def bot_mutate(rng, genes, p_gene=0.3, sigma=0.1, low=ROBOT_LOW, high=ROBOT_HIGH):
    """Gaussian mutation: each gene changes with probability p_gene; sigma is a fraction of its range."""
    mask = rng.random(genes.size) < p_gene
    noise = rng.normal(0.0, sigma * (ROBOT_HIGH - ROBOT_LOW))
    return np.clip(genes + mask * noise, low, high)


def choose_panel(rng, pop, hof, n_top, n_rand, n_hof):
    """Opponents for the other side: top individuals, random ones, hall-of-fame members.

    Only solvable mazes are eligible (the helper ignores individuals with blocked > 0).
    """
    pool = [p for p in pop if not getattr(p, "blocked", 0)]
    ranked = sorted(pool, key=lambda p: -p.fit)
    panel = ranked[:n_top]
    rest = ranked[n_top:]
    if rest and n_rand:
        panel += [rest[i] for i in rng.choice(len(rest), min(n_rand, len(rest)), replace=False)]
    if hof and n_hof:
        panel += [hof[i] for i in rng.integers(len(hof), size=n_hof)]
    elif rest:  # no hall of fame yet (generation 0): fill with more random individuals
        extra = [p for p in rest if p not in panel]
        panel += [extra[i] for i in rng.choice(len(extra), min(n_hof, len(extra)), replace=False)] if extra else []
    seen, unique = set(), []
    for p in panel:
        if p.uid not in seen:
            seen.add(p.uid)
            unique.append(p)
    return unique


# ----------------------------------------------------------------------------- the co-evolution loop
def run_coevolution(seed, n=21, n_mazes=30, n_bots=30, lam_mazes=15, lam_bots=15, gens=100,
                    budget=160, rate=3 / 441, tournament_k=3, panel=(2, 2, 1),
                    bot_p_gene=0.3, bot_sigma=0.1, seed_standard=False, memory=True):
    """Run the co-evolution.

    memory         : allow the dead-end memory gene (p_dead). False pins it to 0, which gives
                     the robot without memory, for an ablation.
    panel          : (top, random, hall-of-fame) opponents each individual faces per generation.
    lam_*          : children per generation. Survivor selection is (mu + lambda) truncation, and
                     parents and children are all scored against the same panel that generation.
    seed_standard  : put the standard A* robot into the first robot population (off by default,
                     so the robots have to find good settings themselves).

    Returns a dict with the final populations, the hall of fame (best maze and best robot
    of every generation), per-generation history and the number of explore() calls made.
    """
    rng = np.random.default_rng(seed)
    arena = Arena()
    low, high = ROBOT_LOW, ROBOT_HIGH.copy()
    if not memory:
        high[ROBOT_NAMES.index("p_dead")] = 0.0

    mazes = []
    while len(mazes) < n_mazes:  # start from solvable mazes so robot panels always exist
        g = repair_budget(rng, random_maze(rng, n), budget)
        if walls_to_open(g) == 0:
            mazes.append(arena.maze(g))
    bots = [arena.bot(rng.uniform(low, high)) for _ in range(n_bots)]
    if seed_standard:
        bots[0] = arena.bot(STANDARD_ROBOT)

    maze_hof, bot_hof = [], []
    hist = {k: [] for k in ("maze_best", "maze_mean", "bot_best", "bot_mean")}
    gene_hist = []
    n_top, n_rand, n_hof = panel

    for gen in range(gens):
        # 1. opponent panels, chosen from the current populations (ranked by last generation's fitness)
        bot_panel = choose_panel(rng, bots, bot_hof, n_top, n_rand, n_hof)
        maze_panel = choose_panel(rng, mazes, maze_hof, n_top, n_rand, n_hof)

        # 2. children
        maze_kids = [arena.maze(repair_budget(rng, mutate(rng, crossover(
            rng, tournament(rng, mazes, tournament_k).grid, tournament(rng, mazes, tournament_k).grid), rate),
            budget)) for _ in range(lam_mazes)]
        bot_kids = [arena.bot(bot_mutate(rng, bot_crossover(
            rng, tournament(rng, bots, tournament_k).genes, tournament(rng, bots, tournament_k).genes,
            low=low, high=high), bot_p_gene, bot_sigma, low, high)) for _ in range(lam_bots)]

        # 3. everyone plays the same panel this generation, so the comparison is consistent
        mazes, bots = mazes + maze_kids, bots + bot_kids
        score_mazes(arena, mazes, bot_panel)
        score_bots(arena, bots, maze_panel)

        # 4. survivor selection
        mazes = sorted(mazes, key=lambda m: -m.fit)[:n_mazes]
        bots = sorted(bots, key=lambda b: -b.fit)[:n_bots]

        # 5. record, and add this generation's champions to the hall of fame
        maze_hof.append(mazes[0])
        bot_hof.append(bots[0])
        gene_hist.append(bots[0].genes.copy())
        hist["maze_best"].append(mazes[0].fit)
        hist["maze_mean"].append(float(np.mean([m.fit for m in mazes])))
        hist["bot_best"].append(-bots[0].fit)
        hist["bot_mean"].append(-float(np.mean([b.fit for b in bots])))

    return dict(mazes=mazes, bots=bots, maze_hof=maze_hof, bot_hof=bot_hof,
                history={k: np.array(v) for k, v in hist.items()},
                gene_history=np.array(gene_hist), evals=arena.evals)


# ----------------------------------------------------------------------------- evaluation
def master_tournament(run, stride=5):
    """Play each generation's best maze against each generation's best robot.

    Returns (generations, cost, normalized) where cost[i, j] is the robot cost of robot
    generation gens[i] on maze generation gens[j], and normalized divides by the shortest path.
    Reading it: moving right (later mazes) should raise the cost for a fixed robot, and moving
    up (later robots) should lower it for a fixed maze.
    """
    last = len(run["maze_hof"]) - 1
    gens = sorted(set(range(0, last + 1, stride)) | {last})
    cost = np.zeros((len(gens), len(gens)))
    norm = np.zeros_like(cost)
    for j, gm in enumerate(gens):
        maze = run["maze_hof"][gm]
        for i, gb in enumerate(gens):
            cost[i, j] = explore(maze.grid, run["bot_hof"][gb].genes)["cost"]
            norm[i, j] = cost[i, j] / maze.shortest
    return gens, cost, norm
