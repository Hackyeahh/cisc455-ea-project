"""Evolve mazes that slow down a fixed pathfinding solver.

Tower-defense framing: the EA "buys" walls and mud tiles to make a pathfinder
work as hard as possible, but a path from start to goal must stay open.

Genotype : n x n int8 grid, 0 = floor, 1 = wall, 2 = mud (integer representation)
Phenotype: the maze the solver searches
Fitness  : number of nodes the solver expands (deterministic measure of search effort)
"""
from collections import deque
from heapq import heappop, heappush

import numpy as np

FLOOR, WALL, MUD = 0, 1, 2
STEP_COST = (1, 0, 5)      # cost to ENTER a tile; mud is slow (walls are never entered)
BUILD_COST = (0, 1, 2)     # price to place a tile; mud costs more than a wall

# The solver pool. The EA trains against SOLVERS["A*"] only; the rest test overfitting.
SOLVERS = {
    "A*": dict(wg=1, wh=1, deep_ties=True),               # A*, ties -> larger g
    "A* (other tie-break)": dict(wg=1, wh=1, deep_ties=False),
    "Greedy best-first": dict(wg=0, wh=1, deep_ties=True),
}


# ----------------------------------------------------------------------------- solvers
def search(grid, wg=1, wh=1, deep_ties=True):
    """Best-first search on a 4-connected grid from top-left to bottom-right.

    priority = wg * g + wh * h, with h = Manhattan distance (admissible: every step costs >= 1).
    wg=1, wh=1 is A*; wg=0, wh=1 is greedy best-first.
    Returns dict(expanded=[cell order], path=[cells] or None, cost=float or None).
    """
    n = grid.shape[0]
    tiles = grid.ravel().tolist()
    goal = n * n - 1
    g = [float("inf")] * (n * n)
    parent = [-1] * (n * n)
    closed = bytearray(n * n)
    g[0] = 0
    heap = [(wh * 2 * (n - 1), 0, 0, 0)]
    order, counter = [], 0
    while heap:
        _, _, _, u = heappop(heap)
        if closed[u]:
            continue
        closed[u] = 1
        order.append(u)
        if u == goal:
            break
        r, c = divmod(u, n)
        for v, ok in ((u - n, r > 0), (u + n, r < n - 1), (u - 1, c > 0), (u + 1, c < n - 1)):
            if not ok or closed[v] or tiles[v] == WALL:
                continue
            ng = g[u] + STEP_COST[tiles[v]]
            if ng < g[v]:
                g[v], parent[v] = ng, u
                vr, vc = divmod(v, n)
                h = (n - 1 - vr) + (n - 1 - vc)
                counter += 1
                heappush(heap, (wg * ng + wh * h, -ng if deep_ties else ng, counter, v))
    else:
        return dict(expanded=order, path=None, cost=None)
    path = [goal]
    while path[-1] != 0:
        path.append(parent[path[-1]])
    return dict(expanded=order, path=path[::-1], cost=g[goal])


def walls_to_open(grid):
    """Fewest walls that must be removed to connect start and goal (0-1 BFS).

    Gives blocked mazes a graded penalty instead of a flat 'death penalty'.
    """
    n = grid.shape[0]
    tiles = grid.ravel().tolist()
    dist = [n * n] * (n * n)
    dist[0] = 0
    dq = deque([0])
    while dq:
        u = dq.popleft()
        r, c = divmod(u, n)
        for v, ok in ((u - n, r > 0), (u + n, r < n - 1), (u - 1, c > 0), (u + 1, c < n - 1)):
            if not ok:
                continue
            w = 1 if tiles[v] == WALL else 0
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                (dq.appendleft if w == 0 else dq.append)(v)
    return dist[-1]


def evaluate(grid, solver="A*"):
    res = search(grid, **SOLVERS[solver])
    res["expansions"] = len(res["expanded"])
    res["blocked"] = 0 if res["path"] else walls_to_open(grid)
    res["build_cost"] = build_cost(grid)
    return res


def build_cost(grid):
    return int((grid == WALL).sum() * BUILD_COST[WALL] + (grid == MUD).sum() * BUILD_COST[MUD])


# ----------------------------------------------------------------------------- variation
def fix_ends(grid):
    grid[0, 0] = grid[-1, -1] = FLOOR
    return grid


def random_maze(rng, n, p_wall=0.25, p_mud=0.05):
    grid = rng.choice(3, size=(n, n), p=[1 - p_wall - p_mud, p_wall, p_mud]).astype(np.int8)
    return fix_ends(grid)


def crossover(rng, a, b, n_points=2):
    """n-point crossover on the row-major genome: children inherit horizontal bands."""
    cuts = np.sort(rng.choice(np.arange(1, a.size), n_points, replace=False))
    take_b = np.zeros(a.size, bool)
    for cut in cuts:
        take_b[cut:] ^= True
    return np.where(take_b, b.ravel(), a.ravel()).reshape(a.shape).astype(np.int8)


def mutate(rng, grid, rate):
    """Random resetting: each cell is redrawn from {floor, wall, mud} with probability `rate`."""
    grid = grid.copy()
    hit = rng.random(grid.shape) < rate
    grid[hit] = rng.integers(0, 3, hit.sum())
    return fix_ends(grid)


def repair_budget(rng, grid, budget):
    """Fixed-budget repair: turn random obstacles back into floor until build cost <= budget."""
    over = build_cost(grid) - budget
    if over > 0:
        cells = np.flatnonzero(grid.ravel() != FLOOR)
        rng.shuffle(cells)
        flat = grid.ravel()
        for i in cells:
            if over <= 0:
                break
            over -= BUILD_COST[flat[i]]
            flat[i] = FLOOR
    return grid


def decay(rng, grid, k):
    """Remove k random walls (wall -> floor). Removing walls can never block a path."""
    walls = np.flatnonzero(grid.ravel() == WALL)
    out = grid.copy()
    out.ravel()[rng.choice(walls, min(k, walls.size), replace=False)] = FLOOR
    return out


# ----------------------------------------------------------------------------- single-objective GA
def objective_value(res, objective):
    """'expansions' = solver search effort; 'path_cost' = travel time of the path the solver returns."""
    return float(res["expansions"] if objective == "expansions" else res["cost"])


def ga_fitness(res, objective="expansions"):
    """Objective if solvable. Blocked mazes get a small graded score (fewer walls to open = better)."""
    return objective_value(res, objective) if res["blocked"] == 0 else 1.0 / (1 + res["blocked"])


def run_ga(seed, n=21, mu=50, lam=20, gens=150, budget=160, rate=3 / 441,
           parent_sel="fps", tournament_k=3, decay_k=0, decay_samples=0, snapshot_every=0,
           objective="expansions"):
    """Steady-state GA: lam children replace the worst lam parents each generation.

    parent_sel  : "fps" (roulette wheel) or "tournament"
    decay_k > 0 : noisy, robust fitness = mean expansions over `decay_samples` decayed copies.
                  All mazes in a generation share the same decay seeds (common random numbers),
                  and the whole population is re-evaluated every generation.
    """
    rng = np.random.default_rng(seed)
    pop = [repair_budget(rng, random_maze(rng, n), budget) for _ in range(mu)]

    def score(grid, gen):
        if not decay_k:
            return ga_fitness(evaluate(grid), objective)
        base = evaluate(grid)
        if base["blocked"]:
            return ga_fitness(base, objective)
        vals = [objective_value(evaluate(decay(np.random.default_rng((seed, gen, s)), grid, decay_k)), objective)
                for s in range(decay_samples)]
        return float(np.mean(vals))

    fit = np.array([score(g, 0) for g in pop])
    history, snapshots = [], []
    for gen in range(gens):
        if parent_sel == "fps":
            probs = fit / fit.sum()
            pick = lambda: rng.choice(mu, p=probs)
        else:
            pick = lambda: max(rng.choice(mu, tournament_k, replace=False), key=lambda i: fit[i])
        kids = [repair_budget(rng, mutate(rng, crossover(rng, pop[pick()], pop[pick()]), rate), budget)
                for _ in range(lam)]
        if decay_k:  # noisy fitness: re-evaluate survivors on this generation's decay seeds
            fit = np.array([score(g, gen + 1) for g in pop])
        kid_fit = np.array([score(k, gen + 1) for k in kids])
        worst = np.argsort(fit)[:lam]
        for slot, kid, f in zip(worst, kids, kid_fit):
            pop[slot], fit[slot] = kid, f
        best = int(np.argmax(fit))
        history.append((fit.max(), fit.mean()))
        if snapshot_every and (gen % snapshot_every == 0 or gen == gens - 1):
            snapshots.append((gen, pop[best].copy()))
    best = int(np.argmax(fit))
    return dict(best=pop[best], history=np.array(history), snapshots=snapshots, pop=pop, fit=fit)


# ----------------------------------------------------------------------------- NSGA-II
def dominance_matrix(F, V):
    """D[i, j] = True if i constrained-dominates j (all objectives minimized).

    Feasible beats infeasible; two infeasible: smaller violation wins; two feasible: Pareto.
    """
    pareto = (F[:, None] <= F[None]).all(-1) & (F[:, None] < F[None]).any(-1)
    feas = V == 0
    return np.where(feas[:, None] & feas[None], pareto, V[:, None] < V[None])


def nondominated_sort(D):
    count = D.sum(0).astype(int)
    fronts, done = [], np.zeros(len(D), bool)
    while not done.all():
        front = np.flatnonzero((count == 0) & ~done)
        fronts.append(front)
        done[front] = True
        count -= D[front].sum(0)
    return fronts


def crowding(F):
    dist = np.zeros(len(F))
    for m in range(F.shape[1]):
        order = np.argsort(F[:, m])
        span = F[order[-1], m] - F[order[0], m]
        dist[order[[0, -1]]] = np.inf
        if span > 0:
            dist[order[1:-1]] += (F[order[2:], m] - F[order[:-2], m]) / span
    return dist


def nsga2_objectives(res, objective="expansions"):
    """Minimize (-objective, build cost); constraint violation = walls to open."""
    value = objective_value(res, objective) if res["blocked"] == 0 else 0.0
    return (-value, res["build_cost"]), res["blocked"]


def run_nsga2(seed, n=21, mu=60, gens=150, rate=3 / 441, record_gens=(), objective="expansions"):
    rng = np.random.default_rng(seed)
    pop = [random_maze(rng, n, p_wall=rng.uniform(0.05, 0.45), p_mud=rng.uniform(0, 0.1))
           for _ in range(mu)]
    evals = [nsga2_objectives(evaluate(g), objective) for g in pop]
    recorded = {}

    def rank_and_crowd(evals, dedupe=False):
        F = np.array([e[0] for e in evals], float)
        V = np.array([e[1] for e in evals], float)
        if dedupe:  # copies of an objective vector go to the back, or clones fill the front
            _, first = np.unique(np.column_stack([F, V]), axis=0, return_index=True)
            dup = np.ones(len(F), bool)
            dup[first] = False
            V = np.where(dup, V + 1e6, V)
        fronts = nondominated_sort(dominance_matrix(F, V))
        rank, crowd = np.zeros(len(F), int), np.zeros(len(F))
        for r, front in enumerate(fronts):
            rank[front] = r
            crowd[front] = crowding(F[front])
        return fronts, rank, crowd

    fronts, rank, crowd = rank_and_crowd(evals)
    for gen in range(gens + 1):
        if gen in record_gens:
            recorded[gen] = [(pop[i].copy(), evals[i]) for i in fronts[0]]
        if gen == gens:
            break

        def pick():  # binary tournament on (rank, crowding distance)
            i, j = rng.choice(mu, 2, replace=False)
            return i if (rank[i], -crowd[i]) < (rank[j], -crowd[j]) else j

        kids = [mutate(rng, crossover(rng, pop[pick()], pop[pick()]), rate) for _ in range(mu)]
        merged = pop + kids
        merged_evals = evals + [nsga2_objectives(evaluate(k), objective) for k in kids]
        m_fronts, _, m_crowd = rank_and_crowd(merged_evals, dedupe=True)
        keep = []
        for front in m_fronts:  # (mu + lambda) survivor selection by front, then crowding
            if len(keep) + len(front) <= mu:
                keep.extend(front)
            else:
                keep.extend(front[np.argsort(-m_crowd[front])][: mu - len(keep)])
                break
        pop = [merged[i] for i in keep]
        evals = [merged_evals[i] for i in keep]
        fronts, rank, crowd = rank_and_crowd(evals)
    front = [(pop[i], evals[i]) for i in fronts[0]]
    return dict(front=front, recorded=recorded, pop=pop, evals=evals)
