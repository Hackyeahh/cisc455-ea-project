"""Evolve mazes that slow down an exploring robot.

The robot does not know the maze. It assumes unseen tiles are open, plans a route
to the goal with A*, walks, inspects the tiles next to it, and replans when it
finds a wall on its route. The EA places walls to make the robot work as hard as
possible, but a path from start to goal must stay open.

Genotype : n x n binary grid, 0 = floor, 1 = wall
Phenotype: the maze the robot explores
Fitness  : robot cost = steps walked + distinct tiles inspected (physical work only;
           the robot's planning computation is free)
"""
from collections import deque
from heapq import heappop, heappush

import numpy as np

FLOOR, WALL = 0, 1

# ----------------------------------------------------------------------------- robot genome
# A robot is a real-valued vector of 8 genes that parametrizes its planner:
#   w_h        heuristic weight. 1 = standard A*, 0 = Dijkstra-like, > 1 = greedy
#   p_unknown  extra step cost for entering a tile the robot has not inspected yet
#   p_visit    extra step cost per earlier visit of a tile (discourages backtracking)
#   p_dead     dead-end memory. Extra step cost near tiles of pockets the robot had to back
#              out of earlier in this run (0 = no memory, see DEAD_RADIUS)
#   key_N/S/W/E  sort keys; argsort gives the order A* expands neighbours in
ROBOT_NAMES = ("w_h", "p_unknown", "p_visit", "p_dead", "key_N", "key_S", "key_W", "key_E")
ROBOT_LOW = np.zeros(8)
ROBOT_HIGH = np.array([3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0])
# Plain A*, original neighbour order (N, S, W, E), no memory. Behaves exactly like the old explorer.
STANDARD_ROBOT = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1 / 3, 2 / 3, 1.0])
DEFAULT_ORDER = (0, 1, 2, 3)
DEAD_RADIUS = 3    # a remembered dead tile raises the cost of tiles within this Manhattan distance


def neighbours(u, n, order=DEFAULT_ORDER):
    """Neighbours of tile u. order is a permutation of (0=N, 1=S, 2=W, 3=E)."""
    r, c = divmod(u, n)
    cand = (u - n if r > 0 else -1, u + n if r < n - 1 else -1,
            u - 1 if c > 0 else -1, u + 1 if c < n - 1 else -1)
    for i in order:
        if cand[i] >= 0:
            yield cand[i]


def decode_robot(robot):
    """Gene vector -> (w_h, p_unknown, p_visit, p_dead, neighbour order)."""
    robot = STANDARD_ROBOT if robot is None else robot
    return (float(robot[0]), float(robot[1]), float(robot[2]), float(robot[3]),
            tuple(int(i) for i in np.argsort(robot[4:], kind="stable")))


# ----------------------------------------------------------------------------- planner
def astar(walls, n, start, goal, deep_ties=True, w_h=1.0, extra=None, order=DEFAULT_ORDER):
    """A* on a 4-connected grid, Manhattan heuristic.

    walls: flat list, 1 = blocked. Returns the path [start, ..., goal] or None.
    deep_ties: on equal f, prefer the node with larger g (the usual choice).
    w_h:   priority = g + w_h * h. 1 is standard A*; other values give a different robot.
    extra: optional flat list of extra step cost for entering each tile (None = unit cost).
    order: neighbour expansion order (permutation of 0..3).
    Step costs stay >= 1, so a path is always found when one exists.
    """
    gr, gc = divmod(goal, n)
    g = {start: 0}
    parent = {}
    closed = set()
    heap = [(0, 0, 0, start)]
    counter = 0
    while heap:
        _, _, _, u = heappop(heap)
        if u in closed:
            continue
        if u == goal:
            path = [u]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1]
        closed.add(u)
        for v in neighbours(u, n, order):
            if walls[v] or v in closed:
                continue
            ng = g[u] + 1 + (extra[v] if extra else 0)
            if ng < g.get(v, 1 << 30):
                g[v], parent[v] = ng, u
                vr, vc = divmod(v, n)
                counter += 1
                heappush(heap, (ng + w_h * (abs(gr - vr) + abs(gc - vc)), -ng if deep_ties else ng, counter, v))
    return None


# ----------------------------------------------------------------------------- robot
def explore(grid, robot=None, deep_ties=True):
    """Run the exploring robot from top-left to bottom-right.

    robot: gene vector (see ROBOT_NAMES). None = STANDARD_ROBOT, the original explorer.

    The robot inspects its 4 neighbours at every cell it stands on. It replans only
    when a newly seen wall lies on its remaining route, and it commits to its route
    until then. For the standard robot, new information can only add walls, so an
    unblocked route stays a shortest route on the known map. For robots with
    p_unknown, p_visit or p_dead > 0 the penalties are applied when the plan is made, so a
    committed route can be stale; that is part of the robot's behaviour.
    Every replan is caused by a newly found wall, so the run always ends at the goal.

    Dead-end memory (p_dead > 0). After a replan, if the new route starts by walking back
    along the trail, the tiles it retraces are a dead-end pocket. The robot remembers them in
    a heat map: every pocket tile adds heat to the tiles within DEAD_RADIUS, fading with
    distance. Planning then pays p_dead * heat to enter a tile, so after being fooled by one
    pocket the robot prefers routes that avoid its neighbourhood (where evolved mazes tend to
    put more pockets). The route is planned again once with the new memory.

    Returns dict(trail=[cells walked], inspected=[0/1 per cell], steps, inspections, cost,
    dead=number of remembered dead tiles, dead_tiles=[their cell indices]).
    """
    w_h, p_unknown, p_visit, p_dead, order = decode_robot(robot)
    n = grid.shape[0]
    truth = grid.ravel().tolist()
    known = [0] * (n * n)              # 1 = known wall; unknown tiles are assumed open
    seen = bytearray(n * n)
    seen[0] = 1
    visits = [0] * (n * n)
    visits[0] = 1
    heat = [0.0] * (n * n)             # dead-end memory: how close each tile is to remembered pockets
    dead = bytearray(n * n)
    goal, pos = n * n - 1, 0
    trail, route, route_set = [0], [], set()

    def plan(start):
        extra = None
        if p_unknown or p_visit or p_dead:
            extra = [p_unknown * (1 - s) + p_visit * k + p_dead * h for s, k, h in zip(seen, visits, heat)]
        return astar(known, n, start, goal, deep_ties, w_h, extra, order)

    def remember_pocket(path):
        """If path (starting at the robot) walks back along the trail, remember the retraced tiles."""
        k = 0
        while k + 2 <= len(trail) and k + 1 < len(path) and path[k + 1] == trail[-2 - k]:
            k += 1
        added = False
        for t in trail[len(trail) - k:] if k else ():       # the k tiles the robot is about to leave
            if dead[t]:
                continue
            dead[t] = added = 1
            tr, tc = divmod(t, n)
            for dr in range(-DEAD_RADIUS, DEAD_RADIUS + 1):
                span = DEAD_RADIUS - abs(dr)
                if 0 <= tr + dr < n:
                    for dc in range(-span, span + 1):
                        if 0 <= tc + dc < n:
                            heat[(tr + dr) * n + tc + dc] += (DEAD_RADIUS + 1 - abs(dr) - abs(dc)) / (DEAD_RADIUS + 1)
        return added
    while pos != goal:
        blocked_route = False
        for v in neighbours(pos, n):
            if not seen[v]:
                seen[v] = 1
                if truth[v] == WALL:
                    known[v] = 1
                    blocked_route |= v in route_set
        if not route or blocked_route:
            route = plan(pos)
            if route is None:          # cannot happen when the maze is solvable
                return None
            if p_dead and remember_pocket(route):
                route = plan(pos)      # plan again with what the pocket just taught the robot
            route = route[1:]
            route_set = set(route)
        pos = route.pop(0)
        route_set.discard(pos)
        visits[pos] += 1
        trail.append(pos)
    inspections = int(sum(seen))
    return dict(trail=trail, inspected=seen, steps=len(trail) - 1, inspections=inspections,
                cost=len(trail) - 1 + inspections, dead=int(sum(dead)),
                dead_tiles=[t for t in range(n * n) if dead[t]])


def wall_follower(grid):
    """Classic maze robot: keep the right hand on a wall. Returns the number of steps.

    Start and goal both touch the outer edge, so following the right-hand wall from
    the start always reaches the goal when a path exists. A step cap guards against loops.
    """
    n = grid.shape[0]
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]   # east, south, west, north (clockwise)
    r = c = 0
    heading, steps = 1, 0                        # start facing south
    while (r, c) != (n - 1, n - 1):
        for turn in (1, 0, -1, 2):               # right, straight, left, back
            h = (heading + turn) % 4
            nr, nc = r + moves[h][0], c + moves[h][1]
            if 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == FLOOR:
                r, c, heading = nr, nc, h
                steps += 1
                break
        if steps > 20 * n * n:
            return None
    return steps


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
        for v in neighbours(u, n):
            w = tiles[v]
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                (dq.appendleft if w == 0 else dq.append)(v)
    return dist[-1]


def evaluate(grid, robot=None, deep_ties=True):
    """Robot cost plus the shortest path a robot that knew the map would walk.

    robot: gene vector, or None for the standard explorer.
    """
    n = grid.shape[0]
    res = dict(blocked=walls_to_open(grid), walls=int(grid.sum()))
    if res["blocked"]:
        return {**res, "cost": 0, "steps": 0, "inspections": 0, "shortest": None, "trail": None}
    run = explore(grid, robot, deep_ties)
    res.update(run)
    res["shortest"] = len(astar(grid.ravel().tolist(), n, 0, n * n - 1)) - 1
    return res


# ----------------------------------------------------------------------------- variation
def fix_ends(grid):
    grid[0, 0] = grid[-1, -1] = FLOOR
    return grid


def random_maze(rng, n, p_wall=0.25):
    return fix_ends((rng.random((n, n)) < p_wall).astype(np.int8))


def crossover(rng, a, b, n_points=2):
    """n-point crossover on the row-major genome: children inherit horizontal bands."""
    cuts = np.sort(rng.choice(np.arange(1, a.size), n_points, replace=False))
    take_b = np.zeros(a.size, bool)
    for cut in cuts:
        take_b[cut:] ^= True
    return np.where(take_b, b.ravel(), a.ravel()).reshape(a.shape).astype(np.int8)


def mutate(rng, grid, rate):
    """Bit-flip mutation: each cell flips wall <-> floor with probability `rate`."""
    grid = grid ^ (rng.random(grid.shape) < rate).astype(np.int8)
    return fix_ends(grid)


def repair_budget(rng, grid, budget):
    """Fixed-budget repair: remove random walls until the wall count <= budget."""
    walls = np.flatnonzero(grid.ravel() == WALL)
    if walls.size > budget:
        grid.ravel()[rng.choice(walls, walls.size - budget, replace=False)] = FLOOR
    return grid


def decay(rng, grid, k):
    """Remove k random walls (wall -> floor). Removing walls can never block a path."""
    walls = np.flatnonzero(grid.ravel() == WALL)
    out = grid.copy()
    out.ravel()[rng.choice(walls, min(k, walls.size), replace=False)] = FLOOR
    return out


# ----------------------------------------------------------------------------- single-objective GA
def ga_fitness(res):
    """Robot cost if solvable. Blocked mazes get a small graded score (fewer walls to open = better)."""
    return float(res["cost"]) if res["blocked"] == 0 else 1.0 / (1 + res["blocked"])


def run_ga(seed, n=21, mu=50, lam=20, gens=150, budget=160, rate=3 / 441,
           parent_sel="fps", tournament_k=3, decay_k=0, decay_samples=0, snapshot_every=0):
    """Steady-state GA: lam children replace the worst lam parents each generation.

    parent_sel  : "fps" (roulette wheel) or "tournament"
    decay_k > 0 : noisy, robust fitness = mean robot cost over `decay_samples` decayed copies.
                  All mazes in a generation share the same decay seeds (common random numbers),
                  and the whole population is re-evaluated every generation.
    """
    rng = np.random.default_rng(seed)
    pop = [repair_budget(rng, random_maze(rng, n), budget) for _ in range(mu)]

    def score(grid, gen):
        base = evaluate(grid)
        if not decay_k or base["blocked"]:
            return ga_fitness(base)
        return float(np.mean([evaluate(decay(np.random.default_rng((seed, gen, s)), grid, decay_k))["cost"]
                              for s in range(decay_samples)]))

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
        history.append((fit.max(), fit.mean()))
        if snapshot_every and (gen % snapshot_every == 0 or gen == gens - 1):
            snapshots.append((gen, pop[int(np.argmax(fit))].copy()))
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


def nsga2_objectives(res):
    """Minimize (-robot cost, wall count); constraint violation = walls to open."""
    return (-res["cost"], res["walls"]), res["blocked"]


def run_nsga2(seed, n=21, mu=60, gens=150, rate=3 / 441, record_gens=()):
    rng = np.random.default_rng(seed)
    pop = [random_maze(rng, n, p_wall=rng.uniform(0.05, 0.45)) for _ in range(mu)]
    evals = [nsga2_objectives(evaluate(g)) for g in pop]
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
        merged_evals = evals + [nsga2_objectives(evaluate(k)) for k in kids]
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
