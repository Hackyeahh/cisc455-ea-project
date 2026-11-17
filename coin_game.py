"""Coin game: an exploring robot that can buy upgrades, and mazes that fight back.

The maze is the same n x n wall grid as in maze_ea.py. K coins lie on floor tiles.
The robot still explores with the free-space assumption, but now it can see coins
inside its sensor range and it decides whether a coin is worth the detour.

Shop (the robot buys automatically when it can pay):
  sensor  : 3 coins, sensor radius +1 (radius 1 -> 2 -> 3). Sees walls and coins earlier.
  hammer  : 2 coins, one charge. Smashing an adjacent wall takes 1 extra step.

Robot genome: 3 genes in [0, 1]
  greed   -> largest detour (extra steps on the known map) it accepts for a visible coin, 0..20
  sensors -> how many sensor upgrades to buy before it buys hammers, 0..2
  smash   -> planner cost of a known wall while it holds a hammer, 1..21 (low = smash often)

Robot cost = actions taken (steps walked + walls smashed). The robot minimizes it.
The maze maximizes it. Inspected tiles do not count, so a bigger sensor is not a penalty.
"""
from collections import deque
from heapq import heappop, heappush

import numpy as np

from maze_ea import WALL, crossover, mutate, neighbours, repair_budget, random_maze, walls_to_open

SENSOR_PRICE, HAMMER_PRICE, MAX_RADIUS = 3, 2, 3
GENE_NAMES = ("greed", "sensors", "smash")

_OFFSETS = {r: [(dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1) if 0 < abs(dr) + abs(dc) <= r]
            for r in range(1, MAX_RADIUS + 1)}


def decode(genes):
    g = np.clip(np.asarray(genes, float), 0, 1)
    return dict(greed=20 * g[0], sensors=min(2, int(3 * g[1])), smash=1 + 20 * g[2])


# Hand-made robots used as baselines
IGNORE_COINS = (0.0, 0.0, 1.0)   # the original explorer: never detours, never smashes
GREEDY = (1.0, 0.0, 0.0)         # chases every coin, buys hammers, smashes whenever cheaper


def bfs(known, n, src):
    """Distances on the known map (unknown = open, known walls blocked)."""
    dist = [-1] * (n * n)
    dist[src] = 0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in neighbours(u, n):
            if dist[v] < 0 and not known[v]:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


def plan(known, n, start, goal, wall_cost):
    """A* with cell costs: open/unknown = 1, known wall = wall_cost (None = blocked)."""
    gr, gc = divmod(goal, n)
    g, parent, closed = {start: 0}, {}, set()
    heap, counter = [(0, 0, 0, start)], 0
    while heap:
        _, _, _, u = heappop(heap)
        if u in closed:
            continue
        if u == goal:
            path = [u]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[-2::-1]       # without start
        closed.add(u)
        for v in neighbours(u, n):
            if v in closed:
                continue
            if known[v]:
                if wall_cost is None:
                    continue
                ng = g[u] + wall_cost
            else:
                ng = g[u] + 1
            if ng < g.get(v, 1 << 30):
                g[v], parent[v] = ng, u
                vr, vc = divmod(v, n)
                counter += 1
                heappush(heap, (ng + abs(gr - vr) + abs(gc - vc), -ng, counter, v))
    return None


def random_coins(rng, grid, k):
    """k distinct floor tiles, never the start or the goal."""
    free = np.flatnonzero(grid.ravel() == 0)
    free = free[(free != 0) & (free != grid.size - 1)]
    return [int(c) for c in rng.choice(free, min(k, free.size), replace=False)]


def play(grid, coins, genes, record=False, shop=True):
    """Run one robot through one maze with one coin layout. Returns a result dict.

    shop=False: the robot still chases coins but cannot buy anything (ablation).
    """
    p = decode(genes)
    n = grid.shape[0]
    N, goal = n * n, n * n - 1
    truth = grid.ravel().tolist()
    known, seen = [0] * N, bytearray(N)
    coin_at, visible = set(coins), set()
    pos, actions, held, hammers, radius = 0, 0, 0, 0, 1
    collected, smashed = 0, 0
    trail, events = [0], []
    route, route_set, target = [], set(), goal
    cap = 6 * N

    def sense():
        nonlocal_flags = [False, False]          # (route blocked, new coin seen)
        r0, c0 = divmod(pos, n)
        for dr, dc in _OFFSETS[radius]:
            r, c = r0 + dr, c0 + dc
            if 0 <= r < n and 0 <= c < n:
                v = r * n + c
                if not seen[v]:
                    seen[v] = 1
                    if truth[v] == WALL:
                        known[v] = 1
                        nonlocal_flags[0] |= v in route_set
                    if v in coin_at:
                        visible.add(v)
                        nonlocal_flags[1] = True
        return nonlocal_flags

    def choose_target():
        if not visible or p["greed"] <= 0:
            return goal
        d_pos, d_goal = bfs(known, n, pos), bfs(known, n, goal)
        base = d_pos[goal] if d_pos[goal] >= 0 else 4 * N
        best, best_detour = goal, p["greed"]
        for c in visible:
            if d_pos[c] < 0 or d_goal[c] < 0:
                continue
            detour = d_pos[c] + d_goal[c] - base
            if detour <= best_detour:
                best, best_detour = c, detour
        return best

    replan = True
    while pos != goal and actions < cap:
        blocked, new_coin = sense()
        if pos in coin_at:
            coin_at.discard(pos)
            visible.discard(pos)
            held += 1
            collected += 1
            replan = True
            while shop:                                   # shop
                if radius - 1 < p["sensors"] and held >= SENSOR_PRICE:
                    held -= SENSOR_PRICE
                    radius += 1
                    events.append((len(trail) - 1, "sensor"))
                    sense()
                elif radius - 1 >= p["sensors"] and held >= HAMMER_PRICE:
                    held -= HAMMER_PRICE
                    hammers += 1
                    events.append((len(trail) - 1, "hammer"))
                else:
                    break
        if replan or blocked or new_coin or not route:
            target = choose_target()
            route = plan(known, n, pos, target, p["smash"] if hammers else None)
            if route is None:                             # only if the goal is cut off on the known map
                return None
            route_set = set(route)
            replan = False
        nxt = route.pop(0)
        route_set.discard(nxt)
        if known[nxt]:                                    # planned through a known wall: smash it
            hammers -= 1
            truth[nxt] = known[nxt] = 0
            actions += 1
            smashed += 1
            events.append((len(trail) - 1, "smash"))
            replan = hammers == 0
        pos = nxt
        actions += 1
        trail.append(pos)
        if pos == target:
            replan = True
    out = dict(cost=actions, steps=len(trail) - 1, collected=collected, smashed=smashed,
               radius=radius, finished=pos == goal)
    if record:
        out.update(trail=trail, events=events, seen=seen)
    return out


# ----------------------------------------------------------------------------- evaluation helpers
def match_cost(grid, coins, genes, shop=True):
    res = play(grid, coins, genes, shop=shop)
    return 6 * grid.size if res is None else res["cost"]


def maze_score(grid, coin_sets, robots):
    """Mean robot cost over robots x coin layouts. Blocked mazes get the graded penalty."""
    blocked = walls_to_open(grid)
    if blocked:
        return 1.0 / (1 + blocked)
    return float(np.mean([match_cost(grid, cs, r) for r in robots for cs in coin_sets]))


# ----------------------------------------------------------------------------- variation for robots
def mutate_robot(rng, genes, sigma=0.12):
    return np.clip(genes + rng.normal(0, sigma, len(genes)), 0, 1)


def blend_robot(rng, a, b):
    """Uniform arithmetic recombination: child = a + u * (b - a), u ~ U(0, 1) per gene."""
    u = rng.random(len(a))
    return a + u * (b - a)


# ----------------------------------------------------------------------------- evolve robots vs fixed mazes
def evolve_robots(seed, mazes, coin_sets, pop=20, gens=40, k=3):
    """(mu + mu) GA on robot genes against a fixed test set. Fitness = mean cost (lower is better)."""
    rng = np.random.default_rng(seed)
    P = rng.random((pop, 3))

    def cost(g):
        return float(np.mean([match_cost(m, cs, g) for m, cset in zip(mazes, coin_sets) for cs in cset]))

    fit = np.array([cost(g) for g in P])
    history = []
    for _ in range(gens):
        def pick():
            idx = rng.choice(pop, k, replace=False)
            return P[idx[np.argmin(fit[idx])]]
        kids = np.array([mutate_robot(rng, blend_robot(rng, pick(), pick())) for _ in range(pop)])
        kid_fit = np.array([cost(g) for g in kids])
        allP, allF = np.vstack([P, kids]), np.concatenate([fit, kid_fit])
        keep = np.argsort(allF)[:pop]
        P, fit = allP[keep], allF[keep]
        history.append((fit.min(), fit.mean()))
    return dict(best=P[0], best_cost=fit[0], history=np.array(history), pop=P, fit=fit)


# ----------------------------------------------------------------------------- competitive coevolution
def coevolve(seed, n=21, budget=160, n_coins=10, maze_pop=24, robot_pop=16, gens=60,
             opponents=4, hof_opponents=2, coin_layouts=2, rate=3 / 441, k=3, bait=False,
             fixed_robot=None):
    """Two populations evolve against each other.

    Mazes maximize robot cost, robots minimize it. Each individual plays `opponents`
    random members of the other population plus `hof_opponents` members of the other
    side's hall of fame (one champion stored per generation). The hall of fame stops
    the populations forgetting old tricks (cycling).

    bait=True: the maze also chooses where its coins lie (coins are part of the maze
    genome). Otherwise coins are random and shared within a generation.
    fixed_robot: if set, the robot side does not evolve (a one-sided maze GA baseline).
    Returns histories, the champion of each generation (for the CIAO plot), and finals.
    """
    rng = np.random.default_rng(seed)
    mazes = [repair_budget(rng, random_maze(rng, n), budget) for _ in range(maze_pop)]
    maze_coins = [random_coins(rng, m, n_coins) for m in mazes] if bait else None
    robots = rng.random((robot_pop, 3)) if fixed_robot is None else np.array([fixed_robot] * robot_pop, float)
    hof_m, hof_r = [], []
    hist = dict(maze_best=[], maze_mean=[], robot_best=[], robot_mean=[], genes=[], collected=[])
    champs = []

    def layouts(gen, grid, coins_of_maze):
        if bait:
            return [coins_of_maze]
        return [random_coins(np.random.default_rng((seed, gen, s, 7)), grid, n_coins) for s in range(coin_layouts)]

    def score_all(gen):
        """Play the scheduled matches once and use each result for both sides."""
        ms, rs = [[] for _ in mazes], [[] for _ in robots]
        for i, m in enumerate(mazes):
            if walls_to_open(m):
                continue
            opp = list(rng.choice(robot_pop, min(opponents, robot_pop), replace=False))
            lay = layouts(gen, m, maze_coins[i] if bait else None)
            for j in opp:
                for cs in lay:
                    c = match_cost(m, cs, robots[j])
                    ms[i].append(c)
                    rs[j].append(c)
            for r in (rng.choice(len(hof_r), min(hof_opponents, len(hof_r)), replace=False) if hof_r else []):
                for cs in lay:
                    ms[i].append(match_cost(m, cs, hof_r[r]))
        for j, r in enumerate(robots):
            if hof_m:
                for h in rng.choice(len(hof_m), min(hof_opponents, len(hof_m)), replace=False):
                    hm, hc = hof_m[h]
                    for cs in ([hc] if bait else layouts(gen, hm, None)):
                        rs[j].append(match_cost(hm, cs, r))
        mf = np.array([np.mean(s) if s else 1.0 / (1 + walls_to_open(m)) for s, m in zip(ms, mazes)])
        rf = np.array([np.mean(s) if s else 6.0 * n * n for s in rs])
        return mf, rf

    def tour(fit, size, better):
        idx = rng.choice(size, k, replace=False)
        return idx[better(fit[idx])]

    mf, rf = score_all(0)
    for gen in range(gens):
        # --- champions into the hall of fame
        bm, br = int(np.argmax(mf)), int(np.argmin(rf))
        hof_m.append((mazes[bm].copy(), list(maze_coins[bm]) if bait else None))
        hof_r.append(robots[br].copy())
        champs.append(dict(maze=mazes[bm].copy(), coins=list(maze_coins[bm]) if bait else None,
                           robot=robots[br].copy()))
        hist["maze_best"].append(mf.max()); hist["maze_mean"].append(mf.mean())
        hist["robot_best"].append(rf.min()); hist["robot_mean"].append(rf.mean())
        hist["genes"].append(robots.mean(0))

        # --- breed mazes (generational, with 2 elites)
        order = np.argsort(-mf)
        new_m = [mazes[i].copy() for i in order[:2]]
        new_c = [list(maze_coins[i]) for i in order[:2]] if bait else None
        while len(new_m) < maze_pop:
            a, b = tour(mf, maze_pop, np.argmax), tour(mf, maze_pop, np.argmax)
            child = repair_budget(rng, mutate(rng, crossover(rng, mazes[a], mazes[b]), rate), budget)
            new_m.append(child)
            if bait:
                pool = [c if rng.random() < 0.5 else d for c, d in zip(maze_coins[a], maze_coins[b])]
                new_c.append(repair_coins(rng, child, pool, n_coins))
        mazes, maze_coins = new_m, new_c

        # --- breed robots (generational, with 2 elites)
        if fixed_robot is None:
            order = np.argsort(rf)
            new_r = [robots[i].copy() for i in order[:2]]
            while len(new_r) < robot_pop:
                a, b = tour(rf, robot_pop, np.argmin), tour(rf, robot_pop, np.argmin)
                new_r.append(mutate_robot(rng, blend_robot(rng, robots[a], robots[b])))
            robots = np.array(new_r)
        mf, rf = score_all(gen + 1)
    hist = {k_: np.array(v) for k_, v in hist.items()}
    return dict(hist=hist, champs=champs, mazes=mazes, maze_coins=maze_coins, robots=robots, mf=mf, rf=rf)


def repair_coins(rng, grid, coins, k, move_rate=0.15):
    """Keep k distinct coins on floor tiles. Moves some coins at random (coin mutation)."""
    flat = grid.ravel()
    out = []
    for c in coins:
        if rng.random() < move_rate:
            c = int(rng.integers(1, grid.size - 1))
        if flat[c] == 0 and c not in out and 0 < c < grid.size - 1:
            out.append(int(c))
    if len(out) < k:
        free = [c for c in np.flatnonzero(flat == 0) if 0 < c < grid.size - 1 and c not in out]
        out += [int(c) for c in rng.choice(free, k - len(out), replace=False)]
    return out


def coin_detours(grid, coins):
    """For each coin: extra steps a robot that knew the map must walk to pick it up.

    Large values mean the coin sits deep in a dead end or far from the shortest route.
    """
    n = grid.shape[0]
    known = grid.ravel().tolist()
    d_s, d_g = bfs(known, n, 0), bfs(known, n, n * n - 1)
    base = d_s[-1]
    return [d_s[c] + d_g[c] - base for c in coins if d_s[c] >= 0]
