"""Breakable walls: the maze decides which walls the robot's hammer can break.

    uv run python breakable.py            # full run
    uv run python breakable.py --quick    # smoke test

Genotype : n x n ternary grid, 0 = floor, 1 = breakable wall, 2 = solid wall
Robot    : the same free-space explorer as maze_ea.py, but it carries a hammer.
           Smashing a breakable wall costs SMASH extra actions. Solid walls stop it.
           It sees the wall type when it inspects a tile.
Fitness  : robot cost = actions (steps + smash actions) + distinct tiles inspected
Budgets  : at most 160 walls in total, and at most `solid` of them solid.
Constraint: solid walls must not cut the start from the goal (graded penalty, as before).
"""
import argparse
import json
import time
from heapq import heappop, heappush
from pathlib import Path

import numpy as np

from maze_ea import crossover, explore as explore_no_hammer, neighbours, walls_to_open
from maze_ea import run_ga as run_binary_ga

FLOOR, BREAK, SOLID = 0, 1, 2
SMASH = 4
OUT = Path(__file__).parent / "out" / "breakable"


# ----------------------------------------------------------------------------- robot
def plan(known, n, start, goal, smash):
    """A* on the known map: unknown/floor = 1, known breakable = 1 + smash, known solid = blocked."""
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
            return path[-2::-1]                    # without start
        closed.add(u)
        for v in neighbours(u, n):
            if v in closed or known[v] == SOLID:
                continue
            ng = g[u] + (1 + smash if known[v] == BREAK else 1)
            if ng < g.get(v, 1 << 30):
                g[v], parent[v] = ng, u
                vr, vc = divmod(v, n)
                counter += 1
                heappush(heap, (ng + abs(gr - vr) + abs(gc - vc), -ng, counter, v))
    return None


def explore(grid, smash=SMASH):
    """Free-space explorer with a hammer. Replans when a newly seen wall lies on its route."""
    n = grid.shape[0]
    N, goal = n * n, n * n - 1
    truth = grid.ravel().tolist()
    known, seen = [FLOOR] * N, bytearray(N)
    seen[0] = 1
    pos, actions, trail, smashed = 0, 0, [0], []
    route, route_set = [], set()
    while pos != goal:
        blocked = False
        for v in neighbours(pos, n):
            if not seen[v]:
                seen[v] = 1
                if truth[v] != FLOOR:
                    known[v] = truth[v]
                    blocked |= v in route_set
        if not route or blocked:
            route = plan(known, n, pos, goal, smash)
            if route is None:                      # solid walls cut the goal off
                return None
            route_set = set(route)
        nxt = route.pop(0)
        route_set.discard(nxt)
        assert truth[nxt] != SOLID
        if truth[nxt] == BREAK:
            truth[nxt] = known[nxt] = FLOOR
            actions += smash
            smashed.append(nxt)
        pos = nxt
        actions += 1
        trail.append(pos)
    inspections = int(sum(seen))
    return dict(trail=trail, smashed=smashed, seen=seen, steps=len(trail) - 1, actions=actions,
                inspections=inspections, cost=actions + inspections)


def evaluate(grid):
    blocked = walls_to_open((grid == SOLID).astype(np.int8))
    if blocked:
        return dict(blocked=blocked, cost=0)
    return dict(blocked=0, **explore(grid))


def fitness(res):
    return float(res["cost"]) if res["blocked"] == 0 else 1.0 / (1 + res["blocked"])


# ----------------------------------------------------------------------------- variation
def random_maze(rng, n, p_wall=0.25, p_solid=0.25):
    walls = rng.random((n, n)) < p_wall
    grid = np.where(walls, np.where(rng.random((n, n)) < p_solid, SOLID, BREAK), FLOOR).astype(np.int8)
    grid[0, 0] = grid[-1, -1] = FLOOR
    return grid


def mutate(rng, grid, rate):
    """Each cell changes with probability `rate` to one of the other two symbols, uniformly."""
    hit = rng.random(grid.shape) < rate
    shift = rng.integers(1, 3, grid.shape)
    out = np.where(hit, (grid + shift) % 3, grid).astype(np.int8)
    out[0, 0] = out[-1, -1] = FLOOR
    return out


def repair(rng, grid, budget, solid):
    """Total walls <= budget (remove random walls), then solid walls <= solid (make random ones breakable)."""
    flat = grid.ravel()
    walls = np.flatnonzero(flat != FLOOR)
    if walls.size > budget:
        flat[rng.choice(walls, walls.size - budget, replace=False)] = FLOOR
    solids = np.flatnonzero(flat == SOLID)
    if solids.size > solid:
        flat[rng.choice(solids, solids.size - solid, replace=False)] = BREAK
    return grid


# ----------------------------------------------------------------------------- GA (same loop as maze_ea.run_ga)
def run_ga(seed, solid, n=21, mu=50, lam=20, gens=150, budget=160, rate=3 / 441, k=3):
    """Steady-state GA, tournament parent selection, lam children replace the worst lam parents."""
    rng = np.random.default_rng(seed)
    pop = [repair(rng, random_maze(rng, n), budget, solid) for _ in range(mu)]
    fit = np.array([fitness(evaluate(g)) for g in pop])
    history = []
    for _ in range(gens):
        def pick():
            idx = rng.choice(mu, k, replace=False)
            return pop[idx[np.argmax(fit[idx])]]
        kids = [repair(rng, mutate(rng, crossover(rng, pick(), pick()), rate), budget, solid) for _ in range(lam)]
        kid_fit = np.array([fitness(evaluate(c)) for c in kids])
        for slot, kid, f in zip(np.argsort(fit)[:lam], kids, kid_fit):
            pop[slot], fit[slot] = kid, f
        history.append(fit.max())
    best = int(np.argmax(fit))
    return dict(best=pop[best], fit=fit[best], history=np.array(history))


# ----------------------------------------------------------------------------- analysis
def load_bearing(grid):
    """Cost drop when one wall is removed (wall -> floor), for solid and for breakable walls."""
    base = evaluate(grid)["cost"]
    drops = {}
    for kind in (SOLID, BREAK):
        d = []
        for c in np.flatnonzero(grid.ravel() == kind):
            g = grid.copy()
            g.ravel()[c] = FLOOR
            d.append(base - evaluate(g)["cost"])
        drops[kind] = d
    return drops


def main(quick):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import to_rgb
    from experiments import BLUE, INK, INK2, INSPECTED_C, ORANGE, SURFACE, WALL_C, plt

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seeds = [0, 1] if quick else [0, 1, 2, 3, 4]
    gens = 30 if quick else 150
    budgets = [0, 10, 20, 40, 160]
    results, log = {}, []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    # 1. How much does the maze need solid walls?
    runs = {s: [run_ga(seed, s, gens=gens) for seed in seeds] for s in budgets}
    for s in budgets:
        say(f"[{time.time() - t0:.0f}s] solid budget {s:3d}: best cost " +
            ", ".join(f"{r['fit']:.0f}" for r in runs[s]) + f" (mean {np.mean([r['fit'] for r in runs[s]]):.0f})")
    results["cost_by_solid_budget"] = {s: [float(r["fit"]) for r in runs[s]] for s in budgets}

    # 2. Mazes evolved against the robot with no hammer: give the robot a hammer (all walls breakable)
    old = [run_binary_ga(seed, gens=gens, parent_sel="tournament")["best"] for seed in seeds]
    no_hammer = [explore_no_hammer(m)["cost"] for m in old]
    hammer = [evaluate(m.astype(np.int8))["cost"] for m in old]       # 1 = breakable
    results["old_mazes"] = dict(no_hammer=no_hammer, hammer=hammer)
    say(f"[{time.time() - t0:.0f}s] old mazes (evolved without hammer): cost {np.mean(no_hammer):.0f} -> "
        f"{np.mean(hammer):.0f} when every wall is breakable")

    # 3. Which walls carry the load? (best maze per seed, solid budget 40)
    lb = [load_bearing(r["best"]) for r in runs[40]]
    solid_drop = [d for x in lb for d in x[SOLID]]
    break_drop = [d for x in lb for d in x[BREAK]]
    results["load_bearing_40"] = dict(solid_mean=float(np.mean(solid_drop)), breakable_mean=float(np.mean(break_drop)),
                                      solid_share_drop_ge_20=float(np.mean(np.array(solid_drop) >= 20)),
                                      breakable_share_drop_ge_20=float(np.mean(np.array(break_drop) >= 20)))
    best40 = max(runs[40], key=lambda r: r["fit"])["best"]
    res = explore(best40)
    trail_adj = set(v for p in res["trail"] for v in neighbours(p, best40.shape[0]))
    for kind, name in ((SOLID, "solid"), (BREAK, "breakable")):
        cells = np.flatnonzero(best40.ravel() == kind)
        results["load_bearing_40"][f"{name}_touching_route"] = float(np.mean([c in trail_adj for c in cells]))
    say(f"[{time.time() - t0:.0f}s] budget 40: removing a solid wall drops cost by {np.mean(solid_drop):.1f} "
        f"on average; removing a breakable wall drops it by {np.mean(break_drop):.1f}")
    say(f"   share of walls whose change drops cost by >= 20: solid "
        f"{results['load_bearing_40']['solid_share_drop_ge_20']:.0%}, breakable "
        f"{results['load_bearing_40']['breakable_share_drop_ge_20']:.0%}")

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    ax = axes[0]
    x = np.arange(len(budgets))
    for i, s in enumerate(budgets):
        ax.scatter([i] * len(seeds), [r["fit"] for r in runs[s]], color=BLUE, alpha=0.5, s=18)
    ax.plot(x, [np.mean([r["fit"] for r in runs[s]]) for s in budgets], color=BLUE, marker="o")
    ax.axhline(np.mean(no_hammer), color=INK2, ls="--", lw=1)
    ax.text(0, np.mean(no_hammer), " old mazes, robot without hammer", va="bottom", fontsize=8, color=INK2)
    ax.axhline(np.mean(hammer), color=ORANGE, ls="--", lw=1)
    ax.text(0, np.mean(hammer), " old mazes, robot with hammer", va="bottom", fontsize=8, color=ORANGE)
    ax.set_xticks(x, budgets)
    ax.set(xlabel="solid wall budget (of 160 walls)", ylabel="robot cost of best maze",
           title="More solid walls, harder maze")

    ax = axes[1]
    n = best40.shape[0]
    img = np.empty(best40.shape + (3,)); img[:] = to_rgb(SURFACE)
    img[np.asarray(res["seen"]).reshape(best40.shape).astype(bool)] = to_rgb(INSPECTED_C)
    img[best40 == BREAK] = to_rgb("#b9a58c")
    img[best40 == SOLID] = to_rgb(WALL_C)
    ax.imshow(img, interpolation="nearest")
    rc = np.array([divmod(p, n) for p in res["trail"]])
    ax.plot(rc[:, 1], rc[:, 0], color=ORANGE, lw=1.4)
    for c in res["smashed"]:
        r, cc = divmod(c, n); ax.plot(cc, r, "x", color="#8a5cd6", ms=6, mew=2)
    ax.set_title(f"Best maze, 40 solid walls (dark), 120 breakable (tan)\ncost {res['cost']} · "
                 f"{len(res['smashed'])} smashed (×)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    ax = axes[2]
    bins = np.arange(min(solid_drop + break_drop) - 5, max(solid_drop + break_drop) + 10, 10)
    ax.hist(break_drop, bins=bins, color="#b9a58c", alpha=0.8, label="remove a breakable wall")
    ax.hist(solid_drop, bins=bins, color=WALL_C, alpha=0.7, label="remove a solid wall")
    ax.set(xlabel="drop in robot cost after removing one wall", ylabel="walls", title="Which walls carry the load?")
    ax.legend(fontsize=8)
    fig.suptitle(f"Breakable walls: the maze decides where the hammer works (smash = {SMASH} actions)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "b1_breakable.png", dpi=150)
    plt.close(fig)

    results["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    (OUT / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    main(p.parse_args().quick)
