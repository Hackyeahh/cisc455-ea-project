"""Run every experiment and write figures + results.json into ./out.

    uv run python experiments.py            # full run (a few minutes, single core)
    uv run python experiments.py --quick    # smoke test
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from maze_ea import (FLOOR, MUD, SOLVERS, WALL, build_cost, decay, evaluate, random_maze,
                     repair_budget, run_ga, run_nsga2)

OUT = Path(__file__).parent / "out"
N, BUDGET = 21, 160

# Palette (validated reference palette, light mode)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WALL_C, MUD_C = "#2b2a28", "#c9a66b"
EXPLORE_EARLY, EXPLORE_LATE = np.array(to_rgb("#cde2fb")), np.array(to_rgb("#256abf"))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


# ----------------------------------------------------------------------------- drawing
def maze_image(grid, res):
    img = np.empty(grid.shape + (3,))
    img[grid == FLOOR] = to_rgb(SURFACE)
    img[grid == WALL] = to_rgb(WALL_C)
    img[grid == MUD] = to_rgb(MUD_C)
    n, order = grid.shape[0], res["expanded"]
    for k, cell in enumerate(order):
        r, c = divmod(cell, n)
        shade = EXPLORE_EARLY + (EXPLORE_LATE - EXPLORE_EARLY) * k / max(1, len(order) - 1)
        mix = 0.55 if grid[r, c] == MUD else 1.0   # keep mud visible under the overlay
        img[r, c] = mix * shade + (1 - mix) * img[r, c]
    return img


def draw_maze(ax, grid, title, solver="A*", compact=False):
    res = evaluate(grid, solver)
    ax.clear()
    ax.imshow(maze_image(grid, res), interpolation="nearest")
    if res["path"]:
        rc = np.array([divmod(p, grid.shape[0]) for p in res["path"]])
        ax.plot(rc[:, 1], rc[:, 0], color=ORANGE, lw=2)
    for (r, c), label in (((0, 0), "S"), ((grid.shape[0] - 1,) * 2, "G")):
        ax.text(c, r, label, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
    stats = (f"{res['expansions']} expanded · path {res['cost']:.0f} · build {res['build_cost']}" if compact else
             f"{res['expansions']} nodes expanded · path cost {res['cost']:.0f} · build cost {res['build_cost']}")
    ax.set_title(f"{title}\n{stats}", fontsize=8 if compact else 9.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return res


MAZE_LEGEND = [Patch(color=WALL_C, label="Wall (cost 1)"), Patch(color=MUD_C, label="Mud (cost 2, slow ×5)"),
               Patch(color=EXPLORE_EARLY, label="Expanded early"), Patch(color=EXPLORE_LATE, label="Expanded late"),
               Line2D([], [], color=ORANGE, lw=2, label="A* path")]


def maze_legend(fig):
    fig.legend(handles=MAZE_LEGEND, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))


# ----------------------------------------------------------------------------- experiments
def budget_random_mazes(count, seed=999):
    rng = np.random.default_rng(seed)
    mazes = []
    while len(mazes) < count:
        m = repair_budget(rng, random_maze(rng, N), BUDGET)
        if evaluate(m)["blocked"] == 0:
            mazes.append(m)
    return mazes


def band_plot(ax, hists, color, label):
    """Mean line with min-max band across seeds; direct label at line end."""
    h = np.array(hists)
    x = np.arange(h.shape[1])
    ax.fill_between(x, h.min(0), h.max(0), color=color, alpha=0.15, lw=0)
    ax.plot(x, h.mean(0), color=color, label=label)
    ax.annotate(label, (x[-1], h.mean(0)[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)


def front_summary(runs, seeds):
    return [dict(seed=s, front=sorted({(int(e[0][1]), int(-e[0][0])) for _, e in r["front"] if e[1] == 0}))
            for s, r in zip(seeds, runs)]


def nsga_figure(run, record, ylabel, fname):
    fig = plt.figure(figsize=(12, 5.6))
    ax = fig.add_subplot(1, 2, 1)
    for gen, color in zip(record, ("#86b6ef", "#3987e5", "#184f95")):
        pts = sorted({(e[0][1], -e[0][0]) for _, e in run["recorded"][gen] if e[1] == 0})
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=color, ms=4, lw=1.5, label=f"Generation {gen}")
    ax.set(xlabel="Build cost (wall = 1, mud = 2)  →  cheaper is better",
           ylabel=f"{ylabel}  →  higher is better", title="NSGA-II first front (seed 0)")
    ax.legend(loc="lower right")
    seen, front = set(), []
    for grid, e in sorted(run["front"], key=lambda t: t[1][0][1]):
        if e[1] == 0 and e[0] not in seen:
            seen.add(e[0]); front.append((grid, e))
    idx = sorted(set(np.linspace(0, len(front) - 1, 4).round().astype(int)))
    for k, i in enumerate(idx):
        grid, e = front[i]
        ax.annotate(chr(65 + k), (e[0][1], -e[0][0]), xytext=(-12, 6), textcoords="offset points",
                    fontsize=10, fontweight="bold")
        sub = fig.add_subplot(2, 4, [3, 4, 7, 8][k])
        draw_maze(sub, grid, f"Front maze {chr(65 + k)}", compact=True)
    maze_legend(fig); fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / fname, dpi=150); plt.close(fig)


def main(quick):
    OUT.mkdir(exist_ok=True)
    gens = 30 if quick else 150
    seeds = [0, 1] if quick else [0, 1, 2, 3, 4]
    results, t0 = {}, time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    # --- 1. Single-objective GA: FPS (group's plan) vs tournament --------------------------
    ga = {sel: [run_ga(s, n=N, gens=gens, budget=BUDGET, parent_sel=sel, snapshot_every=5)
                for s in seeds] for sel in ("fps", "tournament")}
    log("GA runs done")
    randoms = budget_random_mazes(30)
    rand_exp = [evaluate(m)["expansions"] for m in randoms]
    results["random_budget_mazes"] = dict(mean=float(np.mean(rand_exp)), max=int(np.max(rand_exp)))
    for sel, runs in ga.items():
        bests = [evaluate(r["best"])["expansions"] for r in runs]
        results[f"ga_{sel}"] = dict(best_per_seed=bests, mean=float(np.mean(bests)),
                                    final_pop_mean=[float(r["fit"].mean()) for r in runs])

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    band_plot(ax, [r["history"][:, 0] for r in ga["tournament"]], BLUE, "Tournament (k=3)")
    band_plot(ax, [r["history"][:, 0] for r in ga["fps"]], ORANGE, "Roulette wheel (FPS)")
    ax.axhline(np.mean(rand_exp), color=INK2, lw=1, ls="--")
    ax.annotate("random maze, same budget", (0, np.mean(rand_exp)), xytext=(4, 4),
                textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set(xlabel="Generation", ylabel="Best A* nodes expanded",
           title=f"Parent selection: best fitness over generations (mean, band = min–max over {len(seeds)} seeds)")
    ax.legend(loc="center right")
    ax.set_xlim(0, gens * 1.18)
    fig.tight_layout(); fig.savefig(OUT / "fig2_ga_convergence.png", dpi=150); plt.close(fig)

    best_run = max(ga["tournament"], key=lambda r: r["fit"].max())
    median_rand = randoms[int(np.argsort(rand_exp)[len(rand_exp) // 2])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2))
    draw_maze(axes[0], median_rand, "Random maze (median of 30), same budget")
    draw_maze(axes[1], best_run["best"], "Evolved maze (GA, tournament)")
    maze_legend(fig); fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig1_random_vs_evolved.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    snaps = best_run["snapshots"]
    def frame(i):
        gen, grid = snaps[i]
        draw_maze(ax, grid, f"Generation {gen}")
    maze_legend(fig); fig.subplots_adjust(bottom=0.12, top=0.88)
    FuncAnimation(fig, frame, frames=len(snaps)).save(OUT / "evolution.gif", writer=PillowWriter(fps=4))
    plt.close(fig)
    log("GA figures done")

    # --- 2. NSGA-II: maximize expansions vs minimize build cost ----------------------------
    record = (0, gens // 3, gens)
    nsga = [run_nsga2(s, n=N, gens=gens, record_gens=record) for s in seeds[:3]]
    log("NSGA-II runs done")
    nsga_figure(nsga[0], record, "A* nodes expanded", "fig3_nsga2_front.png")
    results["nsga2_expansions"] = front_summary(nsga, seeds)
    log("NSGA-II figure done")

    # --- 3. Robust GA: fitness averaged over wall-decay samples -----------------------------
    robust = [run_ga(s, n=N, gens=gens, budget=BUDGET, parent_sel="tournament",
                     decay_k=3, decay_samples=5) for s in seeds[:3]]
    log("Robust GA runs done")

    # --- 4. Robustness test: random wall decay ---------------------------------------------
    ks = [0, 2, 4, 8, 16, 32]
    groups = {"Evolved (standard fitness)": ([r["best"] for r in ga["tournament"]], BLUE),
              "Evolved (decay in fitness)": ([r["best"] for r in robust], ORANGE),
              "Random, same budget": (randoms[:5], AQUA)}
    samples = 10 if quick else 30
    decay_res = {}
    for name, (mazes, _) in groups.items():
        decay_res[name] = [float(np.mean([evaluate(decay(np.random.default_rng((7, k, s, i)), m, k))["expansions"]
                                          for i, m in enumerate(mazes) for s in range(samples)])) for k in ks]
    results["decay"] = dict(walls_removed=ks, mean_expansions=decay_res)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, (_, color) in groups.items():
        ax.plot(ks, decay_res[name], "-o", color=color, ms=5, label=name)
        ax.annotate(name, (ks[-1], decay_res[name][-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9)
    ax.set(xlabel="Walls removed at random", ylabel="Mean A* nodes expanded",
           title=f"Wall decay test (mean over mazes × {samples} decay samples)")
    ax.set_xlim(0, ks[-1] * 1.45); ax.set_ylim(0)
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.62))
    fig.tight_layout(); fig.savefig(OUT / "fig4_decay.png", dpi=150); plt.close(fig)
    log("Decay test done")

    # --- 5. Solver pool test: was the maze overfit to one A*? -------------------------------
    pool = {}
    for name, (mazes, _) in groups.items():
        pool[name] = {s: float(np.mean([evaluate(m, s)["expansions"] for m in mazes])) for s in SOLVERS}
    results["solver_pool"] = pool
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.26
    x = np.arange(len(SOLVERS))
    for j, (name, (_, color)) in enumerate(groups.items()):
        vals = [pool[name][s] for s in SOLVERS]
        bars = ax.bar(x + (j - 1) * width, vals, width - 0.02, color=color, label=name)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, color=INK2, padding=2)
    ax.set_xticks(x, [f"{s}\n(trained on)" if s == "A*" else s for s in SOLVERS])
    ax.set(ylabel="Mean nodes expanded", title="Same mazes, different solvers")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "fig5_solver_pool.png", dpi=150); plt.close(fig)
    log("Solver pool test done")

    # --- 6. Alternative fitness: path cost (enemy travel time) -----------------------------
    ga_pc = [run_ga(s, n=N, gens=gens, budget=BUDGET, parent_sel="tournament", objective="path_cost")
             for s in seeds]
    log("Path-cost GA runs done")
    rand_pc = [evaluate(m)["cost"] for m in randoms]
    results["ga_path_cost"] = dict(
        best_path_cost=[evaluate(r["best"])["cost"] for r in ga_pc],
        best_expansions=[evaluate(r["best"])["expansions"] for r in ga_pc],
        random_path_cost_mean=float(np.mean(rand_pc)),
        expansion_mazes_path_cost=[evaluate(r["best"])["cost"] for r in ga["tournament"]])
    best_pc = max(ga_pc, key=lambda r: r["fit"].max())
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2))
    draw_maze(axes[0], best_run["best"], "Fitness = nodes expanded")
    draw_maze(axes[1], best_pc["best"], "Fitness = path cost (travel time)")
    maze_legend(fig); fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig6_fitness_comparison.png", dpi=150); plt.close(fig)
    nsga_pc = [run_nsga2(s, n=N, gens=gens, record_gens=record, objective="path_cost") for s in seeds[:3]]
    nsga_figure(nsga_pc[0], record, "A* path cost (travel time)", "fig7_nsga2_pathcost_front.png")
    results["nsga2_path_cost"] = front_summary(nsga_pc, seeds)
    log("Path-cost experiments done")

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    log(f"Wrote {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    main(p.parse_args().quick)
