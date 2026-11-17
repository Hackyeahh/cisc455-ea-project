"""Coin game experiments: upgrades, evolved robots, coevolution, bait coins.

    uv run python coin_experiments.py            # full run
    uv run python coin_experiments.py --quick    # smoke test

Output goes to out/coins/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from coin_game import (GENE_NAMES, GREEDY, IGNORE_COINS, coevolve, coin_detours, decode, evolve_robots,
                       match_cost, play, random_coins)
from experiments import (AQUA, BLUE, INK, INK2, INSPECTED_C, ORANGE, SURFACE, WALL_C, band_plot,
                         budget_random_mazes, plt)
from maze_ea import WALL, run_ga, walls_to_open

OUT = Path(__file__).parent / "out" / "coins"
N, BUDGET, K = 21, 160, 10
GOLD, PURPLE = "#d9a400", "#8a5cd6"


def gene_str(g):
    d = decode(g)
    return f"greed {d['greed']:.1f} · sensors {d['sensors']} · smash {d['smash']:.1f}"


def draw_match(ax, grid, coins, genes, title):
    res = play(grid, coins, genes, record=True)
    n = grid.shape[0]
    img = np.empty(grid.shape + (3,))
    img[:] = to_rgb(SURFACE)
    img[np.asarray(res["seen"]).reshape(grid.shape).astype(bool)] = to_rgb(INSPECTED_C)
    img[grid == WALL] = to_rgb(WALL_C)
    ax.clear()
    ax.imshow(img, interpolation="nearest")
    rc = np.array([divmod(p, n) for p in res["trail"]])
    ax.plot(rc[:, 1], rc[:, 0], color=ORANGE, lw=1.5, alpha=0.9)
    taken = set(res["trail"])
    for c in coins:
        r, cc = divmod(c, n)
        ax.plot(cc, r, "o", ms=5.5, mfc=GOLD if c in taken else "none", mec=GOLD, mew=1.4)
    for step, kind in res["events"]:
        if kind == "smash":
            r, cc = divmod(res["trail"][step + 1], n)
            ax.plot(cc, r, "x", ms=6, mew=2, color=PURPLE)
    for (r, c), label in (((0, 0), "S"), ((n - 1, n - 1), "G")):
        ax.text(c, r, label, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
    ax.set_title(f"{title}\ncost {res['cost']} · {res['collected']} coins · {res['smashed']} smashed · "
                 f"radius {res['radius']}", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return res


def legend(fig):
    fig.legend(handles=[
        Line2D([], [], color=ORANGE, lw=2, label="robot route"),
        Line2D([], [], marker="o", ls="", mfc=GOLD, mec=GOLD, label="coin taken"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=GOLD, label="coin left"),
        Line2D([], [], marker="x", ls="", color=PURPLE, mew=2, label="wall smashed"),
    ], loc="lower center", ncol=4, fontsize=8.5)


def test_cost(genes, mazes, layouts, shop=True):
    return float(np.mean([match_cost(m, cs, genes, shop) for m, ls in zip(mazes, layouts) for cs in ls]))


def main(quick):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log, results = [], {}

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    seeds = [0, 1] if quick else [0, 1, 2]
    ga_gens, robo_gens, co_gens = (30, 10, 12) if quick else (150, 40, 80)
    layouts_per_maze = 3 if quick else 6

    # ------------------------------------------------------------------ test sets
    rand_mazes = [m for m in budget_random_mazes(40) if walls_to_open(m) == 0][:10]
    anti_explorer = [run_ga(s, gens=ga_gens, parent_sel="tournament")["best"] for s in seeds]
    say(f"[{time.time() - t0:.0f}s] test sets ready ({len(anti_explorer)} evolved mazes)")

    def layouts(mazes, base):
        return [[random_coins(np.random.default_rng((base, i, s)), m, K) for s in range(layouts_per_maze)]
                for i, m in enumerate(mazes)]

    # held-out layouts (different seeds from training) for fair testing
    train_r, test_r = layouts(rand_mazes, 1), layouts(rand_mazes, 2)
    train_e, test_e = layouts(anti_explorer, 1), layouts(anti_explorer, 2)

    # ------------------------------------------------------------------ E1: evolve a robot for each test set
    e1 = {}
    for name, mz, tr, te in (("random", rand_mazes, train_r, test_r), ("evolved", anti_explorer, train_e, test_e)):
        runs = [evolve_robots(s, mz, tr, gens=robo_gens) for s in seeds]
        best = min(runs, key=lambda r: r["best_cost"])["best"]
        row = {"no coins (original robot)": float(np.mean([match_cost(m, [], IGNORE_COINS) for m in mz])),
               "never detours": test_cost(IGNORE_COINS, mz, te),
               "greedy": test_cost(GREEDY, mz, te),
               "evolved robot": test_cost(best, mz, te),
               "evolved robot, shop closed": test_cost(best, mz, te, shop=False)}
        e1[name] = dict(costs=row, best_genes=[float(x) for x in best], decoded=decode(best),
                        histories=[r["history"].tolist() for r in runs])
        say(f"[{time.time() - t0:.0f}s] E1 {name} mazes: " + ", ".join(f"{k} {v:.0f}" for k, v in row.items())
            + f" | best robot: {gene_str(best)}")
    results["E1_robot_vs_fixed_mazes"] = e1

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    m, cs = anti_explorer[0], test_e[0][0]
    draw_match(axes[0], m, [], IGNORE_COINS, "Original robot, no coins")
    draw_match(axes[1], m, cs, GREEDY, "Greedy robot")
    draw_match(axes[2], m, cs, e1["evolved"]["best_genes"], "Evolved robot\n" + gene_str(e1["evolved"]["best_genes"]))
    legend(fig)
    fig.suptitle("E1. Coins on a maze evolved against the original robot", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT / "c1_upgrades.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ E2: coevolution, random coins
    co = [coevolve(s, gens=co_gens) for s in seeds]
    say(f"[{time.time() - t0:.0f}s] E2 coevolution done")
    one_sided = [coevolve(s, gens=co_gens, fixed_robot=e1["evolved"]["best_genes"]) for s in seeds]
    say(f"[{time.time() - t0:.0f}s] E2 one-sided baseline done")

    # Final champions, judged on a common held-out test: last 5 champions of every run
    co_mazes = [c["champs"][-1]["maze"] for c in co]
    co_robot = co[0]["champs"][-1]["robot"]
    one_mazes = [c["champs"][-1]["maze"] for c in one_sided]
    test_co, test_one = layouts(co_mazes, 3), layouts(one_mazes, 3)
    robots = {"original (no detour)": IGNORE_COINS, "greedy": GREEDY,
              "E1 robot": e1["evolved"]["best_genes"]}
    for i, c in enumerate(co):
        robots[f"coevolved robot s{i}"] = c["champs"][-1]["robot"]
    table = {}
    for rname, g in robots.items():
        table[rname] = {"coevolved mazes": test_cost(g, co_mazes, test_co),
                        "mazes evolved vs E1 robot": test_cost(g, one_mazes, test_one),
                        "anti-explorer mazes": test_cost(g, anti_explorer, test_e),
                        "random mazes": test_cost(g, rand_mazes, test_r)}
        say(f"   {rname:24s} " + "  ".join(f"{k}: {v:6.1f}" for k, v in table[rname].items()))
    results["E2_cross_table"] = table
    results["E2_final_genes"] = {f"seed {i}": decode(c["champs"][-1]["robot"]) for i, c in enumerate(co)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    band_plot(axes[0], [c["hist"]["maze_best"] for c in co], BLUE, "best maze (max cost)")
    band_plot(axes[0], [c["hist"]["robot_best"] for c in co], ORANGE, "best robot (min cost)")
    band_plot(axes[0], [c["hist"]["maze_best"] for c in one_sided], INK2, "maze vs fixed E1 robot")
    axes[0].set(xlabel="generation", ylabel="robot cost (in-generation matches)",
                title="Arms race: fitness in each generation")
    for gi, (gname, col) in enumerate(zip(GENE_NAMES, (GOLD, BLUE, PURPLE))):
        band_plot(axes[1], [c["hist"]["genes"][:, gi] for c in co], col, gname)
    for t in list(axes[1].texts):   # end labels overlap here, use a legend instead
        t.remove()
    axes[1].legend(fontsize=8)
    axes[1].set(xlabel="generation", ylabel="mean gene value (0–1)", ylim=(0, 1),
                title="Robot population: mean genes")
    fig.suptitle("E2. Competitive coevolution: mazes vs robots (hall of fame, random coins)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "c2_arms_race.png", dpi=150)
    plt.close(fig)

    # CIAO plot: champion maze of gen i vs champion robot of gen j (seed 0)
    champs = co[0]["champs"]
    pick = np.linspace(0, len(champs) - 1, min(16, len(champs))).astype(int)
    ciao_coins = [random_coins(np.random.default_rng((5, s)), champs[i]["maze"], K) for i in pick for s in range(3)]
    C = np.zeros((len(pick), len(pick)))
    for a, i in enumerate(pick):
        for b, j in enumerate(pick):
            C[a, b] = np.mean([match_cost(champs[i]["maze"], cs, champs[j]["robot"])
                               for cs in ciao_coins[3 * a:3 * a + 3]])
    results["E2_ciao"] = dict(gens=pick.tolist(), cost=C.tolist())
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(C, cmap="viridis", origin="lower")
    ax.set_xticks(range(len(pick)), pick, fontsize=7); ax.set_yticks(range(len(pick)), pick, fontsize=7)
    ax.set(xlabel="robot champion of generation", ylabel="maze champion of generation", title=
           "E2. CIAO plot (seed 0): robot cost\nbright = maze wins, dark = robot wins")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "c3_ciao.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ E3: bait coins (maze places the coins)
    bait = [coevolve(s, gens=co_gens, bait=True) for s in seeds]
    bait_vs_greedy = [coevolve(s, gens=co_gens, bait=True, fixed_robot=GREEDY) for s in seeds]
    rand_vs_greedy = [coevolve(s, gens=co_gens, fixed_robot=GREEDY) for s in seeds]
    say(f"[{time.time() - t0:.0f}s] E3 bait runs done")

    def detour_stats(runs, placed):
        d = []
        for r in runs:
            ch = r["champs"][-1]
            cs = ch["coins"] if placed else random_coins(np.random.default_rng(11), ch["maze"], K)
            d += coin_detours(ch["maze"], cs)
        return d

    det_bait = detour_stats(bait_vs_greedy, True)
    det_rand = detour_stats(rand_vs_greedy, False)
    greedy_bait = test_cost(GREEDY, [r["champs"][-1]["maze"] for r in bait_vs_greedy],
                            [[r["champs"][-1]["coins"]] for r in bait_vs_greedy])
    greedy_rand = float(np.mean([r["hist"]["maze_best"][-1] for r in rand_vs_greedy]))
    nocoin_bait = float(np.mean([match_cost(r["champs"][-1]["maze"], [], GREEDY) for r in bait_vs_greedy]))
    e3 = dict(greedy_cost_bait=greedy_bait, greedy_cost_random_coins=greedy_rand,
              greedy_cost_bait_mazes_without_coins=nocoin_bait,
              median_detour_bait=float(np.median(det_bait)), median_detour_random=float(np.median(det_rand)),
              final_greed_bait=[decode(r["champs"][-1]["robot"])["greed"] for r in bait],
              final_greed_random=[decode(c["champs"][-1]["robot"])["greed"] for c in co])
    results["E3_bait"] = e3
    say(f"[{time.time() - t0:.0f}s] E3 greedy robot cost: bait {greedy_bait:.0f}, random coins {greedy_rand:.0f}, "
        f"bait maze with coins removed {nocoin_bait:.0f} | median coin detour: bait {e3['median_detour_bait']:.0f}, "
        f"random {e3['median_detour_random']:.0f}")
    say(f"   coevolved robot greed: bait {np.round(e3['final_greed_bait'], 1)}, "
        f"random coins {np.round(e3['final_greed_random'], 1)}")

    fig = plt.figure(figsize=(12, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    ch = bait_vs_greedy[0]["champs"][-1]
    draw_match(ax1, ch["maze"], ch["coins"], GREEDY, "Maze + coins evolved vs greedy robot")
    ax2 = fig.add_subplot(1, 3, 2)
    bins = np.arange(0, max(det_bait + det_rand + [2]) + 4, 4)
    ax2.hist(det_rand, bins=bins, color=INK2, alpha=0.6, label="random coins")
    ax2.hist(det_bait, bins=bins, color=GOLD, alpha=0.75, label="maze-placed coins")
    ax2.set(xlabel="detour to pick up the coin (known map, steps)", ylabel="coins",
            title="Where the maze puts its coins")
    ax2.legend(fontsize=8)
    ax3 = fig.add_subplot(1, 3, 3)
    band_plot(ax3, [c["hist"]["genes"][:, 0] for c in co], INK2, "random coins")
    band_plot(ax3, [b["hist"]["genes"][:, 0] for b in bait], GOLD, "maze-placed coins")
    ax3.set(xlabel="generation", ylabel="mean greed gene (0–1)", ylim=(0, 1), title="Coevolved robots: greed")
    legend(fig)
    fig.suptitle("E3. Bait: the maze chooses where the coins go", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "c4_bait.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ examples + GIF
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    ch = co[0]["champs"][-1]
    cs = random_coins(np.random.default_rng(42), ch["maze"], K)
    draw_match(axes[0], ch["maze"], [], IGNORE_COINS, "Original robot, no coins")
    draw_match(axes[1], ch["maze"], cs, GREEDY, "Greedy robot")
    draw_match(axes[2], ch["maze"], cs, ch["robot"], "Coevolved robot\n" + gene_str(ch["robot"]))
    legend(fig)
    fig.suptitle("Champion maze of coevolution (seed 0), same coins in panels 2 and 3", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT / "c5_coevolved_examples.png", dpi=150)
    plt.close(fig)

    res = play(ch["maze"], cs, ch["robot"], record=True)
    trail, n = res["trail"], N
    fig, ax = plt.subplots(figsize=(4.4, 4.6))
    stride = max(1, len(trail) // 80)
    frames = list(range(1, len(trail), stride)) + [len(trail)]

    def frame(t):
        ax.clear()
        grid = ch["maze"].copy()
        sm = {res["trail"][s + 1] for s, k in res["events"] if k == "smash" and s + 1 < t}
        for c in sm:
            grid.ravel()[c] = 0
        img = np.empty(grid.shape + (3,)); img[:] = to_rgb(SURFACE)
        img[grid == WALL] = to_rgb(WALL_C)
        ax.imshow(img, interpolation="nearest")
        rc = np.array([divmod(p, n) for p in trail[:t]])
        ax.plot(rc[:, 1], rc[:, 0], color=ORANGE, lw=1.5)
        ax.plot(rc[-1, 1], rc[-1, 0], "s", ms=6, color=ORANGE)
        taken = set(trail[:t])
        for c in cs:
            if c not in taken:
                r, cc = divmod(c, n); ax.plot(cc, r, "o", ms=5, mfc=GOLD, mec=GOLD)
        for c in sm:
            r, cc = divmod(c, n); ax.plot(cc, r, "x", ms=6, mew=2, color=PURPLE)
        ax.set_title(f"coevolved robot · action {t - 1}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    FuncAnimation(fig, frame, frames=frames).save(OUT / "coin_run.gif", writer=PillowWriter(fps=10))
    plt.close(fig)

    results["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    (OUT / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    main(p.parse_args().quick)
