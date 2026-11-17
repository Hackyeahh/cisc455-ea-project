"""Invariant checks for breakable.py.   uv run python test_breakable.py"""
import numpy as np

from breakable import BREAK, FLOOR, SOLID, evaluate, explore, mutate, random_maze, repair
from maze_ea import explore as explore_no_hammer, random_maze as random_binary, repair_budget, walls_to_open


def test_all_solid_equals_original_robot():
    rng = np.random.default_rng(0)
    done = 0
    while done < 15:
        m = repair_budget(rng, random_binary(rng, 21), 160)
        if walls_to_open(m) == 0:
            old, new = explore_no_hammer(m), explore((m * SOLID).astype(np.int8))
            assert old["trail"] == new["trail"] and old["cost"] == new["cost"]
            done += 1


def test_route_valid_and_cost_adds_up():
    rng = np.random.default_rng(1)
    for _ in range(20):
        m = repair(rng, random_maze(rng, 21), 160, 40)
        r = evaluate(m)
        if r["blocked"]:
            continue
        flat = m.ravel()
        assert all(flat[p] != SOLID for p in r["trail"])
        assert all(flat[p] == FLOOR or p in r["smashed"] for p in r["trail"])
        assert r["cost"] == r["steps"] + 4 * len(r["smashed"]) + r["inspections"]


def test_repair_and_mutation_keep_budgets_and_symbols():
    rng = np.random.default_rng(2)
    for _ in range(50):
        m = repair(rng, mutate(rng, random_maze(rng, 21, p_wall=0.6, p_solid=0.7), 0.1), 160, 40)
        assert (m != FLOOR).sum() <= 160 and (m == SOLID).sum() <= 40
        assert set(np.unique(m)) <= {FLOOR, BREAK, SOLID}
        assert m[0, 0] == FLOOR and m[-1, -1] == FLOOR


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
