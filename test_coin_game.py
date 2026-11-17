"""Invariant checks for coin_game.py.   uv run python test_coin_game.py"""
import numpy as np

from coin_game import GREEDY, IGNORE_COINS, coin_detours, play, random_coins, repair_coins
from maze_ea import explore, random_maze, repair_budget, walls_to_open


def solvable_mazes(count, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        m = repair_budget(rng, random_maze(rng, 21), 160)
        if walls_to_open(m) == 0:
            out.append(m)
    return out


def test_no_coins_equals_original_robot():
    for m in solvable_mazes(15):
        assert play(m, [], IGNORE_COINS)["steps"] == explore(m)["steps"]


def test_every_match_finishes_and_cost_adds_up():
    rng = np.random.default_rng(1)
    for m in solvable_mazes(15, seed=1):
        for genes in (IGNORE_COINS, GREEDY, rng.random(3)):
            r = play(m, random_coins(rng, m, 10), genes, record=True)
            assert r["finished"]
            assert r["cost"] == r["steps"] + r["smashed"]
            assert r["trail"][0] == 0 and r["trail"][-1] == m.size - 1
            for a, b in zip(r["trail"], r["trail"][1:]):      # 4-connected moves only
                assert abs(a - b) in (1, 21)


def test_robot_never_walks_through_unsmashed_wall():
    rng = np.random.default_rng(2)
    for m in solvable_mazes(10, seed=2):
        r = play(m, random_coins(rng, m, 10), GREEDY, record=True)
        smashed = {r["trail"][s + 1] for s, kind in r["events"] if kind == "smash"}
        assert all(m.ravel()[p] == 0 or p in smashed for p in r["trail"])


def test_closed_shop_never_smashes():
    rng = np.random.default_rng(3)
    for m in solvable_mazes(10, seed=3):
        r = play(m, random_coins(rng, m, 10), GREEDY, shop=False)
        assert r["smashed"] == 0 and r["radius"] == 1


def test_repair_coins_keeps_k_distinct_floor_coins():
    rng = np.random.default_rng(4)
    m = solvable_mazes(1, seed=4)[0]
    coins = repair_coins(rng, m, [0, 0, 5, 440] + list(range(30, 36)), 10)
    assert len(set(coins)) == 10
    assert all(m.ravel()[c] == 0 and 0 < c < m.size - 1 for c in coins)
    assert all(d >= 0 for d in coin_detours(m, coins))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
