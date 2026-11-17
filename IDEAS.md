# Complexifying the project: ideas, what was built, what we found

## The problem with the current setup

The maze GA has one lever: put walls in the way. The first project's fitness (steps + tiles inspected)
rewards any maze that makes one fixed robot walk a lot, and the budget cap of 160 walls is the only thing that
stops it from turning every maze into a dead-end farm. The result is a solved-looking optimisation with one
opponent that never changes.

Three ways to make it a richer problem, all implemented in this branch:

1. **Give the robot something to optimise besides "reach the goal".** Coins on the map, a step budget, and a
   policy the robot can *evolve*.
2. **Evolve both sides.** Mazes and robots coevolve, so each side's best strategy depends on the other.
3. **Change what "hard" means for a maze.** A maze should be rewarded for beating a robot's *policy*, not for
   being brutal. We use regret (the idea behind PAIRED).

## What was built

| File | What it does |
|---|---|
| `coin_robot.py` | Coins in mazes, the evolvable coin robot, the oracle, operators for mazes with coins |
| `coevolution.py` | Robot evolution strategy, maze GA with two objectives, competitive coevolution with hall of fame |
| `experiments_coin.py` | Runs everything, writes 6 figures and `results_coin.json` into `out/coin/` |
| `test_coin_robot.py` | Checks the oracle against brute force and that no robot can beat the oracle |

```
python experiments_coin.py --quick     # about 1 minute
python experiments_coin.py             # about 6 minutes on 2 cores
python test_coin_robot.py              # about 2 seconds
```

No new dependencies (numpy and matplotlib only).

### The coin robot

The robot has a budget of 85 steps (the shortest start-to-goal path is about 40). It must reach the goal within the
budget. Each coin it picks up on the way scores one point, **but a robot that runs out of steps scores 0**, so
greed is risky. As before, it does not know the maze: it assumes unseen tiles are open and inspects its 4
neighbours. New: a coin detector senses coins within 4 tiles (straight-line, through walls), so a coin can look
close but be far by the real route.

A policy is five numbers, which is the robot genome:

| Gene | Meaning |
|---|---|
| `coin_pull` | how much it likes a sensed coin |
| `explore_pull` | how much it likes the edge of the known map |
| `alpha` | how fast liking falls with walking distance |
| `margin` | fraction of the budget kept in reserve to get home |
| `goal_bias` | how much exploring is steered toward the goal side |

Each time it needs a target, it scores every affordable coin and frontier tile and walks to the best one; if
nothing is affordable it heads to the goal. Robots evolve with a (μ, λ) or (μ+λ) evolution strategy with
self-adaptive step size.

### The oracle

A robot that knows the whole map, solved exactly (dynamic programming over subsets of 8 coins, an orienteering
problem). Nobody can beat it, which `test_coin_robot.py` checks, and it is checked against brute force on 53
mazes with zero mismatches. It is used to reject unusable mazes and as a ceiling for reporting.

### Why regret against *other robots*, not the oracle

We tried regret against the oracle first. It saturated at the maximum within 20 generations (oracle collects 8
coins, robot 0), because a robot that cannot see the map can never close the information gap. That makes it the
same "just make it harder" trap as before. So:

> **regret of a maze = (best score any robot in the pool gets) − (the target robot's score)**

A maze nobody can handle earns nothing. A maze scores high only if the target fails where a differently-behaving
robot succeeds, so it has to exploit a weakness of the target's policy. This is the role the antagonist plays in
PAIRED.

## Results

All numbers are from `out/coin/results_coin.json`. Budget 85, 21×21 mazes, 8 coins.

### 1. Robot evolution works, modestly (`fig6`)

Mean coins on 30 held-out random mazes: random policies **1.6**, hand-written policy **3.3**, evolved robots
**3.8** (6 independent runs: 3.6 to 4.0), oracle **7.2**. The evolved policies agree on one thing (a fairly high
`coin_pull`, 0.4 to 0.7) and disagree on the rest (`alpha` ranges from 0.04 to 1.4), so several different
policies score about the same.

### 2. The maze objective matters more than wall count (`fig7`, `fig8`)

Same target robot, top 5 mazes per run, 3 runs per objective:

| Maze objective | Target robot | Best other robot | Oracle | Walls |
|---|---|---|---|---|
| minimise robot's score | 0.0 | 1.5 | 7.4 | 113 |
| maximise regret | 0.0 | **7.0** | 7.9 | 116 |

Both objectives beat the target completely. Under "minimise score" the mazes beat every robot too (best other:
1.5). Under regret, another robot still collects 7.0 of 7.9, so the maze is exploiting this robot's particular
policy. The wall counts are nearly identical (and close to a random maze's 107), so walls do not explain the
difference. `fig8` shows an example of each.

### 3. Coevolution gives robustness, not a smooth arms race (`fig9`, `fig10`, `fig11`)

Cross-test: robots that were never used to build the mazes in columns 2 and 3 (mean coins):

| Robot | Random mazes | Brutal mazes | Regret mazes |
|---|---|---|---|
| Hand-written | 3.3 | 1.0 | 3.6 |
| Evolved on random mazes (2 runs) | **4.0** | 1.1 | 1.5 |
| Coevolved, final champion (3 runs) | 2.9 | **3.1** | 3.2 |

- Evolving only on random mazes gives the best random-maze score but collapses on adversarial mazes.
- Coevolved robots handle adversarial mazes two to three times better than robots trained only on random mazes, at
  a cost of roughly one coin on random mazes. It is a robustness/peak-performance trade-off.
- The hand-written policy handles the regret mazes just as well (3.6). Those mazes were built against one evolved
  robot, not against it, so the regret mazes are not generally hard.
- The master tournament (`fig10`, run 0) does not show steady progress. The maze champions from generations 20 and
  40 score 0 against every robot champion from generation 20 onward, yet the generation-10 robot gets 7 coins on
  the first and 4 on the second: later robots have forgotten what earlier ones could do. Three other maze
  champions (generations 10, 15, 25) are solved by nearly every late robot (7 to 8 coins), and the generation-35
  robot fails on mazes that its neighbours in generations 30 and 40 solve. That is the cycling and forgetting that
  competitive coevolution is known for, and what a hall of fame is meant to counter.
- The champion maze's regret against the fixed benchmark robots is noisy and does not rise over 40 generations
  (final values 0, 0 and 2.4 across the 3 runs).

## Caveats to put in the report

- **Small samples.** 3 coevolution runs, 2 fresh evolved robots, 1 hand-written robot. No significance tests were
  run. Run at least 10 seeds and use Mann-Whitney (as for fig2) before claiming anything from the cross-test.
- **Settings were tuned after a pilot.** The sensor radius (4) and budget (85) were chosen after the first pilot
  showed the problem saturating (robot 0 coins, oracle 8), and the oracle-regret objective was dropped for the
  same reason. Report this honestly.
- **The hard-maze benchmark is built against one robot** (robot A), so it measures robustness to one kind of
  trap, not to adversaries in general.
- **No hall-of-fame ablation.** We use a hall of fame but did not test coevolution without it, so we cannot say
  it helps.
- **The robot's genome is only 5 numbers.** Policies that score about the same differ a lot in their genes, so
  the landscape has many near-equal optima and 25 generations shows only a small gain over the hand-written policy.

## Ideas not implemented, roughly by effort

| Idea | Effort | Why it is interesting |
|---|---|---|
| **Hall-of-fame ablation and more seeds** | low | Turns the coevolution result into a real finding. The 2024 comparison of coevolution methods below found that archive-free methods can do better, so this is testable. |
| **Multi-objective robot (NSGA-II)**: coins, steps used, chance of failing | low–medium | Reuses `run_nsga2` from `maze_ea.py`. The front shows the greed/safety trade-off that `margin` currently hides. |
| **Coin values, poison tiles** | low | Gives the policy more to learn; `coin_pull` becomes `coin_pull × value`. |
| **Wall decay during the run** (links to fig4) | medium | Mazes change under the robot, so a map-memory robot is punished. |
| **Neural or GP controller** instead of 5 numbers | medium–high | Richer behaviour; maze-solver evolution with neural nets is established (Nitschke). |
| **Minimal criterion coevolution** | medium | No fitness at all: a maze survives if some robot solves it, a robot if it solves some maze. Open-ended, avoids the regret bookkeeping. |
| **Quality-diversity (MAP-Elites) over mazes**, with behaviour = (coins taken, steps used) | medium | Gives a map of *kinds* of difficulty instead of one hardest maze. |
| **Moving guards (tower defense framing)** | high | The README's framing becomes literal; adds a second adversary. |

## References

Only sources I opened or whose titles I saw in searches:

- Dennis et al. (2020), *Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design* (PAIRED):
  [arXiv 2012.02096](https://arxiv.org/pdf/2012.02096), summary at the
  [Google Research blog](https://research.google/blog/paired-a-new-multi-agent-approach-for-adversarial-environment-generation/).
  The maze adversary maximises regret, the gap between an antagonist's and the protagonist's reward.
- *Global progress in competitive co-evolution: a systematic comparison of alternative methods* (2024), Frontiers in
  Robotics and AI: [link](https://www.frontiersin.org/articles/10.3389/frobt.2024.1470886/full). Compares archive,
  bounded-archive and no-archive coevolution, and introduces master-tournament plots for measuring progress.
- Cliff & Miller, *Tracking the Red Queen: measurements of adaptive progress in co-evolutionary simulations*
  (the CIAO plot idea): [PDF](https://eprints.soton.ac.uk/261441/2/ciao.pdf).
- Wang et al. (2019), *Paired Open-Ended Trailblazer (POET)*: [arXiv 1901.01753](https://arxiv.org/pdf/1901.01753).
  Coevolves environments and their solvers.
- Brant & Stanley (2017), *Minimal Criterion Coevolution: A New Approach To Open-Ended Search*:
  [link](https://stars.library.ucf.edu/scopus2015/7496).
- Nitschke (2015), *Evolving Generalised Maze Solvers*:
  [PDF](https://people.cs.uct.ac.za/~gnitschke/projects/papers/2015-Evolving%20Generalised%20Maze%20Solvers.pdf).
- Vansteenwegen et al. (2011), *The orienteering problem: A survey*, EJOR:
  [link](https://econpapers.repec.org/RePEc:eee:ejores:v:209:y:2011:i:1:p:1-10). The oracle solves a small
  orienteering problem.
- *Co-generation of game levels and game-playing agents* (2020): [arXiv 2007.08497](https://arxiv.org/pdf/2007.08497).
