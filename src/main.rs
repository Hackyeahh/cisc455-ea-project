//! Maze Ledger EA: evolve a maze for the points game in the Maze Ledger page.
//!
//! The robot is a step-for-step port of `simulate()` in the page, so a layout scores
//! the same here and in the browser.
//!
//!   ledger_ea eval <layout.txt>                      score one layout
//!   ledger_ea evolve [threads] [seconds] [out.txt]   island GA, one island per thread
//!
//! Genotype: 21x21 grid over {floor, wall, trap, coin}. Fitness: robot escape time
//! (steps + trap seconds - bonus x coins taken). Infeasible mazes rank below every
//! feasible maze: over budget -> -overspend, blocked -> -1000 - 10 x walls to open.
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const N: usize = 21;
const NN: usize = N * N;
const S: usize = 0;
const G: usize = NN - 1;
const FLOOR: u8 = 0;
const WALL: u8 = 1;
const TRAP: u8 = 2;
const COIN: u8 = 3;
const INF: i64 = i64::MAX / 4;

#[derive(Clone, Copy, Debug)]
struct Rules { base: i64, wall_cost: i64, trap_cost: i64, trap_delay: i64, coin_income: i64, coin_bonus: i64,
               max_coins: usize, safe_radius: usize, traps_hidden: bool }
const DEFAULT_RULES: Rules = Rules { base: 120, wall_cost: 1, trap_cost: 3, trap_delay: 5, coin_income: 4,
                                     coin_bonus: 6, max_coins: 6, safe_radius: 2, traps_hidden: false };

type Grid = [u8; NN];

fn nbrs(u: usize) -> ([usize; 4], usize) {
    // same order as the page: up, down, left, right
    let (r, c) = (u / N, u % N);
    let mut o = [0; 4];
    let mut k = 0;
    if r > 0 { o[k] = u - N; k += 1; }
    if r < N - 1 { o[k] = u + N; k += 1; }
    if c > 0 { o[k] = u - 1; k += 1; }
    if c < N - 1 { o[k] = u + 1; k += 1; }
    (o, k)
}

fn in_zone(u: usize, rules: &Rules) -> bool {
    let (r, c) = (u / N, u % N);
    r + c <= rules.safe_radius || (N - 1 - r) + (N - 1 - c) <= rules.safe_radius
}

// ------------------------------------------------------------------------------ robot
struct Dij { dist: [i64; NN], prev: [i32; NN] }

/// cost(v) = seconds to enter v. reverse=true gives the distance TO src. Ties pop in push
/// order (the page's heap breaks ties with a sequence number), so routes match exactly.
fn dijkstra(src: usize, known: &[i8; NN], delay: i64, reverse: bool, heap: &mut BinaryHeap<Reverse<(i64, u64, usize)>>) -> Dij {
    let cost = |v: usize| match known[v] { 1 => INF, 2 => 1 + delay, _ => 1 };
    let mut d = Dij { dist: [INF; NN], prev: [-1; NN] };
    heap.clear();
    let mut seq = 0u64;
    d.dist[src] = 0;
    heap.push(Reverse((0, seq, src)));
    seq += 1;
    while let Some(Reverse((du, _, u))) = heap.pop() {
        if du > d.dist[u] { continue; }
        let (nb, k) = nbrs(u);
        for &v in &nb[..k] {
            if cost(v) >= INF { continue; }
            let nd = du + if reverse { cost(u) } else { cost(v) };
            if nd < d.dist[v] {
                d.dist[v] = nd;
                d.prev[v] = u as i32;
                heap.push(Reverse((nd, seq, v)));
                seq += 1;
            }
        }
    }
    d
}

#[derive(Clone, Copy, Debug, Default)]
struct Outcome { steps: i64, trap_time: i64, collected: i64, coins_placed: i64, escape: i64, spent: i64, cap: i64, over: i64 }

fn spent(g: &Grid, r: &Rules) -> i64 {
    g.iter().map(|&t| match t { WALL => r.wall_cost, TRAP => r.trap_cost, _ => 0 }).sum()
}

fn simulate(grid: &Grid, rules: &Rules) -> Outcome {
    let mut known = [-1i8; NN];
    let mut coins: Vec<usize> = (0..NN).filter(|&i| grid[i] == COIN).collect();
    let coins_placed = coins.len() as i64;
    let (mut pos, mut steps, mut trap_time, mut collected) = (S, 0i64, 0i64, 0i64);
    let mut heap = BinaryHeap::with_capacity(4 * NN);
    let mut path: Vec<usize> = vec![S];

    let sense = |p: usize, known: &mut [i8; NN]| {
        let (nb, k) = nbrs(p);
        for &v in std::iter::once(&p).chain(nb[..k].iter()) {
            if known[v] != -1 { continue; }
            known[v] = match grid[v] {
                WALL => WALL as i8,
                TRAP if !rules.traps_hidden => TRAP as i8,
                _ => FLOOR as i8,
            };
        }
    };
    let decide = |pos: usize, known: &[i8; NN], coins: &Vec<usize>, path: &mut Vec<usize>, heap: &mut BinaryHeap<Reverse<(i64, u64, usize)>>| {
        let fwd = dijkstra(pos, known, rules.trap_delay, false, heap);
        let mut best = G;
        if !coins.is_empty() {
            let back = dijkstra(G, known, rules.trap_delay, true, heap);
            let base = fwd.dist[G];
            let mut best_d = INF;
            for &c in coins {
                let det = if fwd.dist[c] >= INF || back.dist[c] >= INF { INF } else { fwd.dist[c] + back.dist[c] - base };
                let ok = det < rules.coin_bonus;
                if ok && (det < best_d || (det == best_d && fwd.dist[c] < fwd.dist[best])) { best = c; best_d = det; }
            }
        }
        path.clear();
        let mut p = best;
        path.push(p);
        while p != pos {
            let q = fwd.prev[p];
            if q < 0 { path.clear(); path.push(pos); return; }
            p = q as usize;
            path.push(p);
        }
        path.reverse();
    };

    sense(pos, &mut known);
    decide(pos, &known, &coins, &mut path, &mut heap);
    while pos != G && steps < 5000 {
        if path.len() < 2 { break; }
        pos = path[1];
        steps += 1;
        if grid[pos] == TRAP { trap_time += rules.trap_delay; known[pos] = TRAP as i8; }
        if let Some(i) = coins.iter().position(|&c| c == pos) { coins.remove(i); collected += 1; }
        sense(pos, &mut known);
        if pos == G { break; }
        decide(pos, &known, &coins, &mut path, &mut heap);
    }
    let sp = spent(grid, rules);
    let cap = rules.base + rules.coin_income * collected;
    let escape = if pos == G { steps + trap_time - rules.coin_bonus * collected } else { -1 };
    Outcome { steps, trap_time, collected, coins_placed, escape, spent: sp, cap, over: sp - cap }
}

/// Fewest walls to remove so S reaches G (0-1 BFS over walls).
fn walls_to_open(g: &Grid) -> i64 {
    let mut dist = [i64::MAX; NN];
    let mut dq = std::collections::VecDeque::new();
    dist[S] = 0;
    dq.push_back(S);
    while let Some(u) = dq.pop_front() {
        let (nb, k) = nbrs(u);
        for &v in &nb[..k] {
            let w = (g[v] == WALL) as i64;
            if dist[u] + w < dist[v] {
                dist[v] = dist[u] + w;
                if w == 0 { dq.push_front(v) } else { dq.push_back(v) }
            }
        }
    }
    dist[G]
}

fn fitness(g: &Grid, rules: &Rules) -> (f64, Outcome) {
    let blocked = walls_to_open(g);
    if blocked > 0 { return (-1000.0 - 10.0 * blocked as f64, Outcome::default()); }
    let o = simulate(g, rules);
    if o.escape < 0 { return (-2000.0, o); }
    if o.over > 0 { return (-(o.over as f64), o); }
    (o.escape as f64, o)
}

// ------------------------------------------------------------------------------ RNG
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { // splitmix64
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

// ------------------------------------------------------------------------------ variation
fn repair(g: &mut Grid, rules: &Rules, rng: &mut Rng) {
    g[S] = FLOOR;
    g[G] = FLOOR;
    for u in 0..NN { if g[u] == TRAP && in_zone(u, rules) { g[u] = FLOOR; } }
    let mut coins: Vec<usize> = (0..NN).filter(|&u| g[u] == COIN).collect();
    while coins.len() > rules.max_coins { let i = rng.below(coins.len()); g[coins.swap_remove(i)] = FLOOR; }
    let cap = rules.base + rules.coin_income * coins.len() as i64;
    let mut items: Vec<usize> = (0..NN).filter(|&u| g[u] == WALL || g[u] == TRAP).collect();
    let mut sp = spent(g, rules);
    while sp > cap && !items.is_empty() {
        let i = rng.below(items.len());
        let u = items.swap_remove(i);
        sp -= if g[u] == WALL { rules.wall_cost } else { rules.trap_cost };
        g[u] = FLOOR;
    }
}

fn random_grid(rules: &Rules, rng: &mut Rng) -> Grid {
    let mut g = [FLOOR; NN];
    let pw = 0.15 + 0.2 * rng.f();
    for u in 0..NN {
        let x = rng.f();
        g[u] = if x < pw { WALL } else if x < pw + 0.01 { TRAP } else if x < pw + 0.02 { COIN } else { FLOOR };
    }
    repair(&mut g, rules, rng);
    g
}

fn crossover(a: &Grid, b: &Grid, rng: &mut Rng) -> Grid {
    let (mut c1, mut c2) = (1 + rng.below(NN - 1), 1 + rng.below(NN - 1));
    if c1 > c2 { std::mem::swap(&mut c1, &mut c2); }
    let mut k = *a;
    k[c1..c2].copy_from_slice(&b[c1..c2]);
    k
}

/// Each cell changes with probability `rate`. New symbol weights: floor .4, wall .4, trap .1, coin .1
/// (the current symbol is excluded). Plus, with prob 0.3, move one coin or trap to a random tile.
fn mutate(g: &mut Grid, rate: f64, rng: &mut Rng) {
    const W: [f64; 4] = [0.4, 0.4, 0.1, 0.1];
    for u in 0..NN {
        if rng.f() < rate {
            let cur = g[u] as usize;
            let total: f64 = (0..4).filter(|&s| s != cur).map(|s| W[s]).sum();
            let mut x = rng.f() * total;
            for s in 0..4 {
                if s == cur { continue; }
                if x < W[s] { g[u] = s as u8; break; }
                x -= W[s];
            }
        }
    }
    if rng.f() < 0.3 {
        let items: Vec<usize> = (0..NN).filter(|&u| g[u] == COIN || g[u] == TRAP).collect();
        if !items.is_empty() {
            let from = items[rng.below(items.len())];
            let to = rng.below(NN);
            if g[to] != COIN && g[to] != TRAP { g.swap(from, to); }
        }
    }
}

// ------------------------------------------------------------------------------ layout text (same format as the page)
fn layout_text(g: &Grid, r: &Rules, o: Option<&Outcome>) -> String {
    let mut s = format!("rules base={} wall={} trap={} delay={} income={} bonus={} maxCoins={} safe={} traps={}\n",
        r.base, r.wall_cost, r.trap_cost, r.trap_delay, r.coin_income, r.coin_bonus, r.max_coins, r.safe_radius,
        if r.traps_hidden { "hidden" } else { "adjacent" });
    if let Some(o) = o {
        s += &format!("result escape={}s steps={} trapTime={} coins={}/{} spent={} budget={}\n",
                      o.escape, o.steps, o.trap_time, o.collected, o.coins_placed, o.spent, o.cap);
    }
    for row in 0..N {
        for col in 0..N {
            let u = row * N + col;
            s.push(if u == S { 'S' } else if u == G { 'G' } else { ['.', '#', 'T', '$'][g[u] as usize] });
        }
        s.push('\n');
    }
    s
}

fn parse_layout(txt: &str) -> (Grid, Rules) {
    let mut r = DEFAULT_RULES;
    let mut g = [FLOOR; NN];
    let mut row = 0;
    for line in txt.lines() {
        let l = line.trim();
        if let Some(rest) = l.strip_prefix("rules") {
            for kv in rest.split_whitespace() {
                let mut it = kv.splitn(2, '=');
                let (k, v) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
                let n: i64 = v.parse().unwrap_or(0);
                match k { "base" => r.base = n, "wall" => r.wall_cost = n, "trap" => r.trap_cost = n, "delay" => r.trap_delay = n,
                          "income" => r.coin_income = n, "bonus" => r.coin_bonus = n, "maxCoins" => r.max_coins = n as usize,
                          "safe" => r.safe_radius = n as usize, "traps" => r.traps_hidden = v == "hidden", _ => {} }
            }
            continue;
        }
        let cells: Vec<char> = l.chars().filter(|c| !c.is_whitespace()).collect();
        if cells.len() == N && cells.iter().all(|c| ".#T$SG".contains(*c)) && row < N {
            for (c, ch) in cells.iter().enumerate() {
                g[row * N + c] = match ch { '#' => WALL, 'T' => TRAP, '$' => COIN, _ => FLOOR };
            }
            row += 1;
        }
    }
    assert_eq!(row, N, "layout needs {N} rows");
    (g, r)
}

// ------------------------------------------------------------------------------ island GA
struct Shared { best_fit: f64, best: Grid, best_out: Outcome, migrants: Vec<Option<Grid>>, evals: u64 }

fn island(id: usize, rules: Rules, secs: f64, shared: Arc<Mutex<Shared>>, n_islands: usize) {
    let mut rng = Rng(0xC1_5C45_5u64 ^ (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let (mu, lam, k, rate) = (60usize, 30usize, 3usize, 4.0 / NN as f64);
    let mut pop: Vec<Grid> = (0..mu).map(|_| random_grid(&rules, &mut rng)).collect();
    let mut fit: Vec<(f64, Outcome)> = pop.iter().map(|g| fitness(g, &rules)).collect();
    let t0 = Instant::now();
    let mut gen = 0u64;
    let mut local_evals = 0u64;
    while t0.elapsed().as_secs_f64() < secs {
        let mut kids = Vec::with_capacity(lam);
        for _ in 0..lam {
            let mut pick = || { let mut b = rng.below(mu); for _ in 1..k { let c = rng.below(mu); if fit[c].0 > fit[b].0 { b = c; } } b };
            let (a, b) = (pick(), pick());
            let mut kid = crossover(&pop[a], &pop[b], &mut rng);
            mutate(&mut kid, rate, &mut rng);
            repair(&mut kid, &rules, &mut rng);
            kids.push(kid);
        }
        let mut order: Vec<usize> = (0..mu).collect();
        order.sort_by(|&x, &y| fit[x].0.partial_cmp(&fit[y].0).unwrap());
        for (slot, kid) in order.into_iter().zip(kids) {
            let f = fitness(&kid, &rules);
            local_evals += 1;
            pop[slot] = kid;
            fit[slot] = f;
        }
        gen += 1;
        if gen % 50 == 0 {
            let bi = (0..mu).max_by(|&x, &y| fit[x].0.partial_cmp(&fit[y].0).unwrap()).unwrap();
            let mut sh = shared.lock().unwrap();
            sh.evals += local_evals;
            local_evals = 0;
            if fit[bi].0 > sh.best_fit { sh.best_fit = fit[bi].0; sh.best = pop[bi]; sh.best_out = fit[bi].1; }
            // ring migration: send my best to the next island, take the migrant waiting for me
            let next = (id + 1) % n_islands;
            sh.migrants[next] = Some(pop[bi]);
            if let Some(m) = sh.migrants[id].take() {
                let wi = (0..mu).min_by(|&x, &y| fit[x].0.partial_cmp(&fit[y].0).unwrap()).unwrap();
                pop[wi] = m;
                fit[wi] = fitness(&m, &rules);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("eval") => {
            let txt = std::fs::read_to_string(&args[2]).expect("read layout").replace("\r\n", "\n");
            for chunk in txt.split("\n\n").filter(|c| c.contains("rules")) {
                let (g, r) = parse_layout(chunk);
                let o = simulate(&g, &r);
                println!("escape={} steps={} trapTime={} collected={} spent={} cap={} over={}",
                         o.escape, o.steps, o.trap_time, o.collected, o.spent, o.cap, o.over);
            }
        }
        Some("evolve") => {
            let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
            let secs: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(300.0);
            let out = args.get(4).cloned().unwrap_or("best.txt".into());
            let rules = DEFAULT_RULES;
            let shared = Arc::new(Mutex::new(Shared { best_fit: f64::MIN, best: [FLOOR; NN], best_out: Outcome::default(),
                                                      migrants: vec![None; threads], evals: 0 }));
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for id in 0..threads {
                    let sh = shared.clone();
                    s.spawn(move || island(id, rules, secs, sh, threads));
                }
                let mut last = -1.0;
                while t0.elapsed().as_secs_f64() < secs + 1.0 {
                    std::thread::sleep(std::time::Duration::from_secs(10));
                    let sh = shared.lock().unwrap();
                    if sh.best_fit != last || true {
                        let o = sh.best_out;
                        println!("[{:>4.0}s] evals {:>8}  best escape {:>4}s  (steps {} traps +{} coins {}/{} spent {}/{})",
                                 t0.elapsed().as_secs_f64(), sh.evals, sh.best_fit, o.steps, o.trap_time, o.collected,
                                 o.coins_placed, o.spent, o.cap);
                        last = sh.best_fit;
                    }
                }
            });
            let sh = shared.lock().unwrap();
            let o = simulate(&sh.best, &rules);
            let txt = layout_text(&sh.best, &rules, Some(&o));
            std::fs::write(&out, &txt).expect("write");
            println!("\n{txt}");
        }
        _ => eprintln!("usage: ledger_ea eval <layout.txt> | ledger_ea evolve [threads] [seconds] [out.txt]"),
    }
}
