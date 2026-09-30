"""
Lab 3 - Search and A*: warehouse robot navigation
CS F407 Artificial Intelligence

Implements, from scratch and with the standard library only:

  * the warehouse as a search problem P = (S, A, T, s0, G, c)
  * A* search with a pluggable heuristic h(n), f(n) = g(n) + h(n)
  * breadth-first search (BFS), the blind baseline
  * Task 3 tests, Task 5 (BFS vs A*) and Task 6 (heuristic investigation)

Usage:
    python3 astar.py        # runs everything and writes results.txt
                            # (exit code 0 only if every test passes)

Counting convention (same for every algorithm): a state is "expanded" when it
is removed from the frontier and its successors are generated. The goal state
is recognised when it is removed from the frontier and is not counted as
expanded. Ties between equal f-values are broken by smaller h, then by
insertion order; successors are generated in the order Up, Down, Left, Right.
"""

import heapq
import io
import math
import os
import sys
from collections import deque
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))

# The warehouse from the handout.
WAREHOUSE = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""

# A small map used in Task 6 to show what an overestimating heuristic can do.
SMALL_MAP = """\
#########
#S.#.#..#
#....##.#
#.#.....#
#....#..#
##..#..G#
#########"""

# A mostly open room used in Task 5 (in addition to the lab map) to show when
# the heuristic does help.
OPEN_ROOM = """\
######################
#S...................#
#....................#
#...........####.....#
#...........#........#
#...........#........#
#....................#
#..................G.#
######################"""

# Action name -> (row change, column change). Every move costs 1.
ACTIONS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


# ---------------------------------------------------------------------------
# Task 0 / Task 1: the search problem
# ---------------------------------------------------------------------------
class SearchProblem:
    """P = (S, A, T, s0, G, c) for an ASCII warehouse map.

    State      S   : a cell (row, col) that is not '#'.
    Actions    A   : Up, Down, Left, Right.
    Transition T   : transition(s, a) = (row + dr, col + dc) when that cell is valid.
    Initial    s0  : the cell marked S.
    Goal       G   : {the cell marked G}; tested by is_goal().
    Cost       c   : 1 for every move.
    """

    def __init__(self, text):
        self.grid = [list(row) for row in text.strip("\n").splitlines()]
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, symbol):
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == symbol:
                    return (r, c)
        raise ValueError(f"the map has no '{symbol}'")

    def is_valid_cell(self, cell):
        """An action is invalid if it would leave the map or enter a '#'."""
        r, c = cell
        return 0 <= r < len(self.grid) and 0 <= c < len(self.grid[r]) and self.grid[r][c] != "#"

    def transition(self, state, action):
        dr, dc = ACTIONS[action]
        return (state[0] + dr, state[1] + dc)

    def successors(self, state):
        """Yield (action, next_state, step_cost) for every valid action."""
        for action in ACTIONS:
            nxt = self.transition(state, action)
            if self.is_valid_cell(nxt):
                yield action, nxt, 1

    def is_goal(self, state):
        return state == self.goal

    def free_cells(self):
        return [(r, c) for r in range(len(self.grid)) for c in range(len(self.grid[r]))
                if self.grid[r][c] != "#"]

    def render(self, path=()):
        out = [row[:] for row in self.grid]
        for r, c in path:
            if out[r][c] == ".":
                out[r][c] = "*"
        return "\n".join("".join(row) for row in out)


# ---------------------------------------------------------------------------
# Heuristics: h(n) estimates the cost from n to the goal
# ---------------------------------------------------------------------------
def manhattan(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def zero(state, goal):
    return 0


def euclidean(state, goal):
    return math.hypot(state[0] - goal[0], state[1] - goal[1])


def scaled(k, base=manhattan):
    """k times another heuristic (used for the 'multiplied by 2' experiment)."""
    return lambda state, goal: k * base(state, goal)


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------
class Result:
    def __init__(self, found, path, actions, expanded, distinct):
        self.found = found
        self.path = path              # list of cells from s0 to the goal
        self.actions = actions        # list of action names
        self.length = len(actions) if found else None
        self.expanded = expanded      # expansion operations
        self.distinct = distinct      # different states expanded


def reconstruct(parent, state):
    """Follow parent pointers from the goal back to s0."""
    path, actions = [state], []
    while parent[state] is not None:
        state, action = parent[state]
        actions.append(action)
        path.append(state)
    return path[::-1], actions[::-1]


def astar(problem, h=manhattan):
    """A* graph search. Frontier: a binary heap ordered by f = g + h.

    g[s] is the cheapest cost found so far from s0 to s. A state is pushed
    again only when a strictly cheaper route to it is found, and stale heap
    entries are skipped when popped. With an admissible and consistent h no
    state is ever expanded twice.
    """
    start, goal = problem.start, problem.goal
    g = {start: 0}
    parent = {start: None}
    counter = 0                                        # insertion order for ties
    h0 = h(start, goal)
    frontier = [(g[start] + h0, h0, counter, start, 0)]  # (f, h, tie, state, g at push)
    expanded, seen = 0, set()
    while frontier:
        f, hn, _, state, g_push = heapq.heappop(frontier)
        if g_push > g[state]:                          # stale entry: a cheaper path was found later
            continue
        if problem.is_goal(state):
            path, actions = reconstruct(parent, state)
            return Result(True, path, actions, expanded, len(seen))
        expanded += 1
        seen.add(state)
        for action, nxt, cost in problem.successors(state):
            new_g = g[state] + cost
            if nxt not in g or new_g < g[nxt]:         # new state, or a cheaper route to a known one
                g[nxt] = new_g
                parent[nxt] = (state, action)
                counter += 1
                h_n = h(nxt, goal)
                heapq.heappush(frontier, (new_g + h_n, h_n, counter, nxt, new_g))
    return Result(False, None, None, expanded, len(seen))


def bfs(problem):
    """Breadth-first search: FIFO frontier, no heuristic."""
    start = problem.start
    frontier = deque([start])
    parent = {start: None}                             # also the set of visited states
    expanded = 0
    while frontier:
        state = frontier.popleft()
        if problem.is_goal(state):
            path, actions = reconstruct(parent, state)
            return Result(True, path, actions, expanded, expanded)
        expanded += 1
        for action, nxt, _ in problem.successors(state):
            if nxt not in parent:
                parent[nxt] = (state, action)
                frontier.append(nxt)
    return Result(False, None, None, expanded, expanded)


# ---------------------------------------------------------------------------
# Independent checks (share no code with BFS / A*)
# ---------------------------------------------------------------------------
def exact_costs_to_goal(problem, target=None):
    """h*(n): true cost from every cell to the goal, by repeated relaxation
    (Bellman-Ford) with no queue and no search order. Moves are reversible
    with equal cost, so passing target=problem.start gives the cost from s0."""
    cells = problem.free_cells()
    inf = float("inf")
    dist = {c: inf for c in cells}
    dist[target or problem.goal] = 0
    changed = True
    while changed:
        changed = False
        for cell in cells:
            for _, nxt, cost in problem.successors(cell):
                if dist[nxt] + cost < dist[cell]:
                    dist[cell] = dist[nxt] + cost
                    changed = True
    return dist


def replay_is_valid(problem, result):
    """Replay the actions from s0: no invalid move, and end on the goal."""
    if not result.found:
        return False
    state = problem.start
    for action in result.actions:
        state = problem.transition(state, action)
        if not problem.is_valid_cell(state):
            return False
    return problem.is_goal(state)


def show(label, problem, result):
    print(f"{label}")
    if result.found:
        print(f"  solution found: yes | path length: {result.length} | states expanded: {result.expanded}")
        print(f"  path (row, col): " + " -> ".join(f"({r},{c})" for r, c in result.path))
        print(f"  actions: {' '.join(result.actions)}")
        print("  " + problem.render(result.path).replace("\n", "\n  "))
    else:
        print(f"  solution found: no | states expanded: {result.expanded}")


# ---------------------------------------------------------------------------
# Task 3: tests
# ---------------------------------------------------------------------------
def run_tests():
    print("\n" + "=" * 72 + "\nTASK 3: TESTING A* (Manhattan heuristic)\n" + "=" * 72)
    outcomes = []

    def check(name, cond):
        outcomes.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # Test 1: the original warehouse
    print("\nTest 1: original warehouse")
    lab = SearchProblem(WAREHOUSE)
    res = astar(lab)
    show("  A* on the lab map", lab, res)
    truth = exact_costs_to_goal(lab)[lab.start]
    print(f"  independent shortest distance S -> G (Bellman-Ford): {truth}")
    check("a path is found", res.found)
    check("replaying the path never hits a wall and ends at G", replay_is_valid(lab, res))
    check("path length equals the independent shortest distance", res.length == truth)
    check("path length >= Manhattan distance S-G", res.length >= manhattan(lab.start, lab.goal))

    # Test 2: trivial case
    print("\nTest 2: trivial case (goal adjacent to start)")
    trivial = SearchProblem("#####\n#SG##\n#####")
    res = astar(trivial)
    show("  A*", trivial, res)
    check("one-step solution: ['Right']", res.found and res.actions == ["Right"])
    check("exactly 1 move and 1 state expanded", res.length == 1 and res.expanded == 1)

    # Test 3: no solution
    print("\nTest 3: no solution (goal completely inaccessible)")
    blocked = SearchProblem("#######\n#S....#\n###.###\n#...#G#\n#######")
    res = astar(blocked)
    show("  A*", blocked, res)
    check("reports failure (and terminates)", not res.found)
    res_bfs = bfs(blocked)
    check("BFS also reports failure", not res_bfs.found)
    boxed = SearchProblem("#####\n#S#G#\n#####")
    check("start walled in: reports failure", not astar(boxed).found)

    # Test 4: alternative paths
    print("\nTest 4: alternative paths (top route 8 moves, bottom route 12 moves)")
    two_routes = SearchProblem("###########\n#S.......G#\n#.#######.#\n#.........#\n###########")
    res = astar(two_routes)
    show("  A*", two_routes, res)
    truth = exact_costs_to_goal(two_routes)[two_routes.start]
    check("returned path is a shortest path (8 moves)", res.found and res.length == truth == 8)

    print("\nTest 4b: two different shortest paths (both 10 moves)")
    tie = SearchProblem("#########\n#.......#\n#.#####.#\n#S#...#G#\n#.#.#.#.#\n#.......#\n#########")
    res = astar(tie)
    show("  A*", tie, res)
    truth = exact_costs_to_goal(tie)[tie.start]
    check("returns one of the shortest paths (10 moves)", res.found and res.length == truth == 10)

    print("\nExtra checks")
    edge = SearchProblem("#####\n S.G#\n#####")
    res = astar(edge)
    check("gap in the outer wall: never leaves the grid", res.found and res.actions == ["Right", "Right"])
    check("A* and BFS agree on the lab map", astar(lab).length == bfs(lab).length)
    mism = [p for p in (tie, two_routes, lab) if astar(p).length != bfs(p).length]
    check("A* and BFS agree on all test maps", not mism)

    print(f"\n{sum(outcomes)}/{len(outcomes)} checks passed")
    return all(outcomes)


# ---------------------------------------------------------------------------
# Task 5: BFS vs A*
# ---------------------------------------------------------------------------
def compare_bfs_astar():
    print("\n" + "=" * 72 + "\nTASK 5: BFS vs A* (Manhattan) on the lab map\n" + "=" * 72)
    lab = SearchProblem(WAREHOUSE)
    b, a = bfs(lab), astar(lab)
    print(f"{'Measure':<18}{'BFS':>10}{'A*':>10}")
    print(f"{'Solution found':<18}{str(b.found):>10}{str(a.found):>10}")
    print(f"{'Path length':<18}{b.length:>10}{a.length:>10}")
    print(f"{'States expanded':<18}{b.expanded:>10}{a.expanded:>10}")
    print(f"free cells in the warehouse: {len(lab.free_cells())}")
    print(f"same path? {b.path == a.path}")

    # Why do both expand every cell? Look at f = g*(n) + h(n), with g* the
    # true cost from s0 (computed independently of the search code).
    g_star = exact_costs_to_goal(lab, target=lab.start)
    h_star = exact_costs_to_goal(lab)
    opt = g_star[lab.goal]
    f_val = {c: g_star[c] + manhattan(c, lab.goal) for c in g_star}
    on_path = {c for c in g_star if g_star[c] + h_star[c] == opt}
    print(f"\nf = g* + h (Manhattan) per cell; the optimal cost is {opt}:")
    print(f"  cells on some shortest path: {len(on_path)}; cells off every shortest path: {len(g_star) - len(on_path)}")
    print(f"  cells with f < {opt}: {sum(v < opt for v in f_val.values())}")
    print(f"  cells with f = {opt}: {sum(v == opt for v in f_val.values())} "
          f"(all on a shortest path: {all(c in on_path for c in f_val if f_val[c] == opt)})")
    print(f"  off-path cells with f >= {opt}: {sum(1 for c in f_val if c not in on_path and f_val[c] >= opt)}")
    print(f"  Manhattan distance S-G = {manhattan(lab.start, lab.goal)}, true cost = {opt}")
    print("BFS path:")
    print("  " + lab.render(b.path).replace("\n", "\n  "))
    print("A* path:")
    print("  " + lab.render(a.path).replace("\n", "\n  "))

    # Supplementary: the same comparison on a mostly open room.
    room = SearchProblem(OPEN_ROOM)
    rb, ra = bfs(room), astar(room)
    print(f"\nSupplementary comparison on an open room ({len(room.free_cells())} free cells):")
    print(f"{'Measure':<18}{'BFS':>10}{'A*':>10}")
    print(f"{'Solution found':<18}{str(rb.found):>10}{str(ra.found):>10}")
    print(f"{'Path length':<18}{rb.length:>10}{ra.length:>10}")
    print(f"{'States expanded':<18}{rb.expanded:>10}{ra.expanded:>10}")
    print("A* path:")
    print("  " + room.render(ra.path).replace("\n", "\n  "))
    return b, a


# ---------------------------------------------------------------------------
# Task 6: heuristic investigation
# ---------------------------------------------------------------------------
def admissibility(problem, h):
    """Compare h with the exact cost h* on every cell.
    Returns (max over cells of h - h*, number of cells with h > h*)."""
    truth = exact_costs_to_goal(problem)
    worst, bad = -float("inf"), 0
    for cell, hstar in truth.items():
        diff = h(cell, problem.goal) - hstar
        worst = max(worst, diff)
        bad += diff > 1e-9
    return worst, bad


def consistency_violations(problem, h):
    """Number of edges (n, n') with h(n) > c(n, n') + h(n')."""
    bad = 0
    for cell in problem.free_cells():
        for _, nxt, cost in problem.successors(cell):
            if h(cell, problem.goal) > cost + h(nxt, problem.goal) + 1e-9:
                bad += 1
    return bad


def investigate_heuristics():
    print("\n" + "=" * 72 + "\nTASK 6: HEURISTIC INVESTIGATION (lab map)\n" + "=" * 72)
    lab = SearchProblem(WAREHOUSE)
    optimal = bfs(lab).length
    variants = [
        ("Manhattan", manhattan),
        ("h(n) = 0", zero),
        ("Euclidean", euclidean),
        ("2 x Manhattan", scaled(2)),
    ]
    print(f"optimal path length (BFS / independent check) = {optimal}\n")
    print(f"{'Heuristic':<16}{'found':>7}{'length':>8}{'expanded':>10}{'optimal?':>10}"
          f"{'max(h-h*)':>11}{'h>h* cells':>12}{'inconsistent edges':>20}")
    rows = {}
    for name, h in variants:
        res = astar(lab, h)
        worst, bad = admissibility(lab, h)
        incons = consistency_violations(lab, h)
        rows[name] = res
        print(f"{name:<16}{str(res.found):>7}{res.length:>8}{res.expanded:>10}"
              f"{str(res.length == optimal):>10}{worst:>11.2f}{bad:>12}{incons:>20}")
    print("\nPaths:")
    for name, res in rows.items():
        print(f"{name}: length {res.length}, expanded {res.expanded} "
              f"({res.distinct} distinct states)")
        print("  " + lab.render(res.path).replace("\n", "\n  "))

    # How far can the heuristic be scaled before the answer stops being optimal?
    print("\nScaling Manhattan by k (lab map):")
    print(f"{'k':>6}{'length':>8}{'expanded':>10}{'distinct':>10}{'optimal?':>10}")
    for k in (0, 0.5, 1, 1.5, 2, 3, 5, 10, 100):
        res = astar(lab, scaled(k))
        print(f"{k:>6}{res.length:>8}{res.expanded:>10}{res.distinct:>10}{str(res.length == optimal):>10}")

    # On the lab map the overestimating heuristic may still return an optimal
    # path. To test the claim "too aggressive => suboptimal" we also use a small
    # map (found by a random search for maps where 2 x Manhattan goes wrong).
    print("\nSecond map, where an overestimating heuristic goes wrong:")
    small = SearchProblem(SMALL_MAP)
    print("  " + small.render().replace("\n", "\n  "))
    opt = bfs(small).length
    truth = exact_costs_to_goal(small)[small.start]
    print(f"optimal length (BFS) = {opt}; independent check = {truth}")
    print(f"{'Heuristic':<16}{'found':>7}{'length':>8}{'expanded':>10}{'optimal?':>10}"
          f"{'max(h-h*)':>11}")
    small_rows = {}
    for name, h in variants + [("10 x Manhattan", scaled(10))]:
        res = astar(small, h)
        worst, _ = admissibility(small, h)
        small_rows[name] = res
        print(f"{name:<16}{str(res.found):>7}{res.length:>8}{res.expanded:>10}"
              f"{str(res.length == opt):>10}{worst:>11.2f}")
    print("Path returned by A* with 2 x Manhattan:")
    print("  " + small.render(small_rows["2 x Manhattan"].path).replace("\n", "\n  "))
    print("Path returned by A* with Manhattan:")
    print("  " + small.render(small_rows["Manhattan"].path).replace("\n", "\n  "))
    return rows


def main():
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    with redirect_stdout(Tee()):
        print(f"Python {sys.version.split()[0]}")
        print("=" * 72 + "\nWAREHOUSE FROM THE HANDOUT\n" + "=" * 72)
        print(WAREHOUSE)
        lab = SearchProblem(WAREHOUSE)
        print(f"\ns0 = {lab.start}, goal = {lab.goal}, free cells (states) = {len(lab.free_cells())}")
        ok = run_tests()
        compare_bfs_astar()
        investigate_heuristics()

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write(buf.getvalue())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
