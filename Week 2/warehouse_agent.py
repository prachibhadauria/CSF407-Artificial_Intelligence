"""
Lab 2 - Agents: a goal-based agent for warehouse navigation
CS F407 Artificial Intelligence

The agent keeps an explicit goal (the dispatch square G), a model of the
warehouse (the grid) and its current position. It picks actions by searching,
ahead of time, for a sequence of moves whose predicted result is the goal, and
then executes that plan one move at a time.

Search algorithm: breadth-first search (BFS). Every move costs the same, so the
first time BFS reaches G it has found a shortest collision-free path. BFS is
also complete on a finite grid, so it reports "no path" correctly when G is
unreachable.

Usage:
    python3 warehouse_agent.py        # solves the lab map, then runs the
                                      # tests and the scaling experiment
                                      # and writes results.txt

Only the Python 3 standard library is needed.
"""

import io
import os
import sys
import time
from collections import deque
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))

# The warehouse from the handout. S = start, G = goal, # = obstacle, . = free.
WAREHOUSE = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

# Action name -> (row change, column change). One move = one grid square.
ACTIONS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class Environment:
    """The warehouse as a two-dimensional grid, with the positions of S and G."""

    def __init__(self, text):
        self.grid = [list(row) for row in text.strip("\n").splitlines()]
        self.rows = len(self.grid)
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, symbol):
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == symbol:
                    return (r, c)
        raise ValueError(f"the map has no '{symbol}'")

    def is_free(self, pos):
        """True if pos is inside the grid and is not an obstacle."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < len(self.grid[r]) and self.grid[r][c] != "#"

    def result(self, pos, action):
        """Transition model: the square reached by taking `action` from `pos`
        (the result may be a wall or off the grid; is_free() decides)."""
        dr, dc = ACTIONS[action]
        return (pos[0] + dr, pos[1] + dc)

    def render(self, path=()):
        """The map as text, with the squares of `path` drawn as '*'."""
        out = [row[:] for row in self.grid]
        for r, c in path:
            if out[r][c] == ".":
                out[r][c] = "*"
        return "\n".join("".join(row) for row in out)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class GoalBasedAgent:
    """
    Goal-based agent.

    Information it maintains: the map (its model of how moves change the
    position), its current position, its goal, and the plan it is following.
    Decision making: if it has no plan, search for a sequence of actions that
    leads from the current position to the goal; then output the next action
    of that plan.
    """

    def __init__(self, env):
        self.env = env
        self.position = env.start
        self.goal = env.goal
        self.plan = None
        self.expanded = 0            # squares taken off the frontier by search

    def goal_test(self, pos):
        return pos == self.goal

    def search(self):
        """Breadth-first search from the current position.
        Returns a list of actions, or None if the goal cannot be reached."""
        start = self.position
        self.expanded = 0
        if self.goal_test(start):
            return []
        frontier = deque([start])        # FIFO queue: nearest squares first
        parent = {start: None}           # square -> (previous square, action); doubles as visited set
        while frontier:
            pos = frontier.popleft()
            self.expanded += 1
            for action in ACTIONS:
                nxt = self.env.result(pos, action)
                if not self.env.is_free(nxt) or nxt in parent:
                    continue
                parent[nxt] = (pos, action)
                if self.goal_test(nxt):
                    return self._actions_to(parent, nxt)
                frontier.append(nxt)
        return None

    @staticmethod
    def _actions_to(parent, node):
        """Follow parent pointers back to the start and return the actions in order."""
        actions = []
        while parent[node] is not None:
            node, action = parent[node]
            actions.append(action)
        return actions[::-1]

    def choose_action(self):
        """Agent function: (state, goal, model) -> next action, or None."""
        if self.plan is None:
            self.plan = self.search()
        if not self.plan:
            return None
        return self.plan.pop(0)

    def run(self):
        """Execute the plan step by step. Returns the list of visited squares,
        or None if there is no plan."""
        trajectory = [self.position]
        while not self.goal_test(self.position):
            action = self.choose_action()
            if action is None:
                return None
            nxt = self.env.result(self.position, action)
            assert self.env.is_free(nxt), f"collision at {nxt}"
            self.position = nxt
            trajectory.append(nxt)
        return trajectory


def solve(text, verbose=True):
    """Build the environment and agent, find and execute a plan.
    Returns (plan, squares_expanded); plan is None if there is no path."""
    env = Environment(text)
    agent = GoalBasedAgent(env)
    plan = agent.search()
    expanded = agent.expanded
    if plan is None:
        if verbose:
            print("No collision-free path exists from S to G.")
        return None, expanded
    agent.plan = list(plan)
    trajectory = agent.run()
    if verbose:
        print(f"Path found: {len(plan)} moves, {expanded} squares expanded")
        print("Actions:", " ".join(plan))
        print("Squares (row, col):", " -> ".join(f"({r},{c})" for r, c in trajectory))
        print(env.render(trajectory))
    return plan, expanded


# ---------------------------------------------------------------------------
# Independent checks used by the tests
# ---------------------------------------------------------------------------
def replay_is_valid(text, plan):
    """Replay a plan move by move: no move may hit an obstacle or leave the
    grid, and the last square must be G."""
    env = Environment(text)
    pos = env.start
    for action in plan:
        pos = env.result(pos, action)
        if not env.is_free(pos):
            return False
    return pos == env.goal


def shortest_distance(text):
    """Independent oracle that shares no code with BFS: repeatedly relax
    dist[p] = min(dist[p], dist[q] + 1) over all free neighbours q until
    nothing changes (Bellman-Ford on the grid). Returns dist(S, G) or None."""
    env = Environment(text)
    free = [(r, c) for r in range(env.rows) for c in range(len(env.grid[r]))
            if env.is_free((r, c))]
    inf = float("inf")
    dist = {p: inf for p in free}
    dist[env.start] = 0
    changed = True
    while changed:
        changed = False
        for p in free:
            for a in ACTIONS:
                q = env.result(p, a)
                if env.is_free(q) and dist[q] + 1 < dist[p]:
                    dist[p] = dist[q] + 1
                    changed = True
    return None if dist[env.goal] == inf else dist[env.goal]


def dfs_length(text):
    """Depth-first search (LIFO stack) on the same problem, to show why the
    search algorithm matters: it finds a path, but not necessarily a short one."""
    env = Environment(text)
    stack = [env.start]
    parent = {env.start: None}
    while stack:
        pos = stack.pop()
        if pos == env.goal:
            n = 0
            while parent[pos] is not None:
                pos = parent[pos]
                n += 1
            return n
        for action in ACTIONS:
            nxt = env.result(pos, action)
            if env.is_free(nxt) and nxt not in parent:
                parent[nxt] = pos
                stack.append(nxt)
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def run_tests():
    print("\n" + "=" * 70 + "\nTESTS\n" + "=" * 70)
    results = []

    def check(name, cond):
        results.append(bool(cond))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    plan, _ = solve(WAREHOUSE, verbose=False)
    check("lab map: a path is found", plan is not None)
    check("lab map: replaying the plan never hits an obstacle and ends at G",
          replay_is_valid(WAREHOUSE, plan))
    # S = (1,1), G = (1,19), so the Manhattan distance 18 is a lower bound.
    # Row 1 is blocked at column 6, so the vehicle must drop to row 2 and come
    # back up (2 extra moves): 18 + 2 = 20 is achievable, so it is optimal.
    check("lab map: length >= Manhattan lower bound 18", len(plan) >= 18)
    check("lab map: length == 20 (18 + one forced detour down and up)", len(plan) == 20)
    check("lab map: BFS length == independent shortest distance (Bellman-Ford)",
          len(plan) == shortest_distance(WAREHOUSE))

    plan, _ = solve("#####\n#S#G#\n#####", verbose=False)
    check("goal walled off: reports that no path exists", plan is None)

    plan, _ = solve("####\n#SG#\n####", verbose=False)
    check("goal next to start: one move, Right", plan == ["Right"])

    plan, _ = solve("#####\n#S..#\n#...#\n#..G#\n#####", verbose=False)
    check("open room: shortest path = Manhattan distance 4", plan is not None and len(plan) == 4)

    two_routes = "#######\n#S...G#\n#.###.#\n#.....#\n#######"
    plan, _ = solve(two_routes, verbose=False)
    check("two routes (lengths 4 and 8): takes the shorter one", plan is not None and len(plan) == 4)

    plan, _ = solve("#####\n S.G#\n#####", verbose=False)
    check("gap in the outer wall: never steps off the grid", plan == ["Right", "Right"])

    dfs = dfs_length(WAREHOUSE)
    print(f"[INFO] depth-first search on the lab map finds a path of {dfs} moves "
          f"(BFS: 20), so it is complete but not optimal")
    check("DFS path is valid but no shorter than the BFS path", dfs is not None and dfs >= 20)

    print(f"\n{sum(results)}/{len(results)} tests passed")
    return all(results)


# ---------------------------------------------------------------------------
# "Think About It": what happens when the warehouse gets bigger?
# ---------------------------------------------------------------------------
def scaling_experiment():
    print("\n" + "=" * 70)
    print("SCALING: open n x n warehouse, S top-left, G bottom-right")
    print("=" * 70)
    print(f"{'n':>5} {'free squares':>13} {'expanded':>10} {'path length':>12} {'time (ms)':>10}")
    for n in (10, 20, 40, 80, 160):
        rows = ["#" * (n + 2)]
        for r in range(n):
            row = ["."] * n
            if r == 0:
                row[0] = "S"
            if r == n - 1:
                row[-1] = "G"
            rows.append("#" + "".join(row) + "#")
        rows.append("#" * (n + 2))
        text = "\n".join(rows)
        t0 = time.perf_counter()
        plan, expanded = solve(text, verbose=False)
        ms = (time.perf_counter() - t0) * 1000
        print(f"{n:>5} {n * n:>13} {expanded:>10} {len(plan):>12} {ms:>10.1f}")


def main():
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    with redirect_stdout(Tee()):
        print(f"Python {sys.version.split()[0]}")
        print("=" * 70 + "\nWAREHOUSE FROM THE HANDOUT\n" + "=" * 70)
        print(WAREHOUSE)
        print("\n" + "=" * 70 + "\nAGENT RUN\n" + "=" * 70)
        solve(WAREHOUSE)
        ok = run_tests()
        scaling_experiment()

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write(buf.getvalue())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
