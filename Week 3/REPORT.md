# Lab 3: Search and A*

Course: CS F407 Artificial Intelligence

**Files**

- `astar.py` is the only code file: the search problem, A*, BFS, the tests, the BFS/A* comparison and the heuristic investigation.
- `results.txt` is the full output of `python3 astar.py`. Every number in this report is copied from it.

**Setup:** Python 3.11, standard library only (no AI or machine-learning library, as the handout requires). It runs in well under a second and exits with code 0 only if every test passes.

```bash
python3 astar.py
```

**The warehouse** (rows and columns counted from 0):

```
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
```

**Counting convention used for "states expanded"** (the same for BFS and A*): a state is expanded when it is removed from the frontier and its successors are generated. The goal is recognised when it is removed from the frontier and is not counted as expanded. Ties between equal f-values are broken by smaller h, then by insertion order; successors are generated in the order Up, Down, Left, Right. Other conventions give counts that differ by a small constant, so compare counts only within this report.

---

## 1. Formulation of the search problem (Task 0)

| Component | Specification |
|---|---|
| State S | The robot's cell `(row, col)` where the map character is not `#`. The lab map has 64 free cells, so 64 states. |
| Actions A | {Up, Down, Left, Right} |
| Transition T | T((r, c), Up) = (r − 1, c), Down = (r + 1, c), Left = (r, c − 1), Right = (r, c + 1), defined only when the target cell is on the map and is not `#`. |
| Initial state s0 | (1, 1), the cell marked S |
| Goal G | {(7, 15)}, the cell marked G |
| Cost c | c(s, a, s′) = 1 for every move |

**(a) What information is necessary to specify a state?** Only the robot's row and column. The map is fixed and fully known, so it is part of the problem definition, not of the state.

**(b) What makes an action invalid?** The cell it leads to is an obstacle `#`, or lies outside the map.

**(c) Is this a deterministic search problem?** Yes. Each action from a given state has exactly one outcome, the map is static, and the robot always knows where it is.

**(d) What would constitute a solution?** A sequence of valid actions that takes the robot from s0 to G. An *optimal* solution is one with the fewest moves, since every move costs 1.

---

## 2. Design of the agent (Task 1)

This design was written before the LLM was prompted (Task 2).

1. **State in Python:** a tuple `(row, col)`. Tuples are immutable and hashable, so they can be dictionary keys and set members.
2. **Warehouse:** a list of lists of characters, built from the ASCII map. Cell `(r, c)` is `grid[r][c]`. The start and goal are found by scanning for `S` and `G`.
3. **Valid actions:** for each of the four actions, compute the target cell and accept it only if it is inside the grid and is not `#`.
4. **Goal recognition:** `state == goal`, tested when a state is removed from the frontier (for A*, this is what guarantees that the first goal found is optimal).
5. **Frontier contents:** for A*, a binary heap of entries `(f, h, tie-breaker, state, g)`, where f = g + h orders the heap; for BFS, a FIFO queue of states.
6. **Path reconstruction:** a dictionary `parent[state] = (previous state, action)`. From the goal, follow the parent pointers back to s0 and reverse the list.

**What the program reports:** whether a solution was found, the path (cells and actions), the path length, the number of states expanded, and the path drawn on the map with `*`.

---

## 3. Prompts used with the LLM (Task 2)

**LLM used:** Claude (Anthropic).

**Prompt 1 (A\*)**, based on the design above and the handout's example prompt:

> I am implementing a simple goal-based search agent in Python.
> The environment is a grid represented by an ASCII map. The agent starts at S and must reach G. The symbols # represent obstacles and . represents free cells. The agent can move up, down, left, or right, and every movement has cost 1.
> Implement A* search.
> Use Manhattan distance as the heuristic: h(n) = |x − xG| + |y − yG|.
> The program should:
> - represent grid positions as states;
> - maintain an appropriate frontier;
> - calculate g(n), h(n) and f(n);
> - avoid repeatedly expanding the same state;
> - reconstruct the path when the goal is reached;
> - report the path and its length;
> - report the number of states expanded.
>
> Keep the implementation simple and explain the main components of the code.
> (The warehouse map from the handout was pasted in here.) Structure the search as a function that takes a heuristic as a parameter, so that other heuristics can be tried later.

**Prompt 2 (BFS and experiments)**:

> Add a BFS version of the same agent on the same problem class, counting expanded states in exactly the same way. Do not change the warehouse. Then add tests for: the lab map, a one-step map, a map where the goal is unreachable, and maps with two routes. Use an independent method to compute true shortest distances, not BFS.

**Prompt 3 (Task 6, explanation of the heuristic):**

> Explain why Manhattan distance is an appropriate heuristic for a warehouse robot that can only move horizontally and vertically, one cell at a time.

The answer, checked against the experiments, is in section 7 below.

---

## 4. Results of the tests (Task 3)

All 14 checks pass (`14/14 checks passed` in `results.txt`).

**Test 1: original warehouse**

| Measure | Result |
|---|---|
| Path found | yes |
| Path length | 40 |
| States expanded | 63 |

Path (row, col):
`(1,1) → (1,2) → (1,3) → (1,4) → (1,5) → (2,5) → (3,5) → (4,5) → (5,5) → (5,6) → … → (5,13) → (4,13) → (3,13) → (3,12) → … → (3,7) → (2,7) → (1,7) → (1,8) → … → (1,15) → (2,15) → … → (7,15)`

```
#################
#S****#*********#
#.###*#*#######*#
#...#*#*******#*#
###.#*#######*#*#
#...#*********#*#
#.###########.#*#
#.............#G#
#################
```

Validation: replaying the actions never hits a wall and ends at G. The length equals the shortest distance from an independent relaxation-based (Bellman-Ford) computation that shares no code with A* or BFS: 40. It is at least the Manhattan distance between S and G (20).

**Test 2: trivial case** (`#SG##`): the program finds `['Right']`, path length 1, 1 state expanded.

**Test 3: no solution** (the handout's map): the program reports failure after expanding 9 states (all the cells reachable from S) and stops, so it does not loop. BFS also reports failure. I added a second case with S boxed in by walls, which also reports failure.

**Test 4: alternative paths:** on a map with a top route of 8 moves and a bottom route of 12, the path returned has 8 moves, which equals the independent shortest distance. **Test 4b:** on a map with two different shortest routes (both 10 moves), A* returns one of them, of length 10 (11 states expanded).

**Extra checks:** a map with a gap in the outer wall (the robot never leaves the grid), and A* and BFS agree on the path length on all test maps.

---

## 5. Inspecting the A* code (Task 4)

Line numbers refer to `astar.py`.

| Concept | Where it appears in the code |
|---|---|
| State | A tuple `(row, col)`, created in `SearchProblem._find` and in `transition` (line 103). |
| Action | The `ACTIONS` dictionary (name → row/column offset); `successors` (line 107) loops over it. |
| Transition | `SearchProblem.transition(state, action)` (line 103); validity of the result is checked by `is_valid_cell` (line 98). |
| Goal test | `SearchProblem.is_goal` (line 114), called in `astar` at line 191 when a state is popped from the frontier. |
| g(n) | The dictionary `g` (line 181); updated at line 199 using `new_g = g[state] + cost` (line 197). |
| h(n) | The function passed in as `h` (`manhattan` at line 132); called at line 184 for the start and at line 202 for each successor. |
| f(n) | Computed explicitly as `new_g + h_n` when a state is pushed (line 203) and as `g[start] + h0` for the start (line 185). It is the first element of each heap entry. |
| Frontier | The list `frontier`, used as a binary heap through `heapq` (lines 185, 188, 203). |
| Visited states | The dictionary `g` also serves as the record of reached states (`nxt not in g`, line 198); the set `seen` (line 195) only counts distinct expanded states for the report. |
| Path reconstruction | `reconstruct(parent, state)` (line 162), following the `parent` dictionary set at line 200. |

**(a) What data structure is used for the A\* frontier?** A priority queue implemented as a binary min-heap (Python's `heapq`) holding tuples `(f, h, tie-breaker, state, g)`.

**(b) How does the program select the next state to expand?** It pops the heap's smallest entry, which is the state with the lowest f = g + h (ties: lower h, then earlier insertion). Entries that are out of date (a cheaper route to that state was found after it was pushed) are skipped.

**(c) Where is the heuristic calculated?** In the function `manhattan` (line 132), called when a state is generated (line 202), not when it is expanded. Each generated state therefore has its h computed once per push.

**(d) Does the program explicitly calculate f(n) = g(n) + h(n)?** Yes: `new_g + h_n` at line 203. The value is stored in the heap entry and used for ordering.

**(e) How does the program prevent unnecessary repeated exploration?** A successor is pushed only if it has never been reached or if a *strictly cheaper* route to it has been found (line 198). A state that is popped with an out-of-date g is skipped (line 189). With an admissible, consistent heuristic such as Manhattan, this means no state is ever expanded twice.

---

## 6. BFS versus A\* (Task 5)

Both run on the same, unchanged lab map.

| Measure | BFS | A* (Manhattan) |
|---|---:|---:|
| Solution found | yes | yes |
| Path length | 40 | 40 |
| States expanded | 63 | 63 |

**(a) Did both algorithms find a solution?** Yes.

**(b) Did they find paths of the same length?** Yes, both 40, and they returned exactly the same path. That is expected, because A* with an admissible heuristic is optimal and BFS is optimal for unit costs.

**(c) Which algorithm expanded fewer states?** Neither: both expanded 63 of the 64 free cells (every cell except the goal). This is a real property of this warehouse, not a bug.

**(d) Why might A\* expand fewer states?** A* can skip states whose f = g + h is larger than the optimal cost, so it avoids regions that look unpromising. On this map that advantage disappears, for a specific reason: the warehouse is a winding corridor, and the Manhattan distance from S to G is only 20 while the true cost is 40. So f starts at 20 and rises to 40 along the path. Every one of the 23 cells that are *not* on the shortest path (the dead-end branches) has f < 40, so A* has to expand all of them before it is allowed to accept the goal with f = 40. The script checks this directly (see `results.txt`): 49 cells have f < 40, the 15 cells with f = 40 are all on a shortest path, and no off-path cell has f ≥ 40. The heuristic is admissible but not informative here, because the walls make the true distance much larger than the straight-line distance.

To show the case where A* does help, I also ran both algorithms on a mostly open 22 × 9 room (134 free cells, goal in the far corner, with a short wall in the middle; the map is `OPEN_ROOM` in `astar.py`). This is an additional experiment and does not replace the required one.

| Measure (open room) | BFS | A* (Manhattan) |
|---|---:|---:|
| Path length | 24 | 24 |
| States expanded | 131 | 24 |

Both found an optimal path of 24 moves, but A* expanded 24 states against BFS's 131, because in open space the Manhattan estimate is nearly exact and steers the search straight towards the goal.

> **Think about it: what information does the algorithm use to decide where to search next?**
> BFS uses only the *cost so far* (depth): it expands the nearest states first, in every direction. A* also uses an *estimate of the cost to go*, h(n), so it can prefer states that look closer to the goal. The benefit depends entirely on how good that estimate is. When h is close to the true remaining cost (the open room), A* expands a fraction of the states; when h is far below it (the winding warehouse), A* behaves like BFS.

---

## 7. Heuristic investigation (Task 6)

For each version I recorded whether a solution was found, the path length and the number of states expanded. I also measured the heuristic directly against the exact cost h*(n) of every cell, computed independently: `max(h − h*)` (a value above 0 means h overestimates somewhere, so it is not admissible) and the number of cells where h > h*.

**Lab map** (optimal length 40):

| Heuristic | Found | Path length | States expanded | Optimal? | max(h − h\*) | Cells with h > h\* |
|---|:-:|---:|---:|:-:|---:|---:|
| Manhattan | yes | 40 | 63 | yes | 0 | 0 |
| h(n) = 0 | yes | 40 | 63 | yes | 0 | 0 |
| Euclidean | yes | 40 | 63 | yes | 0 | 0 |
| 2 × Manhattan | yes | 40 | 67 | yes | 14 | 18 |

**1. h(n) = 0.** A* still finds the optimal path, because it becomes uniform-cost search (f = g), which for unit costs behaves like BFS. It expanded 63 states, the same as BFS, since it has no information to steer with. So admissibility holds trivially, but the heuristic gives no speed-up.

**2. Euclidean distance.** It is admissible (max(h − h\*) = 0), because a straight line is never longer than the shortest grid path. It found the optimal path and expanded 63 states. It is always ≤ Manhattan distance, so it is a weaker (less informed) estimate; on this map, where even Manhattan gives no saving, there is nothing to lose.

**3. Heuristic multiplied by 2.** It is **not** admissible: it overestimates at 18 cells, by up to 14. A* still returned the optimal 40-move path on the lab map, but made 67 expansions for 63 distinct states, so there were 4 extra (repeated) expansions. The heuristic is also inconsistent (64 edges where h drops by more than the step cost), so a state can be reached first by a worse route and must be re-expanded after a better route is found. The optimal answer here is partly luck: the warehouse is nearly a single corridor, so there is hardly any other path to wrongly prefer.

**How far can the heuristic be scaled?** Multiplying Manhattan by k on the lab map:

| k | Path length | States expanded | Distinct states | Optimal? |
|---:|---:|---:|---:|:-:|
| 0 | 40 | 63 | 63 | yes |
| 0.5 | 40 | 63 | 63 | yes |
| 1 | 40 | 63 | 63 | yes |
| 1.5 | 40 | 65 | 63 | yes |
| 2 | 40 | 67 | 63 | yes |
| 3 | 40 | 77 | 63 | yes |
| 5 | 40 | 83 | 63 | yes |
| 10 | **48** | 57 | 57 | **no** |
| 100 | **48** | 57 | 57 | **no** |

For k ≤ 5 the answer stays optimal on this map, and the work increases slightly because of re-expansions. By k = 10 the search becomes greedy: it expands fewer states (57) but returns a 48-move path, 8 moves longer than the best.

**A second map, where overestimating goes wrong already at 2 ×.** The lab map hides the effect, so I also used a small map (found by a random search for maps where 2 × Manhattan gives a wrong answer; it is `SMALL_MAP` in `astar.py`):

```
#########
#S.#.#..#
#....##.#
#.#.....#
#....#..#
##..#..G#
#########
```

| Heuristic | Path length | States expanded | Optimal? | max(h − h\*) |
|---|---:|---:|:-:|---:|
| Manhattan | 10 | 17 | yes | 0 |
| h(n) = 0 | 10 | 24 | yes | 0 |
| Euclidean | 10 | 17 | yes | 0 |
| 2 × Manhattan | **12** | 14 | **no** | 10 |
| 10 × Manhattan | **12** | 14 | **no** | 90 |

Here the admissible heuristics return the optimal 10 moves, with Manhattan and Euclidean expanding 17 states and h = 0 expanding 24. The overestimating ones return a 12-move path, having expanded fewer states (14).

**Why Manhattan is appropriate (Prompt 3, checked against the experiments).** With four-neighbour moves of cost 1, each move changes |Δrow| + |Δcol| by exactly 1. So no path can be shorter than the Manhattan distance, which makes it admissible. It also satisfies h(n) ≤ 1 + h(n′) for every step, which makes it consistent (the experiment found 0 inconsistent edges). Walls can only make the true path longer than this estimate, never shorter. Euclidean distance is also admissible but is a weaker estimate, since it ignores that the robot cannot move diagonally.

> **Think about it: what happens when the heuristic is too optimistic or too aggressive?**
> "Too optimistic" (far *below* the true cost, such as h = 0) keeps A* optimal but removes its advantage: it degenerates into blind search, expanding as many states as BFS (63 here; 24 versus 17 on the small map). "Too aggressive" (*above* the true cost, such as 2 × Manhattan) breaks admissibility: A* then trusts the estimate and can commit to a route that merely looks close to the goal. Experiments showed this two ways: an answer 2 moves too long on the small map, and 8 moves too long on the lab map at k = 10. It also lets the search expand states repeatedly when the heuristic is inconsistent. In both cases the search looked at fewer states (14 vs 17, 57 vs 63). The price of speed is the loss of the optimality guarantee.

---

## 8. Evaluation of the LLM-generated agent (Task 7)

This section describes what actually happened in this run.

**Provenance**

| | Who / what |
|---|---|
| **Designed before the code** | The search-problem formulation (section 1) and the agent design (section 2), written from the handout's Task 0 and Task 1 before the program. If your course requires these to be entirely your own words, rewrite them in your own words. The heuristic investigation follows the handout's three experiments. |
| **Suggested by the LLM** | The code structure (`SearchProblem`, `astar`, `bfs`), the use of `heapq` with a stale-entry check, the tie-breaking rule, the test maps, and the independent Bellman-Ford checker. |
| **Accepted** | The A* and BFS implementations as written; they passed every test on the first run. |
| **Changed** | (i) After the first run showed A* and BFS expanding the *same* 63 states on the lab map, I added the open-room comparison to show when the heuristic helps. (ii) For Task 6 I replaced a hand-drawn trap map (a draft that needed awkward string patching to be a valid rectangle) with a small map found by a random search for maps where an overestimating heuristic goes wrong. |
| **Tested** | Section 4 (14 checks) plus the admissibility and consistency measurements against the exact h\*. |

**1. What parts of the generated code were correct immediately?**
All of it on this run: A*, BFS, path reconstruction, the no-solution handling and the tests passed on the first execution (14/14 checks). Passing on the first run is a fact about this run, not something to rely on in general.

**2. Did you find any bugs or design problems?**
No bugs in the search code. The design problem was in the *experiment*: on the lab map BFS and A* tied at 63 expanded states, which would have made Task 5 uninformative. The Task 6 trap-map draft mentioned above was also a problem and was replaced.

**3. How did you discover those problems?**
By reading the results, not the code: the equal counts in the comparison table were the prompt to ask why, and the f-value analysis (49 cells with f < 40, the 15 with f = 40 all on the optimal path) gave the explanation. The awkward trap map was noticed when reading the draft before running it.

**4. Did the LLM use terminology or data structures that you did not understand?**
The terms worth understanding before submitting are: *lazy deletion* (leaving out-of-date heap entries in place and skipping them when popped, since `heapq` cannot update an entry), *consistency* (h(n) ≤ c(n, n′) + h(n′), which guarantees that no state is expanded twice), and *re-expansion* (a state expanded again after a cheaper route is found). Replace this paragraph with your own answer if other terms were new to you.

**5. Did you modify the LLM-generated code?**
Yes, as described under "Changed" above: one supplementary experiment and one replacement of the trap map. The algorithms themselves were not changed.

**6. Which tests were most useful?**
The comparison of the returned path length against an independent shortest-distance computation (Test 1 and Test 4): it checks the answer itself, not just that a plausible path came out. Next most useful was the measured admissibility check (h against the exact h\*), which turned "is this heuristic admissible?" from an argument into a measurement. The no-solution test is the one that catches infinite loops.

**7. Could you have trusted the program without testing it?**
No. The output on the lab map (a 40-move path drawn on the map) looks plausible, but plausibility does not show optimality: the 2 × Manhattan run returns a path that also looks perfectly reasonable, and it is wrong by 2 moves on the small map. Only the comparison against an independent answer reveals it.

**8. What did you understand about A\* that you did not understand before implementing it?**
That A* is not automatically faster than BFS: on the lab map they expanded exactly the same states, because the straight-line estimate (20) is half the true cost (40). The benefit comes from the quality of h, not from the algorithm alone. Also, that "admissible" is what makes the answer optimal, while "consistent" is what stops states being expanded twice, and that these are separate properties.

---

## 9. Final reflection

**1. Why is it important to formulate the search problem before writing the search algorithm?**
The formulation fixes what the algorithm must do: what a state is, which actions are legal, what a solution is and what a step costs. Without it there is nothing to check the code against. Writing it first also separates two kinds of mistake: a wrong *problem* (for example, forgetting that leaving the grid is an invalid action) and a wrong *implementation*. Here, the formulation in section 1 became the checklist for the code (`transition`, `is_valid_cell`, `is_goal`, and cost 1) and for the tests.

**2. In what sense is A\* an "informed" search algorithm?**
BFS chooses what to expand using only information about the past (the cost g so far). A* also uses h(n), an estimate of the cost *still to go* that comes from knowledge about the problem (here, that the robot moves on a grid). This lets it rank frontier states by the estimated cost of a complete solution through them, f = g + h, instead of only by how close they are to the start. It is informed in proportion to how good h is, as the open-room experiment (24 expansions against 131) and the warehouse (63 against 63) show.

**3. Why does the choice of heuristic matter?**
It controls both *correctness* and *efficiency*. If h never exceeds the true cost (admissible), A* returns an optimal path; if it does, the guarantee is lost, as shown by the 12-move answer against the optimal 10 with 2 × Manhattan, and the 48-move against 40 at k = 10 on the lab map. Among admissible heuristics, larger is better: h = 0 (63 and 24 expansions) and Euclidean were no more informed than Manhattan here, and on the small map h = 0 expanded 24 states against 17 for Manhattan. A good heuristic therefore has to be as large as possible without exceeding the true cost.

**4. What did the LLM contribute to the engineering process?**
It turned the design into working code quickly: the heap-based frontier with stale-entry handling, BFS for the comparison, the tests with an independent shortest-distance checker, and the experiment scaffolding that printed every table in this report. It did not choose the problem formulation, decide what counts as evidence, or interpret why A* and BFS tied; those came from the design and from reading the results (and, in a real submission, from you).

**5. What could go wrong if an engineer simply accepted LLM-generated code without testing it?**
Plausible but wrong output could be shipped. A path that looks reasonable may be longer than the optimum (2 × Manhattan: 12 against 10 moves); a search that fails to handle an unreachable goal could loop forever; a missing bounds check could let the robot step off the map; and a miscounted "states expanded" would make the BFS/A* comparison meaningless. Here, the program's first run looked correct, and only the independent checks (shortest distance, h\*, replaying every path) show that it is. The engineer remains responsible for understanding, testing and validating it.
