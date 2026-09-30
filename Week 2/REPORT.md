# Lab 2: Agents (Constructing a Goal-Based Agent using an LLM)

Course: CS F407 Artificial Intelligence

**Files**

- `warehouse_agent.py` is the only code file: the environment, the goal-based agent, the tests and the scaling experiment.
- `results.txt` is the full output of `python3 warehouse_agent.py`. Every number in this report is copied from it.

**Setup:** Python 3.11, standard library only. The script takes under a second and exits with code 0 only if all tests pass.

```bash
python3 warehouse_agent.py
```

**The problem.** The map from the handout, with rows and columns counted from 0. S = (1, 1) and G = (1, 19).

```
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

---

## Task 1: Understanding the problem

**1. What is the environment?**
The warehouse floor: a 7 × 21 grid of free squares (`.`) and shelving units (`#`), plus the vehicle. It is fully observable (the whole map is known), deterministic (a move always lands where expected), static (the shelves do not move), discrete and single-agent.

**2. What is the goal of the agent?**
To reach the dispatch square G = (1, 19) from the start S = (1, 1) without entering a `#` square. Among all such paths we want a shortest one.

**3. What actions are available?**
Up, Down, Left and Right. Each moves the vehicle by one grid square. An action is only legal if the square it leads to is inside the grid and is not an obstacle.

**4. What information must the agent maintain to choose its next action?**
- its current position;
- its goal position;
- a model of the warehouse (which squares are blocked) and of how each action changes the position, so that it can predict the result of an action before taking it;
- while searching: the frontier of squares still to explore and the squares already reached, with a pointer from each square to the one it was reached from, so the path can be rebuilt;
- while acting: the plan it is following.

**5. Why is this a goal-based agent and not a simple reflex agent?**
A simple reflex agent maps the current percept straight to an action ("if the square to the right is free, go right"). That fails here. From S, going right along row 1 leads into a dead end at column 5, because (1, 6) is a shelf. Reaching G needs a detour down to row 2 and back up. The agent can only find this by looking ahead: it uses its model to ask "which sequence of actions ends at G?" and chooses actions by their predicted consequences for an explicit goal. That is what makes it goal-based.

> **Think about it: what if the warehouse becomes twice as large?**
> BFS would still be *correct*, because every move still costs 1, so it stays complete and optimal. It stays *feasible* but gets more expensive. Since BFS has no sense of direction, it explores almost the whole warehouse. I measured this on open n × n warehouses:
>
> | n | free squares | squares expanded | path length | time (ms) |
> |---:|---:|---:|---:|---:|
> | 10 | 100 | 98 | 18 | 0.2 |
> | 20 | 400 | 398 | 38 | 1.9 |
> | 40 | 1600 | 1598 | 78 | 3.5 |
> | 80 | 6400 | 6398 | 158 | 17.1 |
> | 160 | 25600 | 25598 | 318 | 68.5 |
>
> Doubling the side length quadruples the area, and the squares expanded (and roughly the time) quadruple with it. Memory for the frontier and the visited set also grows with the area.
>
> Difficulties that would appear in a bigger or more realistic warehouse:
> - **Cost.** Search time and memory grow with area, so an informed search such as A* with a Manhattan-distance heuristic would be worth using to expand far fewer squares.
> - **Changing environment.** If shelves or other vehicles move, a plan made earlier can become invalid and the agent must replan, which is expensive if each search is expensive.
> - **Several vehicles.** They would have to be coordinated so they do not collide.
> - **Unequal move costs.** If turns or congested aisles cost more, plain BFS is no longer optimal, and we need uniform-cost search or A*.
> - **Partial observability.** In a huge real warehouse the vehicle would not know the whole map in advance.

---

## Task 2: Designing the agent

| Component | Design |
|---|---|
| **Environment** | The `Environment` class: the grid as a list of rows, the positions of S and G, `is_free(pos)` (inside the grid and not `#`) and `result(pos, action)`, the transition model. |
| **Current state** | The vehicle's position `(row, col)`. The map is fixed and known, so it is part of the problem, not of the state. |
| **Goal** | `goal_test(pos)`: `pos == G`. |
| **Available actions** | `{Up, Down, Left, Right}`, each a `(row change, column change)` offset. |
| **Decision-making component** | `GoalBasedAgent.search()`: breadth-first search over positions returns a list of actions that reaches the goal. `choose_action()` returns the next action of that plan on each step. |

**Block diagram** (goal-based agent architecture)

```
                +---------------------------------------------------+
                |                       AGENT                       |
                |                                                   |
  percept       |   +----------------+       +-------------------+  |
  (current -----+-->|  State:        |------>|  Model:           |  |
  position)     |   |  where am I?   |       |  "what does action|  |
                |   +----------------+       |  a do?" (map +    |  |
                |                            |  result())        |  |
                |   +----------------+       +---------+---------+  |
                |   |  Goal:         |                 |            |
                |   |  be at G       |                 v            |
                |   +-------+--------+       +-------------------+  |
                |           |                |  Decision making: |  |
                |           +--------------->|  BFS search for   |  |
                |                            |  an action        |  |
                |                            |  sequence that    |  |
                |                            |  reaches the goal |  |
                |                            +---------+---------+  |
                |                                      | plan       |
                |                            +---------v---------+  |
                |                            | next action of    |  |
                |                            | the plan          |  |
                +----------------------------+---------+---------+--+
                                                       |
                                                       v  action: Up / Down / Left / Right
                                 +-------------------------------------------+
                                 |  ENVIRONMENT: the warehouse grid          |
                                 |  (new position = result(position, action))|
                                 +-------------------------------------------+
```

---

## Task 3: Prompt engineering

**LLM used:** Claude (Anthropic).

**Prompt used.** It follows the handout's suggested prompt, plus a precise specification of the actions, the agent structure and the tests:

> Write a well-documented Python program implementing a goal-based agent for the warehouse navigation problem shown below.
>
> ```
> #####################
> #S....#............G#
> #.##....##########..#
> #....##.............#
> #.######.###.#.###..#
> #........#..........#
> #####################
> ```
>
> S is the start, G is the goal, # is an obstacle, . is free space. The vehicle may move Up, Down, Left or Right, one grid square per move, and every move costs the same.
> The program should:
> - represent the warehouse as a two-dimensional grid;
> - be structured as an agent that stores its current position, its goal and a model of the environment, and has a function that chooses its next action;
> - determine a collision-free path from S to G, and find a shortest one;
> - avoid all obstacles and never leave the grid;
> - print either the path found (as a list of actions and drawn on the map) or a suitable message if no path exists;
> - explain the search algorithm that has been chosen and why it is appropriate.
>
> Also write tests: the given map, a map where G is walled off, a map where G is next to S, and a map with two routes of different lengths.

**Result on the lab map** (from `results.txt`):

```
Path found: 20 moves, 55 squares expanded
Actions: Right Right Right Down Right Right Right Up Right Right Right Right Right Right Right Right Right Right Right Right
#####################
#S***.#************G#
#.##****##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

The agent goes right to column 4, down to row 2, right past the shelf at (1, 6), back up to row 1 at column 7, and then straight to G.

**Is 20 moves really the shortest?** G is 18 columns to the right of S, so at least 18 horizontal moves are needed. Because (1, 6) is a shelf, the vehicle must leave row 1 at least once, which costs at least one Down and one Up. So 18 + 2 = 20 is a lower bound, and the path found has exactly that length. I also checked it against an independent shortest-distance computation that shares no code with BFS (see Tests).

**Tests.** `python3 warehouse_agent.py` runs them after the agent run. 11 of 11 pass.

| Test | Expected | Result |
|---|---|---|
| Lab map: a path is found | a path | PASS |
| Lab map: replaying the plan never hits an obstacle and ends at G | legal | PASS |
| Lab map: length ≥ Manhattan lower bound 18 | yes | PASS |
| Lab map: length == 20 (18 + one forced detour down and up) | 20 | PASS |
| Lab map: BFS length equals independent shortest distance (Bellman-Ford relaxation, no queue) | equal | PASS |
| G completely walled off | "no path" message | PASS |
| G next to S | `["Right"]` | PASS |
| Open 3 × 3 room | 4 moves | PASS |
| Two routes, of length 4 and 8 | takes the 4-move route | PASS |
| Gap in the outer wall next to S | never steps off the grid | PASS |
| Depth-first search on the lab map | valid path, no shorter than BFS | PASS (22 moves vs. 20) |

**Questions**

**1. Did the LLM generate a working program on the first attempt?**
Yes for this run: the program ran on its first execution, found a valid 20-move path, and all 11 tests passed, with no fixes needed. That is not guaranteed in general. What made it easy to trust was not the code reading well but the checks around it: replaying the plan, the lower-bound argument, and an independent shortest-distance oracle. If the tests had shown a failure, the next step would have been to paste the failing output back to the LLM and ask for a fix.

**2. If not, how could the prompt be improved?**
It worked, but the prompt could still be tightened. Its main improvements over the handout's suggested prompt were to ask for a *shortest* path explicitly ("collision-free" alone would accept any path), to define the actions and their costs, and to ask for tests, including a failure case. Other things worth specifying up front: the coordinate convention and output format, what to do at the edge of the grid (a map with a gap in the border), and a request to report the number of squares expanded, so different algorithms can be compared.

**3. What search algorithm did the LLM choose?**
Breadth-first search, with a FIFO queue for the frontier, a dictionary of parent pointers that doubles as the visited set, and path reconstruction by following the parent pointers back from G.

**4. Why do you think the LLM selected this algorithm?**
- Every move costs 1, so BFS reaches each square by a shortest path, and the first time it reaches G it has a shortest path. The prompt asks for a shortest path, and BFS guarantees it without any extra machinery.
- It is complete on a finite grid, so if G is unreachable it terminates and the agent can print "no path", which the handout requires.
- It needs no heuristic and is simple to write correctly.
- Depth-first search is the natural alternative, and it is also complete, but it does not guarantee a short path. On this map it returned a 22-move path, against BFS's 20 (see Tests).
- A* would also be optimal and would expand fewer squares, but it needs a heuristic, and at 55 expanded squares on this map the extra complexity is not justified. It becomes attractive for larger warehouses (see the Task 1 discussion).

---

## Summary of what was run

| Item | Result |
|---|---|
| Path on the lab map | 20 moves (optimal), 55 squares expanded |
| No-path case | reports that no path exists |
| Tests | 11/11 pass |
| Scaling (n = 10 to 160) | squares expanded grow in proportion to the area (n² − 2) |
