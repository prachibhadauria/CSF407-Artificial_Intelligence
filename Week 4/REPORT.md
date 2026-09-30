# Lab 4: Logical Reasoning for Planning

Course: CS F407 Artificial Intelligence      *Logic + Search = Planning*

**Files**

- `planner.py` is the Python planning agent: propositions, actions, the BFS planner, an independent plan verifier, and all the tests. It also runs `planner.pl`.
- `planner.pl` is the optional Prolog extension (handout Tasks 6 to 8).
- `results.txt` is the full output of `python3 planner.py`. Every number and plan in this report is copied from it.

**Setup:** Python 3.11, standard library only. The Prolog part uses SWI-Prolog 9.0.4 (`swipl`); `planner.py` skips it, and says so, if `swipl` is not installed. The script exits with code 0 only if every check passes (`28/28 checks passed`).

```bash
python3 planner.py                              # everything; writes results.txt
swipl -q -g run_tests -t halt planner.pl        # just the Prolog checks (14/14 pass)
```

---

## 1. Specification of the planning problem (Task 0)

The planning problem is (I, A, G).

**(a) Initial state**
I = { At(Robot,A), At(Package,A) }

**(b) Goal**
G = { At(Package,C) }

**(c) Actions.** Locations are A, B, C, connected as A – B – C.

- `Move(X,Y)` for each connected pair: (A,B), (B,A), (B,C), (C,B);
- `PickUp(Package,L)` for L in {A, B, C};
- `Drop(Package,L)` for L in {A, B, C}.

**(d) Preconditions and effects** (from `results.txt`; the planner represents these as positive/negative preconditions and positive/negative effects):

| Action | Preconditions | Effects |
|---|---|---|
| `Move(X,Y)` | At(Robot,X) | ¬At(Robot,X), At(Robot,Y) |
| `PickUp(Package,L)` | At(Robot,L), At(Package,L) | ¬At(Package,L), Holding(Package) |
| `Drop(Package,L)` | At(Robot,L), Holding(Package) | ¬Holding(Package), At(Package,L) |

The handout's action descriptions use only positive preconditions, so none of these actions has a negative precondition. The planner supports them anyway (as the handout's prompt asks), and an extra test exercises that code path.

**Which actions are applicable in I?** An action is applicable in S only if S ⊨ Preconditions(a), meaning every precondition is satisfied in the current state.

- **PickUp(Package,A): applicable.** Its preconditions are At(Robot,A) and At(Package,A), and both are in I.
- **Drop(Package,C): not applicable.** It needs At(Robot,C) and Holding(Package). Neither is in I: the robot is at A, not C, and it is not holding the package yet.

Across all actions, exactly two are applicable in I: `Move(A,B)` and `PickUp(Package,A)`.

> **Think about it: an action is not applicable just because it appears in the list of available actions.**
> Being *available* (part of the action set A) only says the action exists. Being *applicable* is a property of the action *and the current state*: all its preconditions must hold in that state. Drop(Package,C) is available but not applicable in I. That test, S ⊨ Preconditions(a), is where logical reasoning enters planning.

---

## 2. The plan constructed by hand (Task 1)

The goal is At(Package,C), so the package has to be picked up at A, carried B and then C, and dropped at C.

| State | After action | Facts |
|---|---|---|
| S0 | (initial) | At(Robot,A), At(Package,A) |
| S1 | PickUp(Package,A) | At(Robot,A), Holding(Package) |
| S2 | Move(A,B) | At(Robot,B), Holding(Package) |
| S3 | Move(B,C) | At(Robot,C), Holding(Package) |
| S4 | Drop(Package,C) | At(Robot,C), At(Package,C) |

S4 ⊨ G, so the plan is valid. `planner.py` replays it with the independent verifier and the states match this table at every step.

**The example in the handout is not a valid plan.** The handout says you "might need to reason about Move(A,B), PickUp(Package,B), Move(B,C), Drop(Package,C)". The package is at A, not B, so after Move(A,B) the action PickUp(Package,B) is not applicable: At(Package,B) is false. The verifier rejects this sequence at step 2 (see `results.txt`). The working order has to pick the package up *before* moving, at A.

---

## 3. The prompt used with the LLM (Task 2)

**LLM used:** Claude (Anthropic).

The handout's suggested prompt, used as written:

> I want to implement a simple planning agent in Python.
> Represent a state as a set of logical propositions.
> Each action should contain:
> - a name;
> - positive preconditions;
> - negative preconditions;
> - positive effects;
> - negative effects.
>
> An action is applicable if all of its preconditions are satisfied by the current state.
> When an action is applied:
> 1. remove its negative effects from the state;
> 2. add its positive effects to the state.
>
> Use breadth-first search to find a sequence of actions that achieves a specified goal.
> The program should also:
> - detect when no plan exists;
> - print the resulting sequence of actions;
> - print the states reached after each action.
>
> Explain the implementation and identify any assumptions you make.
> Run the generated program on the warehouse problem.

**Follow-up prompt** (for the tests and verification):

> Add the three tests from the lab (solvable, impossible because PickUp is removed, irrelevant action that moves the robot but not the package). Write a separate verifier that replays a plan and checks each action's preconditions one at a time, without using the planner's own applicability code. Also add an exhaustive check of the shortest plan length that does not use BFS.

**Prompt for the optional Prolog part:**

> Write planner.pl for this warehouse: connected/2 facts, can_move/2, valid_move/2, and the wet-road example, with self-checks I can run with swipl.

---

## 4. The generated program (and what was generated or modified)

`planner.py`, line numbers as in the file.

**Think about it: where the specification's ideas appear in the program**

| Idea | Question | Where |
|---|---|---|
| Preconditions | When is an action applicable? | `Action.applicable` (line 52): `self.pos_pre <= state and not (self.neg_pre & state)`. It is called in `bfs_plan` at line 138 before any action is applied. |
| Effects | How does the state change? | `Action.apply` (line 57): `(state - self.neg_eff) | self.pos_eff`, which is remove the negative effects, then add the positive ones. |
| Goal | When does planning terminate? | `bfs_plan` line 129: `if goal <= state`, meaning every goal proposition is in the state. Planning also stops, with `No plan found`, when the queue is empty. |
| BFS | How are alternative plans explored? | `bfs_plan` lines 124 to 143: a FIFO queue (`deque`) of states, `parent` pointers for reconstructing the plan, and a `parent` check so a state is never queued twice. Shorter plans are always explored before longer ones, so the first goal found is a shortest plan. |

**Generated by the LLM, modified, and added.** The LLM wrote the structure (the `Action` class, the BFS planner, the warehouse domain, the printing of plans and states). What was added or changed afterwards, as a result of verification:

1. **The action set.** The handout describes each action with an example (`Move(A,B)`, `PickUp(Package,A)`, `Drop(Package,C)`). I generalised `PickUp` and `Drop` to all three locations so that Drop means "at the current location", as the handout states.
2. **The independent verifier** (`verify_plan`). It is a second code path that checks each precondition one at a time and builds the new state fact by fact, so it does not share `applicable` or `apply` with the planner. Without it, a bug in `applicable` would make the planner and its "check" agree with each other and prove nothing.
3. **An exhaustive length check** (`shortest_by_brute_force`), which tries every action sequence without a queue or visited set.
4. **A switch `check_preconditions=False`**, used only to demonstrate what goes wrong without the logic (Reflection question 2).

The BFS and the `applicable`/`apply` logic were kept as generated. They passed every test.

---

## 5. Results of the tests (Task 3)

All tests are in `results.txt`. In each case the plan was validated by the independent verifier.

| Test | Initial state | Goal | Plan found? | Resulting plan | Actually valid? |
|---|---|---|:-:|---|:-:|
| **A** solvable (original warehouse) | {At(Robot,A), At(Package,A)} | {At(Package,C)} | yes | PickUp(Package,A), Move(A,B), Move(B,C), Drop(Package,C) | yes |
| **B** impossible (PickUp removed) | same | same | no: **No plan found** | none | n/a |
| **C1** irrelevant action `Wave(Robot)` | same | same | yes | the same 4 actions; `Wave` is not used | yes |
| **C2** robot-only action `MoveFast(A,C)` | same | same | yes | PickUp(Package,A), MoveFast(A,C), Drop(Package,C) | yes |
| **C3** `MoveFast(A,C)` but PickUp removed | same | same | no: **No plan found** | none | n/a |

**Test A.** The plan has 4 actions (the search expanded 7 states). Verifying every action: each precondition was true in the state where its action executes, and the final state {At(Package,C), At(Robot,C)} satisfies the goal. The verifier's states equal the planner's states at every step. An exhaustive enumeration of every action sequence finds no plan shorter than 4, so BFS returned a shortest plan.

**Test B.** With PickUp removed, the package can never leave A, so the goal is unreachable. The planner reports `No plan found` after expanding 3 states (the only reachable ones) and does not invent an action. The exhaustive enumeration also finds no plan up to length 5.

**Test C.** Two versions of the handout's "an action that moves the robot but does not move the package":
- **C1:** `Wave(Robot)`, which changes nothing the goal mentions. The planner finds the same 4-action plan and never uses it (it only costs more search: 12 expanded states instead of 7).
- **C2:** `MoveFast(A,C)`, a shortcut that moves only the robot. The plan shrinks to 3 actions, but it still has PickUp and Drop: the robot reaching C does not count as the package reaching C. For contrast, with the goal {At(Robot,C)} the plan is just `MoveFast(A,C)`.
- **C3:** if `MoveFast(A,C)` is available but PickUp is removed, the robot *can* reach C, yet the planner still reports `No plan found` for the package. This is the direct test that "robot at C" is not treated as "package at C".

**Extra tests** (beyond the handout's three), all passing:
- a goal that is already true returns the empty plan;
- with the package starting at B, the plan is Move(A,B), PickUp(Package,B), Move(B,C), Drop(Package,C);
- a goal with two propositions is achieved;
- negative preconditions are honoured (an action is applicable when its negative precondition is absent and not applicable when it is present);
- a planner with the precondition check switched off returns the one-action "plan" `Drop(Package,C)`, which the verifier rejects (see Reflection question 2).

---

## 6. Logic and search (Task 4)

The planner uses two different ideas.

- **Logical reasoning** decides *what is possible*: whether S ⊨ Preconditions(a) (`Action.applicable`), and what the world looks like afterwards, S′ = Apply(S, a) (`Action.apply`).
- **Search** decides *what to try*: which of the applicable actions to expand, in what order, and when to stop (`bfs_plan`).

**Completing the diagram:**

```
Current state
      ↓
Check action preconditions        (logic: S ⊨ Preconditions(a) for each action a)
      ↓
Select the applicable actions and apply their effects    <-- the "?" step
   (logic: remove the negative effects, add the positive effects)
      ↓
Generate successor state
      ↓
Search over alternatives          (put new states on the BFS queue, skip ones already seen)
      ↓
Goal?                             (logic again: does the state satisfy G?)
```

**How they work together.** From the current state, the logical part tests every action's preconditions and keeps only the applicable ones. For each of those it computes the successor state, so logic fully describes the legal moves out of a state. Search does not know anything about robots or packages. It takes those legal successors, puts the new ones on a queue, and keeps taking the oldest state, so all shorter plans are tried before longer ones. After every step the goal test, which is also a logical check (G ⊆ S), decides whether to stop. A wrong logic component would make search explore impossible worlds (Test B showed that the planner does not invent a PickUp); a missing search component would leave a correct description of the rules but no way to find a sequence of actions.

> **Think about it: "Logic determines what is possible; search determines what to try."**
> In Test A, at S0 logic says only `Move(A,B)` and `PickUp(Package,A)` are possible (the other eight actions fail their preconditions). Search then tries both, in order, and finds that only the PickUp branch leads to the goal in four steps. This is the same structure as the search module: the actions and transition function come from the problem (here, logic), and the search strategy (BFS) chooses the order.

---

## 7. Optional: can the LLM verify its own plan? (Task 5)

The handout's question to put to the LLM is: "For every action in the plan, identify its preconditions and show that those preconditions are satisfied in the state in which the action is executed." The LLM's (Claude's) explanation for this plan said, for example, that `PickUp(Package,A)` needs At(Robot,A) and At(Package,A), both true in S0, giving S1 = {At(Robot,A), Holding(Package)}; that `Move(A,B)` needs At(Robot,A), true in S1; that `Move(B,C)` needs At(Robot,B), true in S2; and that `Drop(Package,C)` needs At(Robot,C) and Holding(Package), both true in S3, giving S4 = {At(Robot,C), At(Package,C)}.

I then compared this, step by step, with the states computed by the program (`planner.py` holds the explanation's claims as data and checks them). In this run **the explanation matched the executed transitions at every step**: both the claimed preconditions and the claimed resulting states.

**Which should you trust more: (a) the LLM's explanation, or (b) the independently executed state transitions?**
(b). The executed transitions are a mechanical computation that replays the plan from I, checking each precondition against the actual state. The explanation is text that can be fluent and plausible whether or not it is true. Here the two agree, but only the comparison shows it. The handout's own example plan (`Move(A,B), PickUp(Package,B), ...`) is a good illustration: it reads like a reasonable plan, yet the verifier shows that step 2 fails. A generated explanation is not the same as an independent verification.

---

## 8. Optional extension: Prolog as a logical verifier (Tasks 6 to 8)

`planner.pl` holds all the Prolog. I ran it with SWI-Prolog 9.0.4. The 14 self-checks all pass (`14/14 Prolog checks passed`), and `planner.py` also runs them.

### Task 6: Prolog as a plan verifier

```prolog
connected(a,b).   connected(b,a).   connected(b,c).   connected(c,b).
can_move(X,Y) :- connected(X,Y).
```

| Query | Result |
|---|---|
| `?- can_move(a,b).` | true |
| `?- can_move(a,c).` | false |

**(a) Why does Prolog return true for can_move(a,b)?** The fact `connected(a,b)` is in the knowledge base. To prove `can_move(a,b)`, Prolog uses the rule `can_move(X,Y) :- connected(X,Y)` with X = a and Y = b, which leaves the subgoal `connected(a,b)`, and that is a fact.

**(b) Why does it not establish can_move(a,c)?** The same rule would need `connected(a,c)`, which is not a fact and has no rule deriving it (a and c are not directly connected; b is between them). Prolog therefore cannot prove it and answers `false`. Note that Prolog's `false` means "cannot be proved from this knowledge base" (negation as failure), not "proved to be false".

**(c) Relationship with the implication Connected(X,Y) → CanMove(X,Y).** The rule is that implication, read backwards: `head :- body` is "body → head", with X and Y universally quantified. Prolog answers a query by working backwards from the head to the body until it reaches facts.

### Task 7: using Prolog to check a proposed plan

The Python planner's moves are `Move(a,b)` and `Move(b,c)`. With `valid_move(X,Y) :- connected(X,Y).`:

| Query | Result |
|---|---|
| `?- valid_move(a,b).` | true |
| `?- valid_move(b,c).` | true |
| `?- valid_move(a,c).` | false |

As the handout says, the first two succeed and the third fails.

**Challenge: the planner proposes Move(a,c).** Prolog answers `?- valid_move(a,c).` with `false`: that move is not supported by the warehouse knowledge, because there is no fact `connected(a,c)`. To connect the two programs, `planner.pl` also has `valid_path/1`, which checks a whole route, and `planner.py` passes the route of each plan to it:

| Plan from Python | Route | Prolog `valid_path` |
|---|---|---|
| Test A plan: PickUp, Move(A,B), Move(B,C), Drop | `[a,b,c]` | true |
| Test C2 plan (uses the invented `MoveFast(A,C)`) | `[a,c]` | **false** |

The Python planner accepted `MoveFast(A,C)` because it was told that action exists. The Prolog description of the warehouse says a and c are not directly connected, so the independent check flags it. The two descriptions of the warehouse disagree, and that disagreement is exactly what an independent verifier is for.

> **Think about it: Generate → Independent verification.**
> The Python program generates the candidate action or plan; Prolog, which was given only the logical description of the warehouse, checks it. They share no code, so a bug in one is unlikely to be repeated in the other.

### Task 8: connecting Prolog to logical reasoning

```prolog
wet_road.
slippery     :- wet_road.
reduce_speed :- slippery.
```

The query `?- reduce_speed.` succeeds (`true`). Prolog finds the rule `reduce_speed :- slippery`, so it needs `slippery`; the rule `slippery :- wet_road` needs `wet_road`; and `wet_road` is a fact. The corresponding logical reasoning:

**wet_road (fact) ⇒ (wet_road → slippery) ⇒ (slippery → reduce_speed) ⇒ reduce_speed (conclusion)**

> **Think about it: Prolog is not identical to classical logic.**
> Prolog's execution strategy (unification, searching the rules in order, backtracking) is a particular way of searching for a proof, and some constructs, such as negation as failure (point (b) above), do not mean the same as classical negation. For this exercise only the basic idea is needed: facts + rules → inference → query answer.

### 7.2 Reflection

1. **Fact vs rule.** A fact is an unconditional statement that is taken to be true (`connected(a,b).`). A rule is a conditional statement, `head :- body`: the head is true if every goal in the body can be proved.
2. **Query and entailment.** A query asks whether something follows from the knowledge base. Prolog tries to prove it from the facts and rules; `true` means it found a proof, and `false` means it found none.
3. **Why verify a Python plan with Prolog?** The plan generator may contain bugs or invented actions. Prolog holds the knowledge about the warehouse separately and declaratively, so it can check each proposed move (and whole routes) against that knowledge without sharing any code with the planner.
4. **Advantage of an independent verifier for LLM-assisted work.** An LLM can produce plausible but wrong code or plans, and its explanations can sound equally convincing. A verifier written separately and executed mechanically gives evidence that does not depend on the LLM being right. Here it would catch the `MoveFast(A,C)` shortcut that the warehouse facts do not support.

---

## 9. Reflection on the use of the LLM and the reflection questions

**Which parts were generated, modified and tested.** The LLM generated the planner's structure and the Prolog file; I checked and extended them as described in section 4 (action set, verifier, exhaustive check, precondition switch). What was tested: every test in section 5, the extra tests, the Prolog self-checks, and the agreement between the Python planner and the Prolog description (section 8).

**1. Why is it useful to specify action preconditions and effects before asking an LLM to write the planner?**
They define what "correct" means. The LLM then translates a precise specification instead of guessing the domain, and you have a reference to check the code against: each precondition and effect in the table in section 1 became a test. If you only ask it to "solve the problem", you get a plan you cannot easily check, and you cannot tell a wrong rule from a wrong implementation.

**2. Give an example of an error that could occur if the planner failed to check an action's preconditions.**
It can produce a plan that "works" on paper but cannot be executed. I ran exactly that case: with `check_preconditions=False`, BFS returned the one-action plan `Drop(Package,C)` for the goal At(Package,C). Applying the effects alone makes the goal true, but the robot is at A and is not holding anything; the verifier shows both preconditions (At(Robot,C) and Holding(Package)) are false.

**3. Why is a plan that "looks reasonable" not necessarily a valid plan?**
Validity depends on the preconditions holding in the exact state in which each action runs, and that is easy to get wrong by eye. The handout's own example sequence (Move(A,B), PickUp(Package,B), ...) looks sensible but fails at step 2, because the package is still at A. The one-action plan `Drop(Package,C)` also looks like a solution.

**4. What did the LLM contribute to the implementation?**
The structure and boilerplate: the `Action` class, the BFS with parent pointers, the domain, the printing of plans and states, the test scaffolding and the Prolog file. It also produced the explanation in Task 5. It did not decide what counts as correct; that came from the specification and the independent checks.

**5. What did you have to verify independently?**
That the plans are valid (the separate verifier), that the plan is a shortest one (the exhaustive enumeration), that robot-at-C is not confused with package-at-C (Test C2/C3), that "No plan found" appears when it should (Test B), that the handout's example plan really is invalid, and that the LLM's explanation matches the executed transitions.

**6. In this laboratory, where is logical reasoning being used?**
In the applicability test S ⊨ Preconditions(a), in the state update S′ = Apply(S, a), and in the goal test S ⊨ G. In the optional part, it is also in the Prolog knowledge base, where `can_move`, `valid_move` and `reduce_speed` are rules and queries are answered by inference (e.g. wet_road ⇒ slippery ⇒ reduce_speed).

**7. How is planning related to the search algorithms studied in the previous module?**
A planning problem is a search problem: states are sets of propositions, the successor function comes from the actions that are applicable in a state, the initial state is I, the goal test is G ⊆ S, and each action costs 1. BFS here is the same algorithm as in the search lab, and it gives the same guarantee (a shortest plan, for unit costs). What is new is that the states and transitions are not listed in advance: they are *computed* by logic from the action definitions. That is what Logic + Search = Planning means.

**Takeaway.** Understand → Specify → Generate → Execute → Verify: each stage has a concrete artefact here (section 1 specification, section 3 prompt, `planner.py` and `planner.pl`, `results.txt`, and the verifier and Prolog checks).
