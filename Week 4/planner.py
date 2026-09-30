"""
Lab 4 - Logical Reasoning for Planning: a simple planning agent
CS F407 Artificial Intelligence      Logic + Search = Planning

A STRIPS-style planner for the warehouse robot:

  * a state is a set of logical propositions, e.g. {"At(Robot,A)", "At(Package,A)"}
  * an action has positive/negative preconditions and positive/negative effects
  * logic:   an action is applicable in S if S |= Preconditions(a)
             S' = (S - negative effects) | positive effects
  * search:  breadth-first search over states finds a shortest plan, or
             reports "No plan found"

It also runs the handout's tests (A, B, C), independently re-checks every plan
with a separate verifier, and (if SWI-Prolog is installed) runs planner.pl,
the optional Prolog extension, and uses it to check the planner's moves.

Usage:
    python3 planner.py      # runs everything and writes results.txt
                            # (exit code 0 only if every check passes)

Only the Python 3 standard library is needed. planner.pl needs SWI-Prolog
(`swipl`); without it that part is skipped and reported as skipped.
"""

import io
import itertools
import os
import shutil
import subprocess
import sys
from collections import deque
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Logic: states, actions, applicability, effects
# ---------------------------------------------------------------------------
class Action:
    """name, positive preconditions, negative preconditions, positive effects,
    negative effects (all sets of proposition strings)."""

    def __init__(self, name, pos_pre=(), neg_pre=(), pos_eff=(), neg_eff=()):
        self.name = name
        self.pos_pre = frozenset(pos_pre)
        self.neg_pre = frozenset(neg_pre)
        self.pos_eff = frozenset(pos_eff)
        self.neg_eff = frozenset(neg_eff)

    def applicable(self, state):
        """S |= Preconditions(a): every positive precondition is in S and no
        negative precondition is in S."""
        return self.pos_pre <= state and not (self.neg_pre & state)

    def apply(self, state):
        """1. remove the negative effects, 2. add the positive effects."""
        return (state - self.neg_eff) | self.pos_eff

    def unmet(self, state):
        """Which preconditions fail in `state` (for explanations)."""
        missing = sorted(self.pos_pre - state)
        violated = sorted(f"not {p}" for p in self.neg_pre & state)
        return missing + violated


def fmt_state(state):
    return "{" + ", ".join(sorted(state)) + "}"


LOCATIONS = ["A", "B", "C"]
CONNECTED = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]


def robot(x):
    return f"At(Robot,{x})"


def package(x):
    return f"At(Package,{x})"


HOLDING = "Holding(Package)"


def warehouse_actions(include_pickup=True):
    """The actions of the handout's warehouse."""
    acts = [Action(f"Move({x},{y})", pos_pre=[robot(x)], pos_eff=[robot(y)], neg_eff=[robot(x)])
            for x, y in CONNECTED]
    if include_pickup:
        acts += [Action(f"PickUp(Package,{l})", pos_pre=[robot(l), package(l)],
                        pos_eff=[HOLDING], neg_eff=[package(l)]) for l in LOCATIONS]
    acts += [Action(f"Drop(Package,{l})", pos_pre=[robot(l), HOLDING],
                    pos_eff=[package(l)], neg_eff=[HOLDING]) for l in LOCATIONS]
    return acts


INITIAL = frozenset({robot("A"), package("A")})
GOAL = frozenset({package("C")})


# ---------------------------------------------------------------------------
# Search: breadth-first planner
# ---------------------------------------------------------------------------
class PlanResult:
    def __init__(self, plan, states, expanded):
        self.found = plan is not None
        self.plan = plan                  # list of Action, or None
        self.states = states              # [S0, S1, ..., Sn], or None
        self.expanded = expanded          # states expanded by the search

    def names(self):
        return [a.name for a in self.plan] if self.found else None


def bfs_plan(initial, goal, actions, check_preconditions=True):
    """Breadth-first search over sets of propositions.

    Goal test: goal <= state (the state satisfies every goal proposition).
    With check_preconditions=False the planner wrongly applies every action in
    every state; it exists only to show what goes wrong without the logic."""
    initial = frozenset(initial)
    queue = deque([initial])
    parent = {initial: None}                    # state -> (previous state, action)
    expanded = 0
    while queue:
        state = queue.popleft()
        if goal <= state:
            plan, states = [], [state]
            while parent[state] is not None:
                state, action = parent[state]
                plan.append(action)
                states.append(state)
            return PlanResult(plan[::-1], states[::-1], expanded)
        expanded += 1
        for action in actions:
            if check_preconditions and not action.applicable(state):
                continue
            nxt = action.apply(state)
            if nxt not in parent:
                parent[nxt] = (state, action)
                queue.append(nxt)
    return PlanResult(None, None, expanded)


# ---------------------------------------------------------------------------
# Independent verification (separate code path from Action.applicable/apply)
# ---------------------------------------------------------------------------
def verify_plan(initial, goal, names, actions, verbose=False):
    """Replay a list of action names from `initial`. For every step, test each
    precondition one at a time, then compute the new state fact by fact.
    Returns (valid, failing_step_or_None, list_of_states)."""
    by_name = {a.name: a for a in actions}
    state = set(initial)
    states = [frozenset(state)]
    for i, name in enumerate(names, 1):
        a = by_name.get(name)
        if a is None:
            if verbose:
                print(f"  step {i}: {name}: not an available action")
            return False, i, states
        ok = True
        for p in sorted(a.pos_pre):
            holds = p in state
            ok &= holds
            if verbose:
                print(f"  step {i}: {name}: precondition {p}: {'true' if holds else 'FALSE'}")
        for p in sorted(a.neg_pre):
            holds = p not in state
            ok &= holds
            if verbose:
                print(f"  step {i}: {name}: negative precondition not {p}: {'true' if holds else 'FALSE'}")
        if not ok:
            if verbose:
                print(f"  step {i}: {name} is NOT applicable in {fmt_state(state)}")
            return False, i, states
        for p in a.neg_eff:
            state.discard(p)
        for p in a.pos_eff:
            state.add(p)
        states.append(frozenset(state))
        if verbose:
            print(f"  step {i}: {name} -> {fmt_state(state)}")
    return goal <= state, None, states


def shortest_by_brute_force(initial, goal, actions, max_len):
    """Independent length oracle: try every action sequence of length 0, 1, ...
    (no queue, no visited set) and return the first length that verifies."""
    names = [a.name for a in actions]
    for n in range(max_len + 1):
        for seq in itertools.product(names, repeat=n):
            if verify_plan(initial, goal, seq, actions)[0]:
                return n
    return None


# ---------------------------------------------------------------------------
# Optional Prolog extension (planner.pl), run with SWI-Prolog when available
# ---------------------------------------------------------------------------
PL_FILE = os.path.join(HERE, "planner.pl")
SWIPL = shutil.which("swipl")


def prolog_holds(goal):
    """Ask planner.pl whether `goal` succeeds. Returns True/False, or None if
    SWI-Prolog is not available."""
    if not SWIPL:
        return None
    out = subprocess.run(
        [SWIPL, "-q", "-g", f"({goal} -> writeln(true) ; writeln(false))", "-t", "halt", PL_FILE],
        capture_output=True, text=True, timeout=60)
    return out.stdout.strip().splitlines()[-1] == "true" if out.stdout.strip() else None


def route_of(plan_names):
    """Locations visited by the Move actions of a plan, in Prolog syntax,
    e.g. ['Move(A,B)', 'Move(B,C)'] -> [a,b,c]. None if there are no moves."""
    moves = [n[5:-1].split(",") for n in plan_names if n.startswith("Move")]
    if not moves:
        return None
    route = [moves[0][0]] + [m[1] for m in moves]
    return "[" + ",".join(x.lower() for x in route) + "]"


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def show_plan(res):
    if not res.found:
        print("  No plan found")
        print(f"  (search expanded {res.expanded} states)")
        return
    print(f"  plan found: {len(res.plan)} actions (search expanded {res.expanded} states)")
    print(f"  S0: {fmt_state(res.states[0])}")
    for i, a in enumerate(res.plan, 1):
        print(f"  a{i} = {a.name}")
        print(f"  S{i}: {fmt_state(res.states[i])}")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
def task0_and_1(outcomes, check):
    print("=" * 72 + "\nTASK 0: THE PLANNING PROBLEM\n" + "=" * 72)
    print(f"I = {fmt_state(INITIAL)}")
    print(f"G = {fmt_state(GOAL)}")
    acts = warehouse_actions()
    print("\nActions (name | positive preconditions | effects):")
    for a in acts:
        eff = [f"not {p}" for p in sorted(a.neg_eff)] + sorted(a.pos_eff)
        print(f"  {a.name:<20} | {', '.join(sorted(a.pos_pre)):<36} | {', '.join(eff)}")
    by_name = {a.name: a for a in acts}
    print("\nApplicable in I?")
    for name in ("PickUp(Package,A)", "Drop(Package,C)"):
        a = by_name[name]
        ok = a.applicable(INITIAL)
        print(f"  {name}: {'applicable' if ok else 'NOT applicable, unmet: ' + ', '.join(a.unmet(INITIAL))}")
    check("PickUp(Package,A) is applicable in I", by_name["PickUp(Package,A)"].applicable(INITIAL))
    check("Drop(Package,C) is not applicable in I",
          not by_name["Drop(Package,C)"].applicable(INITIAL))
    print("\nAll actions applicable in I: "
          + ", ".join(a.name for a in acts if a.applicable(INITIAL)))

    print("\n" + "=" * 72 + "\nTASK 1: THE PLAN CONSTRUCTED BY HAND, CHECKED BY THE VERIFIER\n" + "=" * 72)
    hand = ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"]
    print("hand-constructed plan: " + ", ".join(hand))
    valid, _, states = verify_plan(INITIAL, GOAL, hand, acts, verbose=True)
    check("the hand-constructed plan is valid and achieves the goal", valid)
    expected = [
        {"At(Robot,A)", "At(Package,A)"},
        {"At(Robot,A)", "Holding(Package)"},
        {"At(Robot,B)", "Holding(Package)"},
        {"At(Robot,C)", "Holding(Package)"},
        {"At(Robot,C)", "At(Package,C)"},
    ]
    check("the states after every action equal the hand-written table",
          [set(s) for s in states] == expected)

    # The handout's "for example" sequence mentions PickUp(Package,B) after Move(A,B).
    print("\nThe example sequence named in the handout: Move(A,B), PickUp(Package,B), "
          "Move(B,C), Drop(Package,C)")
    valid, step, _ = verify_plan(INITIAL, GOAL,
                                 ["Move(A,B)", "PickUp(Package,B)", "Move(B,C)", "Drop(Package,C)"],
                                 acts, verbose=True)
    print(f"  -> valid: {valid}; fails at step {step}")
    check("the handout's example sequence is NOT a valid plan (the package is at A, not B)",
          not valid and step == 2)
    return hand


def task3_tests(check):
    print("\n" + "=" * 72 + "\nTASK 3: TESTING THE PLANNER\n" + "=" * 72)
    acts = warehouse_actions()

    print("\nTest A: solvable problem (original warehouse)")
    print(f"  initial state: {fmt_state(INITIAL)}\n  goal: {fmt_state(GOAL)}")
    res = bfs_plan(INITIAL, GOAL, acts)
    show_plan(res)
    print("  independent verification of every action:")
    valid, _, states = verify_plan(INITIAL, GOAL, res.names(), acts, verbose=True)
    print(f"  plan actually valid: {valid}")
    check("A: a plan is found", res.found)
    check("A: every action's preconditions hold when it is executed, and the goal is reached", valid)
    check("A: the verifier's states equal the planner's states", states == res.states)
    check("A: plan is PickUp(Package,A), Move(A,B), Move(B,C), Drop(Package,C)",
          res.names() == ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"])
    brute = shortest_by_brute_force(INITIAL, GOAL, acts, 5)
    print(f"  shortest plan length by exhaustive enumeration of all sequences: {brute}")
    check("A: BFS plan length equals the exhaustive shortest length (4)", len(res.plan) == brute == 4)

    print("\nTest B: impossible problem (the PickUp actions are removed)")
    no_pick = warehouse_actions(include_pickup=False)
    print(f"  initial state: {fmt_state(INITIAL)}\n  goal: {fmt_state(GOAL)}")
    res = bfs_plan(INITIAL, GOAL, no_pick)
    show_plan(res)
    print(f"  plan actually valid: n/a (no plan)")
    check("B: reports 'No plan found' instead of inventing an action", not res.found)
    check("B: exhaustive enumeration also finds no plan up to length 5",
          shortest_by_brute_force(INITIAL, GOAL, no_pick, 5) is None)

    print("\nTest C1: an irrelevant action, Wave(Robot), which changes nothing the goal mentions")
    wave = Action("Wave(Robot)", pos_eff=["Waved(Robot)"])
    res = bfs_plan(INITIAL, GOAL, acts + [wave])
    show_plan(res)
    valid = verify_plan(INITIAL, GOAL, res.names(), acts + [wave])[0]
    print(f"  plan actually valid: {valid}")
    check("C1: the same 4-action plan is found and Wave is not used",
          res.names() == ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"] and valid)

    print("\nTest C2: an action that moves the robot but not the package, MoveFast(A,C)")
    fast = Action("MoveFast(A,C)", pos_pre=[robot("A")], pos_eff=[robot("C")], neg_eff=[robot("A")])
    res = bfs_plan(INITIAL, GOAL, acts + [fast])
    show_plan(res)
    valid = verify_plan(INITIAL, GOAL, res.names(), acts + [fast])[0]
    print(f"  plan actually valid: {valid}")
    check("C2: the robot reaching C is not treated as the package reaching C "
          "(PickUp and Drop are still in the plan)",
          res.found and valid and res.names() == ["PickUp(Package,A)", "MoveFast(A,C)", "Drop(Package,C)"])
    res_robot = bfs_plan(INITIAL, frozenset({robot("C")}), acts + [fast])
    print(f"  for contrast, goal {{At(Robot,C)}}: plan = {res_robot.names()}")
    check("C2: for the goal At(Robot,C) the planner needs only MoveFast(A,C)",
          res_robot.names() == ["MoveFast(A,C)"])

    print("\nTest C3: MoveFast(A,C) available, but PickUp removed")
    res = bfs_plan(INITIAL, GOAL, no_pick + [fast])
    show_plan(res)
    print(f"  the robot can reach C here: {bfs_plan(INITIAL, frozenset({robot('C')}), no_pick + [fast]).found}")
    check("C3: the robot can reach C but there is still no plan for the package", not res.found)

    return acts


def extra_tests(check):
    print("\n" + "=" * 72 + "\nEXTRA TESTS (beyond the handout's three)\n" + "=" * 72)
    acts = warehouse_actions()

    res = bfs_plan(INITIAL, frozenset({package("A")}), acts)
    print(f"\nGoal already true in I: plan = {res.names()}, states expanded = {res.expanded}")
    check("goal already satisfied: empty plan", res.found and res.plan == [])

    res = bfs_plan(frozenset({robot("A"), package("B")}), GOAL, acts)
    print(f"\nPackage starts at B, robot at A: plan = {res.names()}")
    check("package at B: Move(A,B), PickUp(Package,B), Move(B,C), Drop(Package,C)",
          res.names() == ["Move(A,B)", "PickUp(Package,B)", "Move(B,C)", "Drop(Package,C)"])

    res = bfs_plan(frozenset({robot("A"), package("A")}), frozenset({package("A"), robot("C")}), acts)
    print(f"\nTwo goal propositions, At(Package,A) and At(Robot,C): plan = {res.names()}")
    check("conjunctive goal: both propositions are achieved",
          res.found and frozenset({package("A"), robot("C")}) <= res.states[-1])

    # Negative preconditions: exercise the code path the handout's prompt asks for.
    lock = Action("Lock(Door)", pos_pre=[robot("A")], neg_pre=["Locked(Door)"], pos_eff=["Locked(Door)"])
    start = frozenset({robot("A")})
    print(f"\nNegative precondition: Lock(Door) needs Door NOT locked")
    check("negative precondition satisfied: applicable", lock.applicable(start))
    check("negative precondition violated: not applicable",
          not lock.applicable(start | {"Locked(Door)"}))
    res = bfs_plan(start, frozenset({"Locked(Door)"}), [lock])
    check("negative precondition: plan [Lock(Door)] found", res.names() == ["Lock(Door)"])

    # What goes wrong if preconditions are not checked (Reflection question 2).
    print("\nWhat if the planner did not check preconditions?")
    res = bfs_plan(INITIAL, GOAL, acts, check_preconditions=False)
    show_plan(res)
    valid, step, _ = verify_plan(INITIAL, GOAL, res.names(), acts, verbose=True)
    print(f"  -> the 'plan' {res.names()} looks like a solution, but valid = {valid} "
          f"(first failure at step {step})")
    check("without the precondition check the planner returns an INVALID 1-action plan, "
          "and the verifier rejects it", res.found and len(res.plan) == 1 and not valid)

    # Task 5: an explanation of the plan, compared with the executed transitions.
    print("\n" + "=" * 72 + "\nTASK 5 (optional): AN EXPLANATION OF THE PLAN VS THE EXECUTED TRANSITIONS\n" + "=" * 72)
    claimed = [
        ("PickUp(Package,A)", {"At(Robot,A)", "At(Package,A)"}, {"At(Robot,A)", "Holding(Package)"}),
        ("Move(A,B)", {"At(Robot,A)"}, {"At(Robot,B)", "Holding(Package)"}),
        ("Move(B,C)", {"At(Robot,B)"}, {"At(Robot,C)", "Holding(Package)"}),
        ("Drop(Package,C)", {"At(Robot,C)", "Holding(Package)"}, {"At(Robot,C)", "At(Package,C)"}),
    ]
    print("The explanation below (what the LLM claimed about each step) is compared with the program's transitions.")
    hand = [c[0] for c in claimed]
    _, _, states = verify_plan(INITIAL, GOAL, hand, acts)
    by_name = {a.name: a for a in acts}
    all_match = True
    for i, (name, pre_claim, post_claim) in enumerate(claimed, 1):
        pre_ok = pre_claim == set(by_name[name].pos_pre) and pre_claim <= states[i - 1]
        post_ok = post_claim == set(states[i])
        all_match &= pre_ok and post_ok
        print(f"  step {i} {name}: claimed preconditions match the action and hold in S{i-1}: {pre_ok}; "
              f"claimed S{i} equals the executed state: {post_ok}")
    check("the explanation matches the independently executed transitions at every step", all_match)
    return acts


def task6_to_8(check, plan_names):
    print("\n" + "=" * 72 + "\nTASKS 6-8 (optional): PROLOG, planner.pl\n" + "=" * 72)
    if not SWIPL:
        print("SWI-Prolog (`swipl`) was not found, so planner.pl was NOT run. "
              "Install it and re-run to include this part.")
        return
    version = subprocess.run([SWIPL, "--version"], capture_output=True, text=True).stdout.strip()
    print(f"{version}; running: swipl -q -g run_tests -t halt planner.pl\n")
    out = subprocess.run([SWIPL, "-q", "-g", "run_tests", "-t", "halt", PL_FILE],
                         capture_output=True, text=True, timeout=120)
    print(out.stdout.rstrip())
    if out.stderr.strip():
        print("stderr:", out.stderr.strip())
    check("planner.pl: all Prolog self-checks pass", out.returncode == 0 and "14/14" in out.stdout)

    print("\nChecking the Python planner's moves with the Prolog knowledge base (generate -> verify):")
    route = route_of(plan_names)
    ans = prolog_holds(f"valid_path({route})")
    print(f"  plan {plan_names}\n  route {route}: ?- valid_path({route}).  {ans}")
    check("Prolog accepts the route of the Python planner's plan", ans is True)
    # A proposed Move(a,c) (for example from the invented MoveFast action):
    ans = prolog_holds("valid_move(a,c)")
    print(f"  proposed Move(a,c): ?- valid_move(a,c).  {ans}")
    check("Prolog rejects a proposed Move(a,c)", ans is False)
    ans = prolog_holds("valid_path([a,c])")
    print(f"  route [a,c], the route of the Test C2 plan: ?- valid_path([a,c]).  {ans}")
    check("Prolog flags the route of the Test C2 plan (MoveFast is not a connected move)", ans is False)


def main():
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    outcomes = []

    def check(name, cond):
        outcomes.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    with redirect_stdout(Tee()):
        print(f"Python {sys.version.split()[0]}")
        hand = task0_and_1(outcomes, check)
        task3_tests(check)
        extra_tests(check)
        task6_to_8(check, hand)
        print("\n" + "=" * 72)
        print(f"{sum(outcomes)}/{len(outcomes)} checks passed")

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write(buf.getvalue())
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
