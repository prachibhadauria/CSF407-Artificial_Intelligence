% Lab 4 - Logical Reasoning for Planning: optional Prolog extension
% CS F407 Artificial Intelligence
%
% Covers Tasks 6, 7 and 8 of the handout. Run the self-checks with
%
%     swipl -q -g run_tests -t halt planner.pl
%
% or consult the file in the SWI-Prolog toplevel (swipl planner.pl) and type
% the queries shown in the comments. `python3 planner.py` runs this file too
% and records the output in results.txt.

% ---------------------------------------------------------------------------
% Introductory example (handout, section 7.1): facts + rules + inference
% ---------------------------------------------------------------------------
penguin(polly).

bird(X) :-
    penguin(X).

animal(X) :-
    bird(X).

% ?- animal(polly).          true.   (penguin(polly) => bird(polly) => animal(polly))

% ---------------------------------------------------------------------------
% Task 6: Prolog as a plan verifier
% ---------------------------------------------------------------------------
% Facts: the warehouse has locations a, b, c connected as a - b - c.
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

% Rule: the robot can move directly from X to Y if they are connected.
% Logically: Connected(X,Y) -> CanMove(X,Y), for all X and Y.
can_move(X,Y) :-
    connected(X,Y).

% ?- can_move(a,b).          true.
% ?- can_move(a,c).          false.  (connected(a,c) is not a fact and no rule derives it)

% ---------------------------------------------------------------------------
% Task 7: using Prolog to check a proposed plan
% ---------------------------------------------------------------------------
valid_move(X,Y) :-
    connected(X,Y).

% ?- valid_move(a,b).        true.
% ?- valid_move(b,c).        true.
% ?- valid_move(a,c).        false.  (so a proposed Move(a,c) is NOT supported)

% Extension: check a whole route, i.e. a list of locations the robot visits.
% The route [a,b,c] is the sequence of moves Move(a,b), Move(b,c).
valid_path([_]).
valid_path([X,Y|Rest]) :-
    valid_move(X,Y),
    valid_path([Y|Rest]).

% ?- valid_path([a,b,c]).    true.
% ?- valid_path([a,c]).      false.

% ---------------------------------------------------------------------------
% Task 8: chained inference  Fact => Rule => Rule => Conclusion
% ---------------------------------------------------------------------------
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.

% ?- reduce_speed.           true.

% ---------------------------------------------------------------------------
% Self-checks: swipl -q -g run_tests -t halt planner.pl
% ---------------------------------------------------------------------------
% test(Query, ExpectedAnswer)
test(animal(polly),                      true).
test(can_move(a,b),                      true).
test(can_move(a,c),                      false).
test(findall(Y, can_move(a,Y), [b]),     true).
test(findall(Y, can_move(b,Y), [a,c]),   true).
test(valid_move(a,b),                    true).
test(valid_move(b,c),                    true).
test(valid_move(a,c),                    false).
test(valid_path([a,b,c]),                true).
test(valid_path([a,c]),                  false).
test(reduce_speed,                       true).
test(slippery,                           true).
test(wet_road,                           true).
test(animal(tweety),                     false).

check(Goal, Expected, Status) :-
    (   call(Goal) -> Actual = true ; Actual = false ),
    (   Actual == Expected -> Status = 'PASS' ; Status = 'FAIL' ),
    \+ \+ ( numbervars(Goal, 0, _),
             format("[~w] ?- ~W.   ~w~n",
                    [Status, Goal, [quoted(true), numbervars(true)], Actual]) ).

run_tests :-
    findall(S, (test(G, E), check(G, E, S)), Statuses),
    length(Statuses, N),
    include(==('PASS'), Statuses, Passed),
    length(Passed, P),
    format("~d/~d Prolog checks passed~n", [P, N]),
    (   P =:= N -> true ; halt(1) ).
