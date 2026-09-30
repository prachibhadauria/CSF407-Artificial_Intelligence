"""
Lab 1 - Neural Models: Learning, Depth, Activations, and Output Layers
CS F407 Artificial Intelligence

Runs every experiment in the handout, in order:

  Task 1   linear baseline (single affine map + sigmoid) and a deep linear stack
  Task 4A  basic learning check for the 2-2-1 network, plus repeated seeds
  Task 4B  backpropagation check: autograd vs hand-derived chain rule vs
           finite differences, and mean-loss gradient = average of per-example
  Task 4C  symmetry experiment (all weights zero, and all weights equal)
  Task 4D  activation experiment (sigmoid / tanh / ReLU from identical weights)
  Task 5   three-class extension with softmax + cross-entropy, p - y check,
           shift invariance and the stable-softmax diagnostic

Usage:
    python3 xor_lab.py              # prints everything and writes results.txt

Only needs Python 3, NumPy and PyTorch (CPU).
"""

import copy
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn as nn

# float64 so that finite-difference and p - y checks are not limited by
# float32 round-off. Training behaviour is the same.
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data (Task 1)
# ---------------------------------------------------------------------------
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])          # XOR / disagreement warning
Y3 = torch.tensor([0, 1, 1, 2])                          # Task 5: 0 off, 1 disagree, 2 both on

# Engineering settings shared by all binary runs (Task 3 / 4A)
SEED = 0
LR = 0.05
STEPS = 3000
EARLY_STEP = 10          # "early training step" at which gradient norms are recorded

ACTIVATIONS = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu}


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fmt(t, digits=4):
    """Compact printing of a tensor."""
    return np.array2string(t.detach().numpy(), precision=digits, suppress_small=True,
                           floatmode="fixed")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class XORNet(nn.Module):
    """2 inputs -> 2 hidden units (nonlinear) -> 1 logit.

    The sigmoid output is folded into BCEWithLogitsLoss during training and
    applied explicitly only when we report probabilities.
    """

    def __init__(self, activation="sigmoid", hidden=2, n_out=1):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)      # W1: (hidden, 2), b1: (hidden,)
        self.fc2 = nn.Linear(hidden, n_out)  # W2: (n_out, hidden), b2: (n_out,)
        self.act_name = activation
        self.act = ACTIVATIONS[activation]

    def forward(self, x, return_hidden=False):
        a1 = self.fc1(x)          # pre-activations
        h1 = self.act(a1)         # hidden representation
        out = self.fc2(h1)        # logits
        if return_hidden:
            return out, a1, h1
        return out


def train(model, loss_fn, targets, steps=STEPS, lr=LR, opt_name="adam",
          early_step=EARLY_STEP, watch=None):
    """Full-batch training. Returns a dict of history and diagnostics.

    watch: optional list of steps at which to snapshot fc1.weight (for 4C).
    """
    if opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses, early_grad, snapshots, grad0 = [], None, {}, None
    first_w = next(model.parameters())         # first-layer weight matrix W1
    for step in range(steps):
        if watch is not None and step in watch:
            snapshots[step] = model.fc1.weight.detach().clone()
        opt.zero_grad()
        logits = model(X)                      # forward pass
        loss = loss_fn(logits, targets)        # scalar loss
        loss.backward()                        # reverse-mode AD
        if step == 0:
            grad0 = first_w.grad.norm().item()
        if step == early_step:
            early_grad = first_w.grad.norm().item()
        losses.append(loss.item())
        opt.step()                             # parameter update
    if watch is not None and steps in watch:
        snapshots[steps] = model.fc1.weight.detach().clone()
    with torch.no_grad():
        final = loss_fn(model(X), targets).item()
    losses.append(final)
    return {"losses": losses, "initial": losses[0], "final": final,
            "grad0": grad0, "early_grad": early_grad, "snapshots": snapshots}


def binary_predictions(model):
    with torch.no_grad():
        p = torch.sigmoid(model(X))
    labels = (p >= 0.5).double()
    return p, labels, bool(torch.equal(labels, Y))


# ---------------------------------------------------------------------------
# Task 1: linear baseline
# ---------------------------------------------------------------------------
def task1_linear_baseline():
    section("TASK 1: linear baseline (single affine map + sigmoid output)")
    torch.manual_seed(SEED)
    lin = nn.Linear(2, 1)
    hist = train(lin, nn.BCEWithLogitsLoss(), Y)
    p, labels, ok = binary_predictions(lin)
    print(f"initial loss = {hist['initial']:.6f}   final loss = {hist['final']:.6f}"
          f"   (ln 2 = {np.log(2):.6f})")
    print("x1 x2 | y | p(y=1|x) | label")
    for i in range(4):
        print(f" {int(X[i,0])}  {int(X[i,1])} | {int(Y[i])} |  {p[i].item():.4f}  |  {int(labels[i])}")
    print(f"weights = {fmt(lin.weight.data, 6)}   bias = {fmt(lin.bias.data, 6)}")
    print(f"all four correct? {ok}   correct count = {int((labels == Y).sum())}/4")

    # Deep linear stack: more parameters, still only an affine function of x.
    torch.manual_seed(SEED)
    deep = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 8), nn.Linear(8, 1))
    n_params = sum(q.numel() for q in deep.parameters())
    dhist = train(deep, nn.BCEWithLogitsLoss(), Y)
    # Collapse the three layers into one affine map W x + b and compare outputs.
    W = deep[2].weight @ deep[1].weight @ deep[0].weight
    b = deep[2].weight @ (deep[1].weight @ deep[0].bias + deep[1].bias) + deep[2].bias
    with torch.no_grad():
        gap = (deep(X) - (X @ W.T + b)).abs().max().item()
    _, dl, dok = binary_predictions(deep)
    print()
    print(f"deep linear 2-8-8-1 (no activations, {n_params} parameters): "
          f"final loss = {dhist['final']:.6f}, all four correct? {dok}")
    print(f"collapsed single affine map W = {fmt(W.detach(), 6)}, b = {fmt(b.detach(), 6)}")
    print(f"max |deep(x) - (W x + b)| over the 4 inputs = {gap:.2e}")
    return {"linear": hist, "linear_probs": p, "deep": dhist, "deep_params": n_params,
            "collapse_gap": gap}


# ---------------------------------------------------------------------------
# Task 4A: basic learning check
# ---------------------------------------------------------------------------
def task4a_basic(n_seeds=20):
    section("TASK 4A: basic learning check (2-2-1, sigmoid hidden, BCEWithLogitsLoss, Adam)")
    torch.manual_seed(SEED)
    model = XORNet("sigmoid")
    hist = train(model, nn.BCEWithLogitsLoss(), Y)
    p, labels, ok = binary_predictions(model)
    print(f"seed = {SEED}, lr = {LR}, steps = {STEPS}")
    print(f"initial loss = {hist['initial']:.6f}   final loss = {hist['final']:.6f}")
    print("x1 x2 | y | p(y=1|x) | label")
    for i in range(4):
        print(f" {int(X[i,0])}  {int(X[i,1])} | {int(Y[i])} |  {p[i].item():.4f}  |  {int(labels[i])}")
    print(f"all four correct? {ok}")
    for k in [0, 100, 500, 1000, 2000, STEPS]:
        print(f"  loss at step {k:5d}: {hist['losses'][k]:.6f}")

    # Hidden representation: the learned features make XOR linearly separable.
    with torch.no_grad():
        _, a1, h1 = model(X, return_hidden=True)
    print("hidden representation h = sigmoid(W1 x + b1):")
    for i in range(4):
        print(f"  x = ({int(X[i,0])},{int(X[i,1])})  ->  h = {fmt(h1[i])}   y = {int(Y[i])}")
    print(f"W1 = {fmt(model.fc1.weight.data)}  b1 = {fmt(model.fc1.bias.data)}")
    print(f"W2 = {fmt(model.fc2.weight.data)}  b2 = {fmt(model.fc2.bias.data)}")

    # Repeated-run check
    print()
    print(f"repeated-run check over seeds 0..{n_seeds-1}:")
    seeds_ok, finals = [], []
    for s in range(n_seeds):
        torch.manual_seed(s)
        m = XORNet("sigmoid")
        h = train(m, nn.BCEWithLogitsLoss(), Y)
        seeds_ok.append(binary_predictions(m)[2])
        finals.append(h["final"])
    fails = [s for s, o in enumerate(seeds_ok) if not o]
    print(f"  solved {sum(seeds_ok)}/{n_seeds} seeds; failing seeds: {fails if fails else 'none'}")
    print(f"  final loss: median {np.median(finals):.4f}, max {np.max(finals):.4f}")
    return {"model": model, "hist": hist, "probs": p, "ok": ok, "h1": h1,
            "seeds_ok": sum(seeds_ok), "n_seeds": n_seeds, "fails": fails, "finals": finals}


# ---------------------------------------------------------------------------
# Task 4B: backpropagation check
# ---------------------------------------------------------------------------
def manual_grad_W1(model):
    """Hand-derived dL/dW1 for mean BCE with a sigmoid hidden layer (chain rule)."""
    W1, b1 = model.fc1.weight.detach(), model.fc1.bias.detach()
    W2, b2 = model.fc2.weight.detach(), model.fc2.bias.detach()
    N = X.shape[0]
    a1 = X @ W1.T + b1                  # (4,2)
    h1 = torch.sigmoid(a1)              # (4,2)
    z = h1 @ W2.T + b2                  # (4,1) logit
    p = torch.sigmoid(z)
    dz = (p - Y) / N                    # dL/dz for mean BCE-with-logits
    dh1 = dz @ W2                       # (4,2)
    da1 = dh1 * h1 * (1 - h1)           # sigmoid'(a) = s(a)(1 - s(a))
    return da1.T @ X                    # (2,2) = sum_n da1_n x_n^T


def task4b_backprop():
    section("TASK 4B: backpropagation check (first-layer gradient)")
    torch.manual_seed(SEED)
    model = XORNet("sigmoid")
    loss_fn = nn.BCEWithLogitsLoss()           # reduction='mean'
    model.zero_grad()
    loss = loss_fn(model(X), Y)
    loss.backward()
    g_auto = model.fc1.weight.grad.clone()
    print(f"loss at initialisation = {loss.item():.6f}")
    print(f"fc1.weight.grad (autograd) = dL/dW1 =\n{fmt(g_auto, 8)}")
    print(f"||dL/dW1||_2 = {g_auto.norm().item():.8f}")

    # 1. Hand-derived chain rule
    g_manual = manual_grad_W1(model)
    print(f"hand-derived chain rule dL/dW1 =\n{fmt(g_manual, 8)}")
    print(f"max |autograd - manual| = {(g_auto - g_manual).abs().max().item():.2e}")

    # 2. Central finite differences on each of the 4 entries of W1
    eps = 1e-6
    g_fd = torch.zeros_like(g_auto)
    with torch.no_grad():
        for i in range(2):
            for j in range(2):
                orig = model.fc1.weight[i, j].item()
                model.fc1.weight[i, j] = orig + eps
                lp = loss_fn(model(X), Y).item()
                model.fc1.weight[i, j] = orig - eps
                lm = loss_fn(model(X), Y).item()
                model.fc1.weight[i, j] = orig
                g_fd[i, j] = (lp - lm) / (2 * eps)
    print(f"finite differences dL/dW1 =\n{fmt(g_fd, 8)}")
    print(f"max |autograd - finite diff| = {(g_auto - g_fd).abs().max().item():.2e}")

    # 3. Mean loss => gradient is the average of per-example gradients
    per_example = []
    for n in range(4):
        model.zero_grad()
        loss_fn(model(X[n:n+1]), Y[n:n+1]).backward()
        per_example.append(model.fc1.weight.grad.clone())
    avg = torch.stack(per_example).mean(0)
    print("per-example gradients dL_n/dW1:")
    for n, g in enumerate(per_example):
        print(f"  x = ({int(X[n,0])},{int(X[n,1])}): {fmt(g.flatten(), 6)}")
    print(f"average of the four = {fmt(avg.flatten(), 8)}")
    print(f"max |mean-loss grad - average| = {(g_auto - avg).abs().max().item():.2e}")
    model.zero_grad()
    return {"g_auto": g_auto, "manual_err": (g_auto - g_manual).abs().max().item(),
            "fd_err": (g_auto - g_fd).abs().max().item(),
            "avg_err": (g_auto - avg).abs().max().item(), "per_example": per_example}


# ---------------------------------------------------------------------------
# Task 4C: symmetry experiment
# ---------------------------------------------------------------------------
def task4c_symmetry():
    section("TASK 4C: symmetry experiment (identical initial weights)")
    watch = [0, 1, 2, 3, 5, 10, 50, 100, 500, 1000, STEPS]
    results = {}
    for label, value in [("all zero", 0.0), ("all 0.5", 0.5)]:
        torch.manual_seed(SEED)
        m = XORNet("sigmoid")
        with torch.no_grad():
            for q in m.parameters():
                q.fill_(value)
        hist = train(m, nn.BCEWithLogitsLoss(), Y, watch=watch)
        p, labels, ok = binary_predictions(m)
        print(f"\n[{label}] every weight and bias initialised to {value}")
        print(" step | W1 row 1 (unit 1)        | W1 row 2 (unit 2)        | rows identical?")
        for k in watch:
            W = hist["snapshots"][k]
            same = torch.equal(W[0], W[1])
            print(f"{k:5d} | {fmt(W[0], 6):24s} | {fmt(W[1], 6):24s} | {same}")
        w2 = m.fc2.weight.data[0]
        print(f"final W2 = {fmt(w2, 6)} (output weights identical too: {bool(w2[0] == w2[1])})")
        print(f"gradient norm of W1 at step 0 = {hist['grad0']:.3e}")
        print(f"final loss = {hist['final']:.6f}; probs = {fmt(p.flatten())}; all four correct? {ok}")
        results[label] = {"hist": hist, "ok": ok, "probs": p}
    return results


# ---------------------------------------------------------------------------
# Task 4D: activation experiment
# ---------------------------------------------------------------------------
def task4d_activations(n_seeds=20):
    section("TASK 4D: activation experiment (same initial weights, only activation changes)")
    torch.manual_seed(SEED)
    base = XORNet("sigmoid")
    init_state = copy.deepcopy(base.state_dict())
    rows, curves = [], {}
    for name in ACTIVATIONS:
        m = XORNet(name)
        m.load_state_dict(init_state)          # identical starting weights
        hist = train(m, nn.BCEWithLogitsLoss(), Y)
        p, labels, ok = binary_predictions(m)
        with torch.no_grad():
            _, a1, h1 = m(X, return_hidden=True)
        rows.append((name, hist["final"], ok, hist["grad0"], hist["early_grad"], p, a1, h1))
        curves[name] = hist["losses"]
    print(f"seed {SEED}, lr {LR}, {STEPS} Adam steps, early step = {EARLY_STEP}")
    print(f"{'activation':10s} | {'final loss':>10s} | 4/4 correct? | ||grad W1|| step 0 | step {EARLY_STEP}")
    for name, fl, ok, g0, ge, *_ in rows:
        print(f"{name:10s} | {fl:10.6f} | {str(ok):12s} | {g0:17.6f} | {ge:.6f}")
    for name, fl, ok, g0, ge, p, a1, h1 in rows:
        print(f"\n{name}: probabilities {fmt(p.flatten())}")
        print(f"  final pre-activations a1 (rows = inputs):\n{fmt(a1)}")
        print(f"  final hidden outputs h1:\n{fmt(h1)}")

    # Diagnostic for the Think-About-It question: local derivative f'(a) at the
    # initial weights, which is what scales the backpropagated signal.
    print("\nlocal derivative f'(a1) at initialisation (rows = inputs, cols = hidden units):")
    with torch.no_grad():
        a0 = X @ init_state["fc1.weight"].T + init_state["fc1.bias"]
    derivs = {"sigmoid": torch.sigmoid(a0) * (1 - torch.sigmoid(a0)),
              "tanh": 1 - torch.tanh(a0) ** 2,
              "relu": (a0 > 0).double()}
    print(f"  pre-activations a1 =\n{fmt(a0)}")
    for k, v in derivs.items():
        print(f"  {k:7s}: {fmt(v.flatten())}")

    print(f"\nrepeated runs (seeds 0..{n_seeds-1}), solved count:")
    multi = {}
    for name in ACTIVATIONS:
        solved = 0
        dead = 0
        for s in range(n_seeds):
            torch.manual_seed(s)
            m = XORNet(name)
            train(m, nn.BCEWithLogitsLoss(), Y)
            solved += binary_predictions(m)[2]
            if name == "relu":
                with torch.no_grad():
                    a = m.fc1(X)
                dead += int(((a <= 0).all(0)).any())
        multi[name] = solved
        extra = f"  (runs ending with a dead ReLU unit: {dead})" if name == "relu" else ""
        print(f"  {name:7s}: {solved}/{n_seeds}{extra}")
    return {"rows": rows, "curves": curves, "multi": multi, "n_seeds": n_seeds, "init": init_state}


# ---------------------------------------------------------------------------
# Task 5: three-class extension
# ---------------------------------------------------------------------------
def task5_multiclass():
    section("TASK 5: three-class extension (softmax + cross-entropy)")
    torch.manual_seed(SEED)
    m = XORNet("tanh", hidden=2, n_out=3)
    print(f"final weight matrix fc2.weight shape = {tuple(m.fc2.weight.shape)} (predicted (3, 2))")
    with torch.no_grad():
        print(f"logits shape for the batch = {tuple(m(X).shape)} (3 logits per example)")
    hist = train(m, nn.CrossEntropyLoss(), Y3)
    with torch.no_grad():
        logits = m(X)
        P = torch.softmax(logits, dim=1)
    pred = P.argmax(1)
    print(f"initial loss = {hist['initial']:.6f} (ln 3 = {np.log(3):.6f})   final loss = {hist['final']:.6f}")
    print("x1 x2 | class | P(0)    P(1)    P(2)   | predicted")
    for i in range(4):
        print(f" {int(X[i,0])}  {int(X[i,1])} |   {int(Y3[i])}   | {P[i,0]:.4f}  {P[i,1]:.4f}  {P[i,2]:.4f} | {int(pred[i])}")
    ok = bool(torch.equal(pred, Y3))
    print(f"all four correct? {ok}")

    # Sum-to-one check on one example
    ex = 1
    print(f"\nexample x = (0,1): softmax vector = {fmt(P[ex], 8)}, sum = {P[ex].sum().item():.16f}")

    # p - y check: gradient of the loss wrt the logits
    z = logits.clone().requires_grad_(True)
    loss_sum = nn.CrossEntropyLoss(reduction="sum")(z, Y3)
    loss_sum.backward()
    onehot = torch.nn.functional.one_hot(Y3, 3).double()
    pmy = torch.softmax(z.detach(), 1) - onehot
    print("dL/dz from autograd (sum reduction) vs p - y:")
    for i in range(4):
        print(f"  x=({int(X[i,0])},{int(X[i,1])}): autograd {fmt(z.grad[i], 6)}   p - y {fmt(pmy[i], 6)}")
    pmy_err = (z.grad - pmy).abs().max().item()
    print(f"max |autograd - (p - y)| = {pmy_err:.2e}  (with mean reduction it is (p - y)/N)")

    # Optional diagnostic: shift invariance and stable softmax
    shifted = torch.softmax(logits[ex] + 100.0, dim=0)
    shift_err = (shifted - P[ex]).abs().max().item()
    print(f"\nsoftmax(z + 100) = {fmt(shifted, 8)}; max difference from softmax(z) = {shift_err:.2e}")
    big = logits[ex] + 1000.0
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(big.numpy())
        naive = e / e.sum()
    stable = np.exp(big.numpy() - big.numpy().max())
    stable = stable / stable.sum()
    print(f"naive softmax(z + 1000) = {naive}   <- exp overflows to inf, inf/inf = nan")
    print(f"stable softmax(z + 1000 - max) = {np.array2string(stable, precision=8)}")

    # The 3-class task is only a function of s = x1 + x2, so it is solvable
    # even without a hidden layer. Check that with a linear softmax model.
    torch.manual_seed(SEED)
    lin3 = nn.Linear(2, 3)
    lh = train(lin3, nn.CrossEntropyLoss(), Y3)
    with torch.no_grad():
        lok = bool(torch.equal(lin3(X).argmax(1), Y3))
    print(f"\ncomparison: linear softmax model (no hidden layer) final loss = {lh['final']:.4f}, "
          f"all four correct? {lok}")
    return {"P": P, "ok": ok, "hist": hist, "pmy_err": pmy_err, "shift_err": shift_err,
            "linear_ok": lok, "model": m}


def main():
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    with redirect_stdout(Tee()):
        print(f"PyTorch {torch.__version__}, NumPy {np.__version__}, "
              f"Python {sys.version.split()[0]}, CPU, float64")
        t1 = task1_linear_baseline()
        t4a = task4a_basic()
        t4b = task4b_backprop()
        t4c = task4c_symmetry()
        t4d = task4d_activations()
        t5 = task5_multiclass()

        section("SUMMARY OF CHECKS")
        checks = [
            ("linear baseline cannot solve XOR", not binary_predictions_linear_ok(t1)),
            ("deep linear stack equals one affine map (gap < 1e-8)", t1["collapse_gap"] < 1e-8),
            ("2-2-1 sigmoid net classifies all four", t4a["ok"]),
            ("final loss < 0.05", t4a["hist"]["final"] < 0.05),
            ("autograd matches hand chain rule (< 1e-10)", t4b["manual_err"] < 1e-10),
            ("autograd matches finite differences (< 1e-7)", t4b["fd_err"] < 1e-7),
            ("mean-loss grad = average of per-example grads", t4b["avg_err"] < 1e-12),
            ("zero init fails to learn XOR", not t4c["all zero"]["ok"]),
            ("3-class net classifies all four", t5["ok"]),
            ("logit gradient equals p - y (< 1e-12)", t5["pmy_err"] < 1e-12),
            ("softmax invariant to +100 shift (< 1e-12)", t5["shift_err"] < 1e-12),
        ]
        for name, passed in checks:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        all_ok = all(p for _, p in checks)
        print(f"\n{sum(p for _, p in checks)}/{len(checks)} checks passed")

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write(buf.getvalue())
    return 0 if all_ok else 1


def binary_predictions_linear_ok(t1):
    return bool(torch.equal((t1["linear_probs"] >= 0.5).double(), Y))


if __name__ == "__main__":
    sys.exit(main())
