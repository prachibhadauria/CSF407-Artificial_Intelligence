"""
Lab 8: Bayesian Networks and Autoregressive Language Models (CS F407)

An autoregressive language model is a Bayesian network whose nodes are the
words X1, X2, ..., XT of a sentence:

    first order  (bigram)  : X1 -> X2 -> X3 -> ...          P(Xt | Xt-1)
    second order (trigram) : Xt-2 -> Xt <- Xt-1              P(Xt | Xt-2, Xt-1)

Both models are one class, NGramLM(order), estimated by counting (maximum
likelihood).  Plain Python only: no ML library, no pretrained model.

Run:   python3 bn_language_model.py
It prints every part of the lab and writes, next to this file,
    lab_results.txt          (everything printed)
    first_order_samples.txt  (25 sampled sentences, first-order model)
    second_order_samples.txt (25 sampled sentences, second-order model)
"""

import io
import os
import random
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout

START, END = "<START>", "<END>"
SEED = 8
MAX_LEN = 30          # cap on generated words; needed because greedy can cycle
N_SAMPLES = 25

CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def tokenise(sentence):
    return sentence.lower().split()


# ---------------------------------------------------------------------------
# The model: P(X_t | previous `order` tokens), a conditional probability table
# ---------------------------------------------------------------------------
class NGramLM:
    def __init__(self, sentences, order):
        self.order = order
        # counts[context][word] = how often `word` followed `context`
        self.counts = defaultdict(Counter)
        for tokens in sentences:
            padded = [START] * order + tokens + [END]
            for i in range(order, len(padded)):
                self.counts[tuple(padded[i - order:i])][padded[i]] += 1
        # cpt[context][word] = counts / total for that context
        self.cpt = {}
        for ctx, nxt in self.counts.items():
            total = sum(nxt.values())
            self.cpt[ctx] = {w: c / total for w, c in nxt.items()}

    # -- the conditional distribution ---------------------------------------
    def context_of(self, history):
        return tuple(history[-self.order:])

    def distribution(self, history):
        """P(X_t | context); an empty dict if the context was never observed."""
        return self.cpt.get(self.context_of(history), {})

    def predict(self, history):
        """arg max of the distribution; ties go to the alphabetically first word."""
        dist = self.distribution(history)
        if not dist:
            return None
        return max(sorted(dist), key=dist.get)

    def sample(self, history, rng):
        dist = self.distribution(history)
        if not dist:
            return None
        words = sorted(dist)
        return rng.choices(words, weights=[dist[w] for w in words])[0]

    # -- generation -----------------------------------------------------------
    def generate(self, rng=None, greedy=False):
        """Return (words, reason it stopped)."""
        history = [START] * self.order
        words, seen_contexts = [], set()
        while len(words) < MAX_LEN:
            ctx = self.context_of(history)
            if greedy:
                if ctx in seen_contexts:
                    return words, "greedy path entered a cycle"
                seen_contexts.add(ctx)
                nxt = self.predict(history)
            else:
                nxt = self.sample(history, rng)
            if nxt is None:
                return words, "unseen context"
            if nxt == END:
                return words, "END"
            words.append(nxt)
            history.append(nxt)
        return words, f"hit MAX_LEN={MAX_LEN}"

    def sentence_probability(self, words):
        """Chain rule: product of the conditional probabilities, ending in <END>."""
        history, p = [START] * self.order, 1.0
        for w in words + [END]:
            p *= self.distribution(history).get(w, 0.0)
            history.append(w)
        return p

    # -- sizes ----------------------------------------------------------------
    def stats(self, vocab):
        # contexts that can really occur: START-padded prefixes, then vocab words
        possible = {(START,) * self.order}
        for k in range(1, self.order + 1):
            for tail in _products(vocab, k):
                possible.add((START,) * (self.order - k) + tail)
        outcomes = len(vocab) + 1                        # words + <END>
        nonzero = sum(len(d) for d in self.cpt.values())
        return {
            "possible contexts": len(possible),
            "full CPT cells": len(possible) * outcomes,
            "observed contexts": len(self.cpt),
            "unseen contexts": len(possible - set(self.cpt)),
            "non-zero entries": nonzero,
            "free parameters": nonzero - len(self.cpt),
        }


def _products(vocab, k):
    out = [()]
    for _ in range(k):
        out = [c + (w,) for c in out for w in vocab]
    return out


# ---------------------------------------------------------------------------
# Deliberately faulty builder, used only to show the normalisation test works
# ---------------------------------------------------------------------------
def buggy_first_order_cpt(sentences):
    """Bug: a +1 smoothing term is added to the denominator but not to the
    numerators, so every row loses probability mass."""
    counts = defaultdict(Counter)
    for tokens in sentences:
        padded = [START] + tokens + [END]
        for prev, nxt in zip(padded, padded[1:]):
            counts[prev][nxt] += 1
    return {c: {w: n / (sum(nx.values()) + 1) for w, n in nx.items()} for c, nx in counts.items()}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def banner(title):
    print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72)


def fmt(dist):
    return ", ".join(f"{w} {p:.4f}" for w, p in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))


def name(ctx):
    return ", ".join(ctx)


def max_normalisation_error(cpt):
    return max(abs(sum(d.values()) - 1.0) for d in cpt.values())


def all_sentences(model, history, p, out):
    """Enumerate every sentence the model can produce (finite only if acyclic)."""
    for w, q in model.distribution(history).items():
        if w == END:
            out.append((history[model.order:], p * q))
        else:
            all_sentences(model, history + [w], p * q, out)
    return out


def write_lines(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
def run(outdir):
    rng = random.Random(SEED)
    data = [tokenise(s) for s in CORPUS]
    vocab = sorted({w for s in data for w in s})
    lm1, lm2 = NGramLM(data, 1), NGramLM(data, 2)
    checks = []

    def check(label, ok):
        checks.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    banner("Part III: training data (lower-cased, one word per token)")
    for s in data:
        print(" ".join([START] + s + [END]))
    print(f"vocabulary ({len(vocab)} words): {vocab}")

    banner("Part IV / Question 3: first-order CPT  P(next | current)")
    for w in [START, "the", "cat", "dog", "sat", "ran", "on", "to", "mat", "rug", "park"]:
        ctx = (w,)
        zero = [v for v in vocab + [END] if v not in lm1.cpt[ctx]]
        print(f"P(. | {w}) = {{{fmt(lm1.cpt[ctx])}}}   counts {dict(lm1.counts[ctx])}")
        print(f"    zero-probability next words: {zero}")
    print(f"\n<END> is never a context, so P(. | {END}) is not defined.")

    banner("Part VII / Question 8: normalisation test, sum over v of P(v | w)")
    for ctx, d in lm1.cpt.items():
        print(f"  {name(ctx):8s} {sum(d.values()):.12f}")
    e1, e2 = max_normalisation_error(lm1.cpt), max_normalisation_error(lm2.cpt)
    print(f"  worst |total - 1|: first-order {e1:.2e}, second-order {e2:.2e}")
    check("first-order CPT rows sum to 1", e1 < 1e-12)
    check("second-order CPT rows sum to 1", e2 < 1e-12)

    bad = buggy_first_order_cpt(data)
    bad = buggy_first_order_cpt(data)
    print("\n  Faulty builder (+1 added to the denominator only), same test:")
    for ctx in ["the", "cat", "sat", "mat"]:
        print(f"    {ctx:5s} total = {sum(bad[ctx].values()):.4f}")
    worst_bad = max(abs(sum(d.values()) - 1) for d in bad.values())
    print(f"  worst |total - 1| = {worst_bad:.4f}")
    check("normalisation test detects the faulty builder", worst_bad > 0.1)

    banner("Independent checks that the tables mean what they should")
    recount = Counter()
    for s in data:
        p = [START] + s + [END]
        recount.update(zip(p, p[1:]))
    check("first-order counts equal an independent bigram recount",
          all(lm1.counts[(a,)][b] == c for (a, b), c in recount.items())
          and sum(recount.values()) == sum(sum(c.values()) for c in lm1.counts.values()))
    recount3 = Counter()
    for s in data:
        p = [START, START] + s + [END]
        recount3.update(zip(p, p[1:], p[2:]))
    check("second-order counts equal an independent trigram recount",
          all(lm2.counts[(a, b)][c] == n for (a, b, c), n in recount3.items())
          and sum(recount3.values()) == sum(sum(c.values()) for c in lm2.counts.values()))
    check("P(cat | the) = 3/12 and P(sat | cat) = 2/3 (hand-computed)",
          abs(lm1.cpt[("the",)]["cat"] - 3 / 12) < 1e-12 and abs(lm1.cpt[("cat",)]["sat"] - 2 / 3) < 1e-12)
    n = 200_000
    drawn = Counter(lm1.sample(["the"], rng) for _ in range(n))
    print(f"\n  sampler check, {n} draws after 'the':")
    worst = 0.0
    for w, p in sorted(lm1.cpt[("the",)].items()):
        worst = max(worst, abs(drawn[w] / n - p))
        print(f"    {w:5s} CPT {p:.4f}  empirical {drawn[w] / n:.4f}")
    check("sampler frequencies match the CPT within 0.005", worst < 0.005)
    support = all_sentences(lm2, [START, START], 1.0, [])
    total = sum(p for _, p in support)
    check(f"second-order sentence probabilities sum to 1 ({len(support)} sentences, total {total:.12f})",
          abs(total - 1) < 1e-12)
    w = tokenise("the cat sat on the mat")
    manual = (lm1.cpt[(START,)]["the"] * lm1.cpt[("the",)]["cat"] * lm1.cpt[("cat",)]["sat"]
              * lm1.cpt[("sat",)]["on"] * lm1.cpt[("on",)]["the"] * lm1.cpt[("the",)]["mat"]
              * lm1.cpt[("mat",)][END])
    check("sentence probability equals the product of the CPT entries (chain rule)",
          abs(lm1.sentence_probability(w) - manual) < 1e-15)

    banner("Part VIII / Question 9: next-word prediction, first-order")
    for w in [START, "the", "cat", "dog", "sat", "ran", "on", "to"]:
        d = lm1.distribution([w])
        top = max(d.values())
        tied = sorted(v for v, p in d.items() if p == top)
        note = f"   TIE {tied}, alphabetical pick" if len(tied) > 1 else ""
        print(f"  P(. | {w:7s}) = {{{fmt(d)}}}")
        print(f"      arg max = {lm1.predict([w])} (p = {top:.4f}){note}")
    print(f"  unseen word: P(. | elephant) = {lm1.distribution(['elephant'])}, predict -> {lm1.predict(['elephant'])}")
    check("unseen context gives an empty distribution and no prediction",
          lm1.distribution(["elephant"]) == {} and lm1.predict(["elephant"]) is None)

    banner(f"Part IX: {N_SAMPLES} sampled sentences, first-order model")
    first = []
    for i in range(N_SAMPLES):
        ws, why = lm1.generate(rng)
        first.append(" ".join(ws))
        print(f"  {i + 1:2d}. {' '.join(ws)}   [{why}]")
    write_lines(os.path.join(outdir, "first_order_samples.txt"), first)

    banner("Part X / Question 10: greedy vs sampling, first-order")
    print("  Mode A (greedy):")
    greedy = [lm1.generate(greedy=True) for _ in range(5)]
    for i, (ws, why) in enumerate(greedy, 1):
        print(f"    {i}. {' '.join(ws)}   [{why}]")
    print("  Mode B (sampling):")
    mode_b = [lm1.generate(rng) for _ in range(5)]
    for i, (ws, why) in enumerate(mode_b, 1):
        print(f"    {i}. {' '.join(ws)}   [{why}]")
    check("greedy gives 5 identical sentences (deterministic)", len({tuple(g[0]) for g in greedy}) == 1)
    check("greedy first-order never reaches <END>", all(g[1] != "END" for g in greedy))
    print(f"  distinct sentences: greedy {len({tuple(g[0]) for g in greedy})}, sampling {len({tuple(g[0]) for g in mode_b})}")

    banner("Part XI / Question 11: second-order CPT  P(next | two previous words)")
    for ctx in sorted(lm2.cpt):
        print(f"  P(. | {name(ctx)}) = {{{fmt(lm2.cpt[ctx])}}}")
    print(f"  unseen pair: P(. | the, sat) = {lm2.distribution(['the', 'sat'])}")

    banner(f"Part XII: {N_SAMPLES} sampled sentences, second-order model")
    second = []
    for i in range(N_SAMPLES):
        ws, why = lm2.generate(rng)
        second.append(" ".join(ws))
        print(f"  {i + 1:2d}. {' '.join(ws)}   [{why}]")
    write_lines(os.path.join(outdir, "second_order_samples.txt"), second)
    g2, why2 = lm2.generate(greedy=True)
    print(f"  second-order greedy: {' '.join(g2)}   [{why2}]")
    check("second-order greedy terminates at <END>", why2 == "END")
    print("  all sentences the second-order model can produce:")
    for ws, p in sorted(support, key=lambda x: (-x[1], x[0])):
        print(f"    {p:.4f}  {' '.join(ws)}")

    banner("Part XIII / Question 12: comparing the two models")
    train = set(CORPUS)
    rows = []
    for label, m, samples in [("first-order", lm1, first), ("second-order", lm2, second)]:
        st = m.stats(vocab)
        rows.append((label, st, len(set(samples)), len(set(samples) - train)))
    keys = ["possible contexts", "full CPT cells", "observed contexts", "unseen contexts",
            "non-zero entries", "free parameters"]
    print("| measure | first-order | second-order |")
    print("|---|---|---|")
    for k in keys:
        print(f"| {k} | {rows[0][1][k]} | {rows[1][1][k]} |")
    print(f"| distinct sentences in {N_SAMPLES} samples | {rows[0][2]} | {rows[1][2]} |")
    print(f"| novel sentences (not in training data) | {rows[0][3]} | {rows[1][3]} |")
    print("\n  sentence probabilities:")
    for s in ["the cat sat on the mat", "the dog ran to the park", "the cat sat on the park",
              "the dog ran to the mat", "the cat sat on the cat sat on the mat"]:
        t = tokenise(s)
        print(f"    {s:40s} first {lm1.sentence_probability(t):.6f}   second {lm2.sentence_probability(t):.6f}")
    check("second-order has more unseen contexts than first-order",
          rows[1][1]["unseen contexts"] > rows[0][1]["unseen contexts"])

    banner("Summary")
    print(f"{sum(checks)}/{len(checks)} checks passed (seed {SEED})")
    return all(checks)


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = run(outdir)
    text = buf.getvalue()
    with open(os.path.join(outdir, "lab_results.txt"), "w") as f:
        f.write(text)
    sys.stdout.write(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
