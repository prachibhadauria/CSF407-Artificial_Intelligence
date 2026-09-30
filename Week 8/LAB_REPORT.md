# Lab 8: Bayesian Networks and Autoregressive Language Models

CS F407 Artificial Intelligence

Every number below is printed in `lab_results.txt`, produced by `python3 bn_language_model.py` (plain Python, seed 8, 14/14 checks pass). The two sets of generated sentences are in `first_order_samples.txt` and `second_order_samples.txt`. One class, `NGramLM(order)`, implements both models: `order=1` is the first-order model and `order=2` is the second-order model.

## Part I

**Question 1. Why is the chain-rule decomposition useful for generating text?**
P(x1,...,xT) = P(x1) P(x2|x1) ... P(xT|x1..xT-1) is exact, and it turns the impossible task of modelling a whole sentence at once into a sequence of next-word problems. To generate text we only need, at each step, one distribution over the next word given what has been written so far. Generation is then a loop: sample X1, then X2 given X1, and so on until `<END>`. The same factors also give the probability of any finished sentence, as their product.

## Part II

**Question 2. What independence assumption does X1 -> X2 -> X3 -> X4 make?**
Each word depends only on the word just before it:

P(Xt | X1, ..., Xt-1) = P(Xt | Xt-1), equivalently Xt is independent of {X1, ..., Xt-2} given Xt-1.

So P(X1,X2,X3,X4) = P(X1) P(X2|X1) P(X3|X2) P(X4|X3). This is the first-order Markov assumption.

## Part III: dataset

The six sentences of the handout, lower-cased, one word per token, wrapped as `<START> ... <END>`. The vocabulary has 10 words: cat, dog, mat, on, park, ran, rug, sat, the, to.

## Part IV

**Question 3. P(next word | current word)**, from C(wi,wj) / sum_k C(wi,wk):

| current | P(next \| current) | counts |
|---|---|---|
| `<START>` | the 1.0 | the 6 |
| the | cat 0.25, dog 0.25, mat 0.1667, park 0.1667, rug 0.1667 | cat 3, dog 3, mat 2, park 2, rug 2 |
| cat | sat 0.6667, ran 0.3333 | sat 2, ran 1 |
| dog | sat 0.6667, ran 0.3333 | sat 2, ran 1 |
| sat | on 1.0 | on 4 |
| ran | to 1.0 | to 2 |
| on | the 1.0 | the 4 |
| to | the 1.0 | the 2 |
| mat, rug, park | `<END>` 1.0 | 2 each |

"the" is the current word 12 times (6 at the start of a sentence, 6 after "on" or "to"), so P(cat | the) = 3/12, not the 3/5 of the handout's illustration.

Zero-probability transitions: after "the", the words on, ran, sat, the, to and `<END>`; after "cat" and "dog", everything except sat and ran (for example P(on | cat) = 0 and P(`<END>` | cat) = 0); after "sat" everything except "on"; after "ran" everything except "to"; after "on", "to" and `<START>` everything except "the"; after "mat", "rug" and "park" everything except `<END>`. Of the 11 x 11 = 121 (context, next) cells, only 17 are non-zero.

## Part V: prompt given to the LLM

> Write a simple Python implementation of a first-order autoregressive language model. The model should: (1) take a list of tokenised sentences as training data; (2) count transitions between consecutive tokens; (3) construct the conditional distribution P(Xt|Xt-1); (4) display the probabilities for a specified previous token; (5) predict the most probable next token; (6) generate a sentence by repeatedly sampling the next token; (7) stop when the `<END>` token is generated. Do not use a machine-learning library or a pretrained language model. Use ordinary Python data structures and random sampling. Add `<START>` and `<END>` markers, support greedy and sampling modes, and use a seeded `random.Random`.

## Part VI: inspecting the code (`bn_language_model.py`)

**Question 4. Where are the transition counts stored?**
In `NGramLM.__init__`, in `self.counts`, a `defaultdict(Counter)`. `self.counts[context][word]` is how often `word` followed `context`; for the first-order model the context is a one-word tuple such as `("the",)`.

**Question 5. Where is P(Xt|Xt-1) computed?**
Right after counting, in the same `__init__`: `self.cpt[ctx] = {w: c / total ...}` with `total = sum(nxt.values())`. `distribution(history)` looks up the row for the last `order` tokens.

**Question 6. How does the program choose the next word?**
It supports both. `predict()` always returns the arg max (ties go to the alphabetically first word). `sample()` draws from the row with `rng.choices`. The greedy choice is a deterministic function of the context, so it can only ever produce one sentence from a given start. Sampling picks each word with probability proportional to the table, so less likely continuations also appear and runs differ. I checked that the sampler follows the table: over 200,000 draws after "the", the frequencies were cat 0.2523, dog 0.2506, mat 0.1663, park 0.1651, rug 0.1658 against 0.25, 0.25, 0.1667, 0.1667, 0.1667.

**Question 7. What happens with a context that was never observed?**
`self.cpt.get(context, {})` returns an empty distribution, `predict` and `sample` return `None`, and `generate` stops with the reason "unseen context" (for example `P(. | elephant) = {}` and `predict -> None`). Without that guard, `rng.choices([], weights=[])` raises an error. This is the zero-frequency problem of maximum-likelihood n-grams: nothing is known about a context that was not in the training data. Real systems add smoothing or back off to a shorter context.

## Part VII: normalisation test

For every context w the program prints the sum over v of P(v|w). All eleven first-order rows total 1.000000000000 (worst deviation 1.1e-16), and the second-order table has worst deviation 0.

**Question 8. What would a total of 0.87 mean?**
That the row is not a probability distribution: 13% of the probability mass has been lost, so the implementation has a bug. To show the test catches this, `buggy_first_order_cpt` in the program adds 1 to the denominator only (a smoothing term that does not reach the numerators). The same test then reports totals of 0.9231 for "the", 0.7500 for "cat", 0.8000 for "sat" and 0.6667 for "mat" (worst deviation 0.3333), and the check "normalisation test detects the faulty builder" passes. Other causes of a total like 0.87 are a denominator that counts events the numerators do not, dropped transitions, or rounded probabilities. Sampling from such a table would be quietly biased.

Further checks, all passing: the counts equal an independent bigram and trigram recount of the raw sentences; P(cat|the) = 3/12 and P(sat|cat) = 2/3 match hand calculation; the sentence probability equals the product of the table entries; and the probabilities of all sentences the second-order model can produce sum to 1.000000000000.

## Part VIII: next-word prediction

| context | distribution | arg max |
|---|---|---|
| `<START>` | the 1.0 | the |
| the | cat .25, dog .25, mat .1667, park .1667, rug .1667 | cat (tie with dog, alphabetical pick) |
| cat | sat .6667, ran .3333 | sat |
| dog | sat .6667, ran .3333 | sat |
| sat | on 1.0 | on |
| ran | to 1.0 | to |
| on | the 1.0 | the |
| to | the 1.0 | the |

**Question 9. Do the predictions match what I would expect?**
Mostly, but not after "the". The table gives cat, dog, mat, park and rug similar probabilities, whereas a person expects an animal after a sentence-initial "the" and a place after "on the". The first-order model cannot tell those two uses apart because it sees only one word of context. It also produces an exact tie between cat and dog, which means nothing linguistically. A probability model reports co-occurrence frequencies in its training data under its independence assumptions. Human expectations come from meaning, grammar and much more context, so the model is right relative to its data, not relative to language.

## Part IX: generated text (first-order)

25 sampled sentences are in `first_order_samples.txt`. A few:

```
the dog sat on the rug
the cat ran to the mat
the mat
the dog sat on the dog sat on the dog sat on the dog ran to the rug
```

17 of the 25 are distinct and 13 are not in the training data. Ten of the 25 are two-word sentences such as "the mat" or "the park": after "the" the model goes to `<END>` through mat, rug or park with total probability 0.5. Some sentences loop ("the dog sat on the dog sat on ..."), and some pair an animal with the wrong place, like "the cat ran to the mat".

## Part X: greedy and sampling

Mode A (greedy), five runs, all identical: `the cat sat on the`, after which the path returns to the context "the" and the generator stops with "greedy path entered a cycle". Mode B (sampling), five runs: `the park`, `the rug`, `the mat`, `the dog sat on the cat sat on the cat sat on the mat`, `the rug`.

**Question 10. Which mode varies more, and why?**
Sampling: it gave 4 distinct sentences in 5 runs against 1 for greedy. Greedy picks the arg max at every step, so from a fixed start it follows one fixed path every time. Here that path is the -> cat -> sat -> on -> the -> cat ..., a cycle, because mat, rug and park (the only routes to `<END>`) each have lower probability after "the" than cat or dog. So greedy decoding in this model never terminates by itself, which is why the generator needs a cycle check and a length cap. Sampling draws from the whole distribution, so at each "the" it has a 0.5 chance of choosing a word that ends the sentence.

## Parts XI and XII: second-order model

**Question 11. How does the second-order model differ?**

1. **Graph:** each Xt has two parents, Xt-2 -> Xt <- Xt-1, instead of one. The independence assumption is weaker: Xt is independent of X1..Xt-3 given Xt-2 and Xt-1.
2. **CPT:** it is indexed by a pair of previous words. There are 111 possible contexts (one double-`<START>`, 10 of the form `<START>`, word, and 100 word pairs) instead of 11, so the full table has 1221 cells instead of 121.
3. **Context:** two words instead of one. The model can now separate "`<START>`, the" (next: cat or dog) from "on, the" (next: mat or rug) and "to, the" (next: park).
4. **Data:** much more. Each of the extra rows has to be estimated from the occurrences of that exact pair; here 96 of the 111 possible contexts never occur in the data.

**Prompt for Part XII:**

> Modify the existing first-order autoregressive model into a second-order model. It should estimate P(Xt|Xt-2,Xt-1), represented by counts of observed triples turned into conditional distributions. Pad each sentence with two `<START>` tokens. Keep the same interface. Do not replace the model with a neural network or a pretrained language model.

**What should change, stated before accepting the code:** only the context. Counts become `counts[(w_t-2, w_t-1)][w_t]`, the padding becomes two `<START>` tokens, and the context is the last two tokens. Normalisation, sampling, arg max and stopping should stay the same. In the program that is exactly the `order` parameter of `NGramLM`: `order=2` changes the padding and the context tuple length and nothing else. The second-order table (15 observed contexts) is printed in `lab_results.txt`; for example (`<START>`,the) gives cat 0.5 and dog 0.5, (on,the) gives mat 0.5 and rug 0.5, (to,the) gives park 1.0, and (the,cat) gives sat 0.6667 and ran 0.3333. The pair (the,sat) was never observed, so it has no distribution.

The 25 sampled sentences are in `second_order_samples.txt`, for example `the dog ran to the park`, `the cat sat on the rug`, `the dog sat on the mat`. Greedy second-order generation gives `the cat sat on the mat` and ends at `<END>`.

## Part XIII: comparing the models

| measure | first-order | second-order |
|---|---|---|
| possible contexts | 11 | 111 |
| full CPT cells | 121 | 1221 |
| observed contexts | 11 | 15 |
| **unseen (zero-information) contexts** | **0** | **96** |
| non-zero entries | 17 | 19 |
| free parameters (non-zero entries minus one per observed context) | 6 | 4 |
| distinct sentences in 25 samples | 17 | 6 |
| novel sentences (not in the training data) | 13 | 0 |

| sentence | first-order | second-order |
|---|---|---|
| the cat sat on the mat | 0.0278 | 0.1667 |
| the dog ran to the park | 0.0139 | 0.1667 |
| the cat sat on the park | 0.0278 | 0 |
| the dog ran to the mat | 0.0139 | 0 |
| the cat sat on the cat sat on the mat | 0.0046 | 0 |

Coherence: every second-order sentence is grammatical and sensible, but only because that model can produce nothing except the six training sentences (exactly six, each with probability 1/6). The first-order model is inventive but incoherent: it gives the odd "the cat sat on the park" the same probability as the real "the cat sat on the mat".

**Question 12. Why can more context help, and why does it make estimation harder?**
More context lets the model condition on information that actually determines the next word: after "on the" comes mat or rug, after "`<START>` the" comes cat or dog. That sharpens the distributions and removes absurd continuations, as the sentence probabilities show (0.1667 against 0.0278 for a real sentence, 0 against 0.0278 for an odd one). But the table grows as |V|^k with k words of context (121 cells, then 1221, and a third-order table would have 11^3 rows of 11 outcomes) while the amount of data stays fixed. Most rows then come from one or two examples or none at all (96 of 111 contexts unseen), so maximum-likelihood estimates overfit: here the second-order model memorised the corpus and gives probability 0 to anything new. More context lowers bias and raises variance.

## Part XIV: connection to modern language models

A neural language model keeps the same factorisation and objective, P(x1..xT) = product of P(xt | x1..xt-1). What changes is how the conditional distribution is represented: instead of a table looked up by the last k words, a network maps the whole preceding text to a softmax over the vocabulary, with parameters shared across contexts so it does not need a separate row for each one. It is learned by gradient-based training rather than counting, and generation is still sampling from the conditional distribution.

## Part XV

**Question 13. Why is Approach B better than "write me a language model"?**

- **Specifying the intended behaviour:** B names the model to build (P(Xt|Xt-1) from transition counts, with sampling-based generation), so "correct" is defined before any code exists. Approach A lets the LLM choose the model, which could be a bigram, a transformer call or anything else.
- **Understanding the representation:** with B I know the model is a table of normalised counts, and I could work out P(cat|the) = 3/12 by hand before running anything.
- **Validating the implementation:** a specification has testable consequences: counts must match a recount, rows must sum to 1, samples must follow the table, and sentence probabilities must equal products of table entries. With A there is nothing to compare the code against.
- **Testing probabilistic invariants:** row sums and "all sentence probabilities sum to 1" are properties of the model, so they catch errors that merely running the program would not show. The faulty builder in the program (a denominator that is too big) still produces plausible-looking text, but its row totals are 0.92, 0.75, 0.80 and 0.67.
- **Distinguishing implementation from model:** B keeps the mathematical object (the network and its tables) separate from the code. The LLM becomes a translator from a known model into Python, and that is what let the first-order model become the second-order model by changing only the context.

## Question 14. What did thinking of the language model as a Bayesian network add?

- **A representation of dependencies:** the graph shows what each word may depend on. Adding the edge Xt-2 -> Xt is the whole difference between the two models.
- **A factorisation of the joint:** the probability of a sentence is the product of local table entries. This gave the sentence probabilities above and the check that all second-order sentence probabilities sum to 1.
- **A principled method for generation:** ancestral sampling (sample each node given its already sampled parents, in order) is autoregressive text generation.
- **A way to reason about independence assumptions:** the first-order model's confusion between sentence-initial "the" and "on the" follows directly from Xt being independent of Xt-2 given Xt-1.
- **A way to understand the effect of more context:** more parents means a table that grows exponentially, which accounts for the 96 unseen contexts and the memorisation.
- **A way to test the implementation:** each column of a CPT is a distribution and each entry is a ratio of counts, so the normalisation, recount and sampler tests follow directly from the network's semantics.

## Reflection on how the LLM was used

I gave the LLM the behavioural specification above rather than "write a language model", then read the code against the specification and tested it.

What I checked and how: I worked out P(cat|the) = 3/12 and P(sat|cat) = 2/3 by hand first; ran the normalisation test on both models; recounted bigrams and trigrams independently; compared 200,000 sampled words with the table; and checked that the second-order sentence probabilities sum to 1.

Code I inspected and corrected:

1. **The stopping rule.** The specification says to stop at `<END>`, which does not guarantee termination. Tracing greedy decoding by hand gives the cycle the -> cat -> sat -> on -> the, which never reaches `<END>`, so literal greedy generation would run forever. I added a `MAX_LEN` cap and a cycle check that stops the loop and reports why; all five greedy runs stop that way.
2. **Unseen contexts.** `rng.choices` on an empty distribution raises an error. `distribution` now returns an empty dict and `generate` stops with "unseen context".
3. **Ties.** `max` over a dictionary breaks the cat/dog tie by insertion order, which depends on the order of the data. The tie-break is now explicit (alphabetical), so greedy output is reproducible.
4. **My own test of the test.** My first faulty builder dropped the last pair of every sentence. The normalisation check did not flag it: the rows for mat, rug and park disappeared entirely and all remaining rows still summed to 1, so the check reported no problem even though the table was wrong. I replaced it with a bug that really loses mass (+1 in the denominator only) and added a check that the test flags it. This showed that the row-sum test catches rows that are too small but not rows that are missing, which is why the recount check sits beside it.
