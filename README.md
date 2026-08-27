# King of the Hill: performant A/B/n testing beyond the basics

[![PyPI](https://img.shields.io/pypi/v/koth)](https://pypi.org/project/koth/)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

King of the hill: how to allocate among the arms of an A/B/n test with correlated arms,
decided on the top-k contenders (k = 2, 3).

```
pip install koth            # install from PyPI (numpy implementation)
pip install koth[torch]     # if you want to use Torch
pip install koth[arena]     # plus the benchmark arena
```

```python
import numpy as np
import koth

# A control and two treatments, one epoch = one day. What each arm measured
# per day, and the share of the allocation it had that day: five days at an
# even split (synthetic here; yours come from your data).
rng = np.random.default_rng(5)
allocations = np.full((5, 3), 1 / 3)
outcomes = np.array([0.0, 3.0, 2.0]) + rng.normal(size=(5, 3)) * 30.0 / np.sqrt(allocations)

test = koth.Test(rho=0.01, sigma=30.0)  # 1/rho = 100 days; sigma = one day's noise at full allocation
initial = koth.State.flat(arms=3, std=100.0)    # initial state: "uninformed 
state = test.observe(initial, outcomes, allocations)
decision = test.decide(state)           # k = 2 or 3, min(3, arms) by default

decision.allocation   # [0.24 0.51 0.25]: share per arm; a vertex is a commit
decision.committed    # -1: no commit yet (else the arm's index)
decision.contenders   # [0 1 2]: the k arms in play
decision.value        # 2044.0: value of continuing, in the units of the outcomes
```


## Why this package is so cool

### What is even the problem?

A/B testing is hard because of the [exploration-exploitation dilemma](https://en.wikipedia.org/wiki/Exploration%E2%80%93exploitation_dilemma): every
observation you spend on an arm in order to learn about it is one you did not
spend on the arm you currently believe is best. It is about the money, not
about being right, so hypothesis testing, the naive thing people reach for, is
not the right tool: a p-value says whether an effect is there, not how much of
the allocation each arm deserves.

Targeted solutions do exist: Thompson sampling is a heuristic that works (easy, generalizable, decent performance), but it explores too much. If your arms are _independent_, the Gittins index is known to be the optimal strategy. However that's not always the case:

- Shared control: every treatment is measured against the same baseline. Learning that B is better than A teaches you something about C vs A (maybe B is better because A is just bad to begin with).
- Optimizing over a continuum: bidding 10 cents and 11 cents should yield very similar results.
- Matrix designs: explosive combinations of variations, but few underlying factors.


### What this package does

I came up with a heuristic that can be very effective for the general correlated case. It is based on the fact that I have [a numeric solution](https://github.com/tokahuke/pinn) for two- and three-armed gaussian bandits. Here is the idea: 

> Suppose I have an n-armed correlated bandit case. Among all combinations of size `k`, choose the one that looks most promising at a given point in time. Play according to that allocation, ignoring all other arms.

That's it! This works because:

1. The numeric solutions can tell me "how promising" a combination is (aka, the [_value function_](https://en.wikipedia.org/wiki/Value_function)).
2. Ignoring arms is a sub-solution of the problem (it's strictly _worse_ than or equal the best strategy) and a _maximum_ of subsolutions is also a subsolution.

### But how good is this really?

Good question. We have an _arena_ for exactly that: backtesting strategies. Let's consider a simple example, say 12 _independent_ arms. The _best_ solution is known: Gittins Index\*. Here are the other contestants:
* Thompson sampling (proportional allocation): allocate proportional to the probability of that arm being the best.
* Z-test, sequential: drop arms when it's significantly dominated by another arm (with p < 5%).
* King-of-the-hill based on exact 2 and 3 problems: our heuristic.

"Good" here means time-discounted regret: how much less "money" I make with my strategy vs. a crystal ball. Nobody beats a crystal ball, but we can get close. Here are the numbers for some reasonable parameters:

![12 independent arms, 2000 tests: regret and allocation on losing arms per strategy](https://raw.githubusercontent.com/tokahuke/koth/main/resources/arena_independent12.png)

Every strategy plays the same 2000 random tests: 12 arms, true effects drawn
from `N(0, 0.5)`, noise `sigma = 1` per epoch, `gamma = 0.99` (so `1/rho = 100`
epochs) over 500 epochs, and nobody is told the effect distribution. The spec
is [`resources/arena_independent12.spec.yaml`](https://github.com/tokahuke/koth/blob/main/resources/arena_independent12.spec.yaml);
to reproduce it (seeds are fixed, ~11 min on 4 cores):

```
pip install koth[arena]
koth-arena simulate --spec resources/arena_independent12.spec.yaml
koth-arena plot data/independent12.pkl --out resources/arena_independent12.png
```

Some notes:
* If Gittins is "optimal", why does it still lose? Well, Gittins is not optimal for this _simulation_ because of the _prior_. In this simulation, we don't tell strategies the range of effects we are sampling from. They have to start from somewhere "flat". This is more realistic: in "real life", the range is but a well-informed guess.
* The numbers on the right are just a measure of of how much deliberate exploration each strategy took. This answers the "how much time?" question; the "money question" is still answered solely by regret.


## More analyses

* [Misjudging the noise](docs/misspecified_sigma.md): what a wrong `sigma` costs, for koth and for the baselines.

## Frequently asked questions

* Does this test tell me when to stop testing? *No*, emphatically. KotH will eventually stop, but it's out of its own convenience. To know when a test should be stopped, one needs to know the _opportunity cost of stopping_ (testing is always beneficial, even if 0.00001% beneficial. If you have _no external reason_ to stop a test, why stop _ever_?). This is not modeled here and it would be downright dishonest to imply that it could tell you when to stop without taking opportunity cost into consideration. BTW, do you even know _yours_?

* What the heck is `rho`? That is your _time preference_: how fast you want to get results. It is your answer to ["one marshmallow today or two tomorrow?"](https://en.wikipedia.org/wiki/Stanford_marshmallow_experiment). If you want results _right now_, you must live with the probability of being (most likely) wrong. You cannot want results "in the long run", because in the long run you will be dead. The _inverse_ of `rho` is measured in time units (if your data comes every day, it's in days, in hours, it's in hours, etc...) and is the timescale by which one marshmallow then is worth ~36% of a marshmallow now.

* What about drift and parameter changes? You don't model those! They make the computed base models way more complex to train reliably. In addition, they are a "second order" problem. A simple solution you can make is to cap your input data to a reasonable horizon (i.e., recalibrate often). This should give you 80% of the real deal.

* Where do `mean` and `cov` come from? From _you_: they are your posterior over the arms' _effects_ (control included), in whatever units your metric has (koth never sees your raw data). For the common shared-control case, that is just one mean and one variance per arm, `cov = diag(var)`: the correlation between _lifts_ (every lift shares the control's noise) is derived by koth from the control's variance, you do not enter it. If you have a prior from past tests (empirical Bayes, correlated or not), `cov` is exactly where it goes.

* What is `sigma`? And my arms have different noise levels! `sigma` is the noise of one arm's estimate over one epoch at full allocation, and it is the _same_ for every arm. That is not laziness: the inner maximization is a quadratic only because `sigma**2` factors out of the observation covariance. Conversion rates are within a few percent of this (`p(1 - p)` barely moves between arms). Revenue-vs-conversion arms are a different problem, not a parameter.

* My metric is a conversion rate, not Gaussian! The Gaussian is on your _posterior of the rate_, not on the clicks. After a few conversions per arm, that is a fine, if crude approximation. Below that, for sparse data, this indeed might not be the package for you.

* How many arms can I throw at it? Which `k`? Each decision evaluates every `k`-subset: 12 arms is 66 pairs or 220 triples, 50 arms is 1,225 pairs or 19,600 triples. Koth3 is _cubic_ in arms while Koth2 is _quadratic_ (Koth4 would be _quartic_, yikes!). To add insult to injury, the underlying network for Koth3 is also way bigger than Koth2. Depending on what you are doing, Koth2 might just beat the tradeoff by quite a margin.

* Where is the math? In the [pinn](https://github.com/tokahuke/pinn) repo: `kb/` holds the derivations, and the graveyard of what did not work.
