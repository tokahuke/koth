# Robustness

The nets were trained on one model of the world: Gaussian noise of a known
scale, effects that stay put. These studies run KotH and the baselines in
worlds that break one assumption at a time and read the regret. The setup is
the README's throughout (6 independent arms, effects `N(0, 0.5)`, `sigma = 1`,
`gamma = 0.999` over 3000 epochs, flat priors, 1000 tests per point, paired
seeds), with one thing changed.

* [Misjudging the noise](misspecified_sigma/README.md): the strategy believes `sigma`
  is 0.25 to 4 times what it is.
* [Outliers](outliers/README.md): Student-t noise instead of Gaussian.
* [Drift](drift/README.md): the true effects wander between epochs, and the
  filter knows it; [drift, unaware](drift_unaware/README.md): the same worlds
  with a filter that does not.
* [Autocorrelated noise](autocorrelation/README.md): readings that remember
  the previous epoch's noise; [corrected](autocorrelation_corrected/README.md):
  the same worlds with the strategies told the long-run noise instead.
* [Bernoulli data](bernoulli/README.md): conversions at 5%, and fewer of
  them per epoch, until the Gaussian posterior stops describing them.

Each study is a folder: the write-up (`README.md`), the one spec that produced
the sweep (`spec.yaml`; its `environment` lists the worlds) and the figure
(`koth-arena plot` draws it).

## What this means for your test

### Measure `sigma`, do not guess it

A factor of 2 costs KotH 39% more regret when guessed low and 66% when
guessed high; a factor of 4, 2.6x and 3.3x ([misjudging the
noise](misspecified_sigma/README.md)). The two directions are not the same
mistake: too low is overconfidence, which commits early and is sometimes
wrong; too high is under-confidence, which keeps buying information after the
answer is in, and that one never stops costing. It is worst for Thompson
sampling (11% at half, 119% at double), so if you must err, err low.

### Decide as often as your data allows

Strategies that commit gain on Thompson sampling as the number of decisions
per horizon grows: at 100 epochs per `1/rho` Thompson was 8% behind Gittins in
our arena, at 1000 it was 29%. Batching readings into coarse epochs gives
nothing back, and `sigma` rescales with the epoch anyway. Daily if you decide
daily, hourly if you can.

### Leave the outliers in

Heavy-tailed readings at the same variance cost every strategy 5-10% and
change no ordering ([outliers](outliers/README.md)); the filter averages them
out. Trim only where it keeps the variance at what `sigma` says, which is the
`sigma` point again, not a tails point.

### If your readings remember, whiten them first

Autocorrelated noise is the one world that reorders the field
([autocorrelation](autocorrelation/README.md)): at a lag-one correlation of
0.3 KotH loses 40% and Gittins 10%; at 0.9 Gittins is the only strategy still
standing. The filter credits every reading as fresh, and no strategy here
models the dependence. The one-number fix, telling the strategies the
long-run `sigma * sqrt((1 + phi) / (1 - phi))`, makes it worse
([corrected](autocorrelation_corrected/README.md)): it under-credits the
short runs the decisions turn on. Aggregate to epochs long enough that
consecutive readings are independent, or difference out the structure, before
the test; do not inflate `sigma`.

### If your effects move, tell the filter

Two studies, same worlds: one where every strategy's filter forgets at the
world's drift rate ([drift](drift/README.md)), one where it does not
([drift, unaware](drift_unaware/README.md)). Up to three steps per horizon
they agree. At ten, the unaware filter costs KotH 30% and Gittins 17% over
their aware selves; at thirty, every strategy pays 2.5x and the ordering is
gone. The forgetting is worth more than the choice of strategy from there,
and the package's `State.update` does not do it: between epochs, scale the
covariance up by the drift you expect, or keep the test short enough that
the world does not move during it.

### Bring a prior if you have one

Every study here starts from ignorance, which is the fair regime and the one
most tests run in. It is also expensive: a Gittins index told the true
distribution of effects beat the flat-prior one by 25% in an earlier arena.
That number is for Gittins and untested for KotH, but KotH takes a prior
through `cov` where Gittins can only take an independent one, and a team
with a history of tests has that prior.
