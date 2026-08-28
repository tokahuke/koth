# CLAUDE.md

## What this is

`koth` (king of the hill): the PyPI package for the top-k allocation heuristic
(k = 2, 3) for A/B/n tests with correlated arms. It decides how to split
the allocation, not when to stop: stopping is an opportunity-cost question and it is
not modelled here. Never describe it as a stopping rule. The research side -- the PINNs that produced the
nets, the derivations, the graveyards -- lives in the sibling `pinn` repo and
is NOT a dependency: nothing here imports `pinn`. Code may be copied from it.

## Hard rules

- Spelling: `koth` (lowercase) only for the module, package, pip name and
  CLI; `KotH` for the heuristic and the product in prose, and in class names
  (`KotH`, `KotH2`, `KotH3`) and contestant labels (`KotH, k = 3`).
- `numpy` and `scipy` are the only required dependencies. `import koth` must
  work with those alone; torch is the `torch` extra (a backend the numpy path must not need)
  and `arena` adds click and pyyaml on top of it.
- The arena (`koth/arena/`, copied from pinn's) benchmarks policies on torch:
  `poetry run koth-arena simulate ... --arms N`, then `koth-arena analyze`
  (the table) and `koth-arena plot` (regret and epochs exploring per policy,
  a PNG beside the pickle). The package enters as `KotH2`/`KotH3`
  (`policies/koth.py`: `Test` on the torch `Decider` over the filter's
  posterior, flattest-supported prior). Baselines: ETC, probability
  matching, elimination (z-test), Gittins (Brezzi-Lai index on the filter's
  flat prior, so not the Bayes optimum for the environment's effect draw).
  Time a `--size 64` probe at 2-4 threads before a long sweep. Production
  parameter values live outside the repo, never committed; `data/` is
  gitignored and holds sweeps and plots.
- Layout: one *contract* module per piece (`koth/two_arm.py`,
  `koth/three_arm.py`, `koth/decide.py`, `koth/state.py`, `koth/net.py`: ABCs generic over the array type,
  plus the exported weights and pinn's fixtures) and one *implementation* per
  backend in its own realm, `koth/numpy_/` (central differences) and
  `koth/torch_/` (autograd), both public. No namespace sniffing, no shared
  array shim. `koth/__init__.py` holds `Test`, the dimensionful layer: the
  readout dictionary (`mhat = m / (sigma sqrt(rho))`, `cov_hat = cov / (rho
  sigma**2)`, `V = (sigma / sqrt(rho)) vhat`) and nothing else, written once
  against the `Decider` ABC and taking the backend as an argument.
- `k` is a number outside and a class inside: `Test.decide(state, k)` takes
  2 or 3, and the `Decider` dispatches on `{net.K: net}` through the `Net`
  contract (`subset_value`/`subset_policy` on contrast means and their
  precision matrix). Nothing in `decide` knows a net's chart signature; a new
  arm count is a new `Net` subclass, not a branch.
- The backends carry the symmetry (the `|muhat|` reflection at two arms, the
  S3 fold and the projection onto the trained support at three), not the
  dimensionful layer. Differences on numpy are taken on the premium alone
  (the commit term differenced across the wall reads `1 / h`), with steps on
  the precision entries scaled by `det / tau`; the measured agreement with
  autograd is in each backend's `STEP` docstring.
- `State.cov` is the covariance of the *arm effects*, control included. The
  shared-control correlation between *lifts* is derived from the control's
  variance when contrasts are formed; a diagonal `cov` already carries it.
  Entering the lift covariance (`var(control)` off the diagonal) as if it
  were the arm covariance makes every contrast near-certain and commits on
  the spot: the README quickstart was written that way once.
- `sigma` is scalar and common to every arm by derivation: the Hamiltonian is
  quadratic in the allocation only because `sigma**2` factors out of the
  observation covariance. Unequal-noise arms are not a helper; they are
  another problem.
- `tools/from_pinn.py` exports weights and fixtures; it runs in pinn's venv,
  never here.
- `docs/` holds ANALYSES linked from the README, a chapter per folder
  (`docs/robustness/`) with an index `README.md`, and ONE FOLDER PER STUDY
  inside it: the write-up (`README.md`), the ONE spec that produced the
  sweep (`spec.yaml`, its `environment` naming every world) and the figure
  (`figure.png`, drawn by `koth-arena plot` from the pickle in gitignored
  `data/`; several worlds draw regret-vs-world lines, one world draws bars).
  No per-study scripts. It is human-facing and not a knowledge base: no
  derivations, no graveyards, no agent notes there.
- A spec has NO `params`: `environments` is a mapping of NAMED worlds, each
  spelling out every `Params` field plus `kind` (`normal` when absent) and
  that environment's own options; repetition is a YAML anchor on the first
  world and `<<: *base` merges (PyYAML resolves them, `save` writes worlds in
  full). `size` (tests per world) sits at the top level. Every contestant
  plays every world, runs carry the world label, `Study.environments` keeps
  each world's `Params` by label, and the label is the figure's axis text; an optional `told: {sigma: ...}`
  on a world is what every strategy is told instead of the truth (the
  per-world counterpart of a contestant's `sigma_factor`). Drift is `{<<: *base, eta: 0.01}`, outliers `{<<: *base, kind: student,
  df: 3}`.
- A contestant label `<name> x<value>` (`KotH, k = 3 x0.5`) is one point of
  `<name>`'s curve at `<value>`: `plot` draws such a sweep as lines against
  the value (log x over a decade or more), coloured by `<name>`, and the
  spec writes the labels that way; a `<name> (flat)` label is that strategy
  without a prior, drawn dashed in its colour. Worlds are the other line axis; plain
  names in one world are bars.
- A spec contestant is `{strategy: Name, key: value, ...}`; every extra key
  becomes an uppercased class attribute on that contestant's subclass
  (`sigma_factor: 0.5` sets `Bayesian.SIGMA_FACTOR`, the policy's belief
  about sigma as a multiple of the truth; `eta_factor: 0` a filter that
  does not know the world drifts). Misspecification studies are spec
  files, not code. Environments: `normal`, `student` (t noise at `df`,
  variance matched), `ar1` (standardized noise with lag-one correlation
  `phi`, variance matched), `bernoulli` (success rates at `rate + effect`
  over `round(a_i * trials)` draws; its `describe` derives the `sigma` the
  policies are told, `sqrt(rate (1 - rate) / trials)`, and a spec that
  writes one is refused), `matern` (a bid ladder: arms at bids 0..N-1,
  effects one GP draw with a Matern-5/2 kernel at `lengthscale`, no control).
  A kind that fixes a `Params` field derives it in `describe`, never in the
  spec. Every environment exposes `prior()`, its effect law as a Gaussian
  over the arms; a contestant with `prior: world` is `prime`d with it after
  `init`: `Bayesian` keeps the diagonal, `KotH` and `JointThompson` take it
  whole on KotH's own full-covariance `State` (with the same drift step as
  the per-arm filter). `docs/correlation/` is the chapter for that world.
- Every module carries an `assert`-based self-check: `poetry run python -m
  KotH` (units), `-m koth.numpy_.two_arm` / `.three_arm` / `.decide` / `.state`,
  the same four under `koth.torch_`, and `koth.arena.harness`,
  `.policies.n_arm`, `.policies.koth`, `.spec`.
- Final step of any Python change: `poetry run black <files>`.

## Style

Ponytail (lazy-minimal): shortest working code, no speculative scaffolding.
Pedro's house style: full type-hinted signatures, real-bool conditionals,
ASCII only, black as the final pass. Comments are not a notebook: keep
measured numbers and the reason a constant has its value, cut narration.
