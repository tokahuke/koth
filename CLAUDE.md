# CLAUDE.md

## What this is

`koth` (king of the hill): the PyPI package for the top-k allocation heuristic
(k = 2, 3) for A/B/n tests with correlated arms. It decides how to split
the allocation, not when to stop: stopping is an opportunity-cost question and it is
not modelled here. Never describe it as a stopping rule. The research side -- the PINNs that produced the
nets, the derivations, the graveyards -- lives in the sibling `pinn` repo and
is NOT a dependency: nothing here imports `pinn`. Code may be copied from it.

## Hard rules

- `numpy` and `scipy` are the only required dependencies. `import koth` must
  work with those alone; torch is the `torch` extra (a backend the numpy path must not need)
  and `arena` adds click and pyyaml on top of it.
- The arena (`koth/arena/`, copied from pinn's) benchmarks policies on torch:
  `poetry run koth-arena simulate ... --arms N`, then `koth-arena analyze`
  (the table) and `koth-arena plot` (regret and epochs exploring per policy,
  a PNG beside the pickle). The package enters as `Koth2`/`Koth3`
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
- `docs/` holds ANALYSES linked from the README, one per name: the spec that
  produced the sweep (`<name>.spec.yaml`), the script that draws it
  (`<name>.py`, reads the pickle from gitignored `data/`), the figure
  (`<name>.png`) and the write-up (`<name>.md`). It is human-facing and not
  a knowledge base: no derivations, no graveyards, no agent notes there.
  `resources/` holds what the README itself embeds.
- A spec contestant is `{strategy: Name, key: value, ...}`; every extra key
  becomes an uppercased class attribute on that contestant's subclass
  (`sigma_factor: 0.5` sets `Bayesian.SIGMA_FACTOR`, the policy's belief
  about sigma as a multiple of the truth). Misspecification studies are
  spec files, not code.
- Every module carries an `assert`-based self-check: `poetry run python -m
  koth` (units), `-m koth.numpy_.two_arm` / `.three_arm` / `.decide` / `.state`,
  the same four under `koth.torch_`, and `koth.arena.harness`,
  `.policies.n_arm`, `.policies.koth`, `.spec`.
- Final step of any Python change: `poetry run black <files>`.

## Style

Ponytail (lazy-minimal): shortest working code, no speculative scaffolding.
Pedro's house style: full type-hinted signatures, real-bool conditionals,
ASCII only, black as the final pass. Comments are not a notebook: keep
measured numbers and the reason a constant has its value, cut narration.
