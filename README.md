# FPL Sequential Optimisation

This repository explores an FPL manager that makes decisions over a season rather than optimizing one isolated gameweek. Its research objective is to estimate uncertain player performance, then choose squads and actions under FPL rules to maximize cumulative points.

The dissertation frames this as sequential optimization under partial observability. It proposes Bayesian latent-form models and decision methods including Thompson sampling, tactical lookahead, and deep reinforcement learning. The source contains implementations and prototypes at different levels of maturity; it is not yet an end-to-end, validated autonomous manager. Some dissertation chapters are proposals or placeholders, so theoretical coverage should not be read as evidence that a method is implemented or evaluated.

## Repository map

- `src/data/`, `src/api/`, and `src/pipeline/` contain data acquisition, feature preparation, the pipeline, and a FastAPI service.
- `src/models/` contains latent-form model experiments and the belief-state representation.
- `src/optimization/selection/` contains squad-selection and single-gameweek decision methods.
- `src/optimization/planning/` contains MCTS and rollout code; `ddqn/` and `mfg/` are separate research prototypes.
- `src/mlops/` and `src/utils/` contain monitoring, tracking helpers, and JAX utilities.
- `docs/thesis/` contains the dissertation chapters and bibliography. `docs/engineering/` describes the implemented code.

## Setup and tests

Use Python and install the dependencies listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set credentials only if using authenticated FPL endpoints. Run the test suite with:

```bash
python -m pytest -q
```

## Documentation

Start at [`docs/README.md`](docs/README.md) for the document index and [`docs/engineering/architecture.md`](docs/engineering/architecture.md) for the source-backed architecture. The review and reorganization sequence is recorded in [`WORK_PLAN.md`](WORK_PLAN.md).
