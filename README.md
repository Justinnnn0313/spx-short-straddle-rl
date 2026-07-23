# MLP-Initialized Reinforcement Learning for SPX Short-Straddle Exit Timing

This repository contains the code for a research project on exit timing in SPX short-straddle portfolios. The project studies a two-stage learning framework:

1. A supervised MLP exit policy trained from ex-post exit labels.
2. Reinforcement-learning fine-tuning initialized from the supervised MLP actor.

The repository is a GitHub-ready copy of the project code. Raw option datasets, generated training datasets, model checkpoints, generated figures, paper drafts, and large experiment outputs are intentionally excluded.

## Repository Layout

```text
spx_straddle_rl/          Core trading environment, RL agents, training, and evaluation code
scripts/        Main experiment scripts used by the paper workflow
requirements.txt
```

## Data Policy

The following files are not included:

- raw SPX option data
- generated episode datasets such as `episodes.pkl`
- imitation datasets such as `.npz` files
- model checkpoints such as `.pt` files
- full `results/` experiment folders
- generated paper figures and poster assets
- research notes and paper drafts

The code expects these artifacts to be available locally when reproducing the full experiments. Keep large or licensed data outside the repository.

## Expected Data Format

The full experiments require a local SPX option dataset converted into short-straddle episode files. Each row should represent one decision point for one candidate straddle episode. At minimum, the input data should identify the episode, timestamp, contract, underlying level, remaining time to maturity, quote state, and trading signals used by the environment.

Typical required columns include:

```text
episode_id
date or timestamp
step
underlying_price
strike
ttm_bars
moneyness
can_enter
can_exit
quote_available
```

The environment expects either leg-level quotes that can be combined into a straddle quote:

```text
call_bid, call_ask, put_bid, put_ask
```

or precomputed straddle quote columns:

```text
straddle_bid
straddle_ask
straddle_mid
spread_rel
```

The final state and reward design also use volatility, liquidity, and risk features when available:

```text
vrp
score
rv13_change
cvar or straddle_cvar
timevalue
synthetic
```

The repository does not include the proprietary/raw SPX option data or generated episode datasets.

## Main Workflow Scripts

The release keeps only the scripts needed for the paper's main workflow:

```text
spx_straddle_rl/build_simquote_episodes.py
scripts/build_exit_imitation_dataset.py
scripts/train_exit_imitation_policy.py
scripts/evaluate_exit_imitation_policy.py
scripts/train_exit_actor_critic_finetune.py
scripts/train_exit_advantage_weighted_finetune.py
scripts/evaluate_exit_actor_critic_policy.py
scripts/evaluate_hold_to_expiry_benchmark.py
scripts/summarize_portfolio_results.py
```

Research diagnostics, oracle analyses, DQN experiments, figure-generation scripts, checkpoint sweeps, paper drafts, and generated results are excluded from this release copy.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, replace the activation command with:

```bash
source .venv/bin/activate
```

## Notes

Full experiment reproduction requires local data preparation and result generation using the scripts under `spx_straddle_rl/` and `scripts/`.
