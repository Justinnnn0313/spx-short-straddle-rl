from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "spx_straddle_rl"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from env import ENTER_SHORT, WAIT, StraddleEnv  # noqa: E402
from test_agent import fill_price, signed_pnl, terminal_value  # noqa: E402


DEFAULT_ARTIFACT_DIR = Path("artifacts_v3r_simquote")
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACT_DIR / "imitation_exit"


def load_artifacts(artifact_dir: str | Path):
    artifact_dir = Path(artifact_dir)
    with (artifact_dir / "episodes.pkl").open("rb") as f:
        episodes = pickle.load(f)
    with (artifact_dir / "norm_stats.pkl").open("rb") as f:
        norm_stats = pickle.load(f)
    return episodes, norm_stats


def short_net_pnl(entry_fill: float, exit_fill: float, contract_multiplier: float, commission_per_option_contract: float) -> float:
    return float(
        signed_pnl(+1, entry_fill, exit_fill, contract_multiplier, contracts=1)
        - 4.0 * commission_per_option_contract
    )


def oracle_exit_step(
    episode: pd.DataFrame,
    min_hold_bars: int,
    contract_multiplier: float,
    commission_per_option_contract: float,
) -> int:
    entry = episode.iloc[0]
    entry_fill = fill_price(entry, position=+1, is_entry=True)
    best_step = min(max(min_hold_bars, 1), len(episode) - 1)
    best_pnl = -np.inf
    for step, row in episode.iloc[best_step:].iterrows():
        if int(row["ttm_bars"]) == 0 or step == len(episode) - 1:
            exit_fill = terminal_value(row)
        else:
            exit_fill = fill_price(row, position=+1, is_entry=False)
        pnl = short_net_pnl(
            entry_fill,
            exit_fill,
            contract_multiplier,
            commission_per_option_contract,
        )
        if pnl > best_pnl:
            best_pnl = pnl
            best_step = int(step)
    return int(best_step)


def build_rows_for_episode(
    episode_id: int,
    episode: pd.DataFrame,
    norm_stats,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[int], list[dict]]:
    oracle_step = oracle_exit_step(
        episode,
        args.min_hold_bars,
        args.contract_multiplier,
        args.commission_per_option_contract,
    )
    env = StraddleEnv(
        {
            "episode_data": episode,
            "norm_stats": norm_stats,
            "model_variant": "exit",
            "obs_features": "position",
            "reward_mode": "direct",
            "min_hold_bars": args.min_hold_bars,
        }
    )
    obs, _ = env.reset()
    obs, _reward, done, _truncated, info = env.step(ENTER_SHORT)
    features: list[np.ndarray] = []
    labels: list[int] = []
    meta: list[dict] = []
    while not done:
        step = int(env.t)
        row = episode.iloc[min(step, len(episode) - 1)]
        label = int(step >= oracle_step and step >= args.min_hold_bars)
        features.append(np.asarray(obs, dtype=np.float32))
        labels.append(label)
        meta.append(
            {
                "episode_id": episode_id,
                "step": step,
                "label": label,
                "oracle_exit_step": oracle_step,
                "date": str(row["date"]),
                "year": int(episode.iloc[0]["year"]),
                "ttm_bars": int(row["ttm_bars"]),
                "synthetic_bar": bool(row.get("synthetic_bar", False)),
            }
        )
        obs, _reward, done, _truncated, info = env.step(WAIT)
    return features, labels, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build oracle-guided exit imitation dataset.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--min-hold-bars", type=int, default=26)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--contract-multiplier", type=float, default=100.0)
    parser.add_argument("--commission-per-option-contract", type=float, default=1.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes, norm_stats = load_artifacts(args.artifact_dir)
    selected = [
        ep
        for ep in episodes
        if args.start_year <= int(ep.iloc[0]["year"]) <= args.end_year
    ]
    if args.max_episodes is not None:
        selected = selected[: args.max_episodes]

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    meta_rows: list[dict] = []
    for episode_id, episode in enumerate(selected):
        if episode_id == 0 or (episode_id + 1) % 500 == 0 or episode_id + 1 == len(selected):
            print(f"imitation episode {episode_id + 1}/{len(selected)}", flush=True)
        features, labels, meta = build_rows_for_episode(episode_id, episode, norm_stats, args)
        x_rows.extend(features)
        y_rows.extend(labels)
        meta_rows.extend(meta)

    x = np.vstack(x_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    dataset_path = output_dir / f"exit_imitation_{args.start_year}_{args.end_year}_minhold{args.min_hold_bars}.npz"
    meta_path = output_dir / f"exit_imitation_meta_{args.start_year}_{args.end_year}_minhold{args.min_hold_bars}.csv"
    np.savez_compressed(
        dataset_path,
        x=x,
        y=y,
        min_hold_bars=np.asarray([args.min_hold_bars], dtype=np.int64),
    )
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
    print(f"rows: {len(y)}")
    print(f"features: {x.shape[1]}")
    print(f"exit_label_rate: {float(y.mean()):.6f}")
    print(f"dataset: {dataset_path}")
    print(f"meta: {meta_path}")


if __name__ == "__main__":
    main()
