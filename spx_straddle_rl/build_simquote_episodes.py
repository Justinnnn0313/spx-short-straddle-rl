from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC3 = ROOT / "src3"
if str(SRC3) not in sys.path:
    sys.path.insert(0, str(SRC3))

from build_episodes import compute_norm_stats  # noqa: E402


DEFAULT_SOURCE_ARTIFACT_DIR = Path("artifacts_v3_settlement_fixed")
DEFAULT_ARTIFACT_DIR = Path("artifacts_v3r_simquote")

BASE_COLS = [
    "quote_datetime",
    "expiration",
    "strike",
    "date",
    "year",
    "ttm_trading",
    "ttm_bars",
    "tod",
    "straddle_mid",
    "straddle_bid",
    "straddle_ask",
    "spread_abs",
    "spread_rel",
    "straddle_iv",
    "straddle_delta",
    "vrp",
    "score",
    "delta_rv13",
    "rv",
    "cvar",
    "straddle_cvar",
    "short_straddle_pnl_ret",
    "underlying_price",
    "moneyness",
    "synthetic_terminal",
    "synthetic_bar",
    "quote_available",
    "can_enter",
    "can_exit",
    "sim_time_value",
    "sim_slippage",
    "sim_short_exit_cost",
    "sim_long_exit_value",
    "sim_entry_long_cost",
]


def load_source_episodes(source_artifact_dir: str | Path) -> list[pd.DataFrame]:
    with (Path(source_artifact_dir) / "episodes.pkl").open("rb") as f:
        return pickle.load(f)


def clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def simulated_quote(
    *,
    underlying: float,
    strike: float,
    entry_premium: float,
    rv: float,
    rv_entry: float,
    remaining_fraction: float,
    time_value_fraction: float,
    moneyness_decay_scale: float,
    base_spread_mid_fraction: float,
    base_spread_underlying_fraction: float,
    stress_spread_fraction: float,
    stress_spread_moneyness_scale: float,
) -> dict[str, float]:
    intrinsic = abs(underlying - strike)
    log_moneyness_distance = abs(np.log(max(underlying, 1e-8) / max(strike, 1e-8)))
    rv_entry = rv_entry if np.isfinite(rv_entry) and rv_entry > 1e-8 else max(rv, 1e-8)
    vol_scale = clip(rv / rv_entry, 0.5, 2.0)
    otm_decay = float(np.exp(-log_moneyness_distance / moneyness_decay_scale))
    time_value = (
        time_value_fraction
        * entry_premium
        * np.sqrt(max(remaining_fraction, 0.0))
        * vol_scale
        * otm_decay
    )
    sim_mid = intrinsic + time_value
    base_spread = max(
        base_spread_mid_fraction * sim_mid,
        base_spread_underlying_fraction * underlying,
    )
    stress_spread = (
        stress_spread_fraction
        * sim_mid
        * (log_moneyness_distance / stress_spread_moneyness_scale)
    )
    sim_spread = max(0.0, base_spread + stress_spread)
    sim_bid = max(0.0, sim_mid - 0.5 * sim_spread)
    sim_ask = sim_mid + 0.5 * sim_spread
    spread_rel = sim_spread / sim_mid if sim_mid > 1e-8 else 0.0
    sim_iv_proxy = rv + time_value / max(underlying, 1.0)
    vrp = sim_iv_proxy - rv
    score = vrp / spread_rel if spread_rel > 1e-8 else 0.0
    return {
        "intrinsic": float(intrinsic),
        "time_value": float(time_value),
        "sim_mid": float(sim_mid),
        "sim_bid": float(sim_bid),
        "sim_ask": float(sim_ask),
        "sim_spread": float(sim_spread),
        "spread_rel": float(spread_rel),
        "sim_iv_proxy": float(sim_iv_proxy),
        "vrp": float(vrp),
        "score": float(score),
    }


def interpolate_underlying_path(
    start_row: pd.Series,
    terminal_row: pd.Series,
    bars: int,
) -> pd.DataFrame:
    start_time = pd.Timestamp(start_row["quote_datetime"])
    end_time = pd.Timestamp(terminal_row["quote_datetime"])
    if end_time <= start_time:
        end_time = start_time + pd.Timedelta(minutes=30 * bars)
    times = pd.date_range(start=start_time, end=end_time, periods=bars + 2)[1:-1]
    start_s = float(start_row["underlying_price"])
    end_s = float(terminal_row["underlying_price"])
    values = np.linspace(start_s, end_s, bars + 2)[1:-1]
    return pd.DataFrame({"quote_datetime": times, "underlying_price": values})


def expand_episode(
    episode: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    episode = episode.copy().reset_index(drop=True)
    for col, default in [
        ("synthetic_terminal", False),
        ("synthetic_bar", False),
        ("quote_available", True),
        ("can_enter", True),
        ("can_exit", False),
        ("sim_time_value", 0.0),
        ("sim_slippage", 0.0),
        ("sim_short_exit_cost", np.nan),
        ("sim_long_exit_value", np.nan),
        ("sim_entry_long_cost", np.nan),
    ]:
        if col not in episode:
            episode[col] = default

    episode["synthetic_terminal"] = episode["synthetic_terminal"].astype(bool)
    terminal_is_synthetic = bool(episode.iloc[-1]["synthetic_terminal"])
    real_part = episode.iloc[:-1].copy() if terminal_is_synthetic else episode.copy()
    terminal = episode.iloc[-1].copy() if terminal_is_synthetic else None

    real_part["synthetic_bar"] = False
    real_part["quote_available"] = True
    real_part["can_enter"] = True
    real_part["can_exit"] = True
    real_part["sim_time_value"] = 0.0
    real_part["sim_slippage"] = 0.0
    real_part["sim_short_exit_cost"] = real_part["straddle_ask"].astype(float)
    real_part["sim_long_exit_value"] = real_part["straddle_bid"].astype(float)
    real_part["sim_entry_long_cost"] = real_part["straddle_ask"].astype(float)

    parts = [real_part]
    if terminal_is_synthetic and terminal is not None and len(real_part) > 0:
        last_real = real_part.iloc[-1]
        first = real_part.iloc[0]
        entry_premium = float(first["straddle_bid"])
        strike = float(first["strike"])
        rv_entry = float(first.get("rv", 0.0))
        path = interpolate_underlying_path(last_real, terminal, args.continuation_bars)
        synthetic_rows = []
        for idx, path_row in path.iterrows():
            remaining_fraction = (len(path) - idx) / max(len(path), 1)
            underlying = float(path_row["underlying_price"])
            rv = float(last_real.get("rv", 0.0))
            quote = simulated_quote(
                underlying=underlying,
                strike=strike,
                entry_premium=entry_premium,
                rv=rv,
                rv_entry=rv_entry,
                remaining_fraction=remaining_fraction,
                time_value_fraction=args.time_value_fraction,
                moneyness_decay_scale=args.moneyness_decay_scale,
                base_spread_mid_fraction=args.base_spread_mid_fraction,
                base_spread_underlying_fraction=args.base_spread_underlying_fraction,
                stress_spread_fraction=args.stress_spread_fraction,
                stress_spread_moneyness_scale=args.stress_spread_moneyness_scale,
            )
            row = last_real.copy()
            row["quote_datetime"] = path_row["quote_datetime"]
            row["date"] = str(pd.Timestamp(path_row["quote_datetime"]).date())
            row["year"] = int(pd.Timestamp(path_row["quote_datetime"]).year)
            row["underlying_price"] = underlying
            row["moneyness"] = strike / underlying if underlying > 0 else 0.0
            row["straddle_mid"] = quote["sim_mid"]
            row["straddle_bid"] = quote["sim_bid"]
            row["straddle_ask"] = quote["sim_ask"]
            row["spread_abs"] = quote["sim_spread"]
            row["spread_rel"] = quote["spread_rel"]
            row["straddle_iv"] = quote["sim_iv_proxy"]
            row["straddle_delta"] = 0.0
            row["vrp"] = quote["vrp"]
            row["score"] = quote["score"]
            row["straddle_cvar"] = float(row.get("cvar", 0.0))
            row["synthetic_terminal"] = False
            row["synthetic_bar"] = True
            row["quote_available"] = False
            row["can_enter"] = False
            row["can_exit"] = True
            row["sim_time_value"] = quote["time_value"]
            row["sim_slippage"] = 0.5 * quote["sim_spread"]
            row["sim_short_exit_cost"] = quote["sim_ask"]
            row["sim_long_exit_value"] = quote["sim_bid"]
            row["sim_entry_long_cost"] = quote["sim_ask"]
            synthetic_rows.append(row)
        if synthetic_rows:
            parts.append(pd.DataFrame(synthetic_rows))

        terminal = terminal.copy()
        intrinsic = abs(float(terminal["underlying_price"]) - float(terminal["strike"]))
        terminal["straddle_mid"] = intrinsic
        terminal["straddle_bid"] = intrinsic
        terminal["straddle_ask"] = intrinsic
        terminal["spread_abs"] = 0.0
        terminal["spread_rel"] = 0.0
        terminal["straddle_iv"] = 0.0
        terminal["straddle_delta"] = 0.0
        terminal["vrp"] = 0.0
        terminal["score"] = 0.0
        terminal["synthetic_bar"] = True
        terminal["quote_available"] = False
        terminal["can_enter"] = False
        terminal["can_exit"] = False
        terminal["sim_time_value"] = 0.0
        terminal["sim_slippage"] = 0.0
        terminal["sim_short_exit_cost"] = intrinsic
        terminal["sim_long_exit_value"] = intrinsic
        terminal["sim_entry_long_cost"] = intrinsic
        parts.append(terminal.to_frame().T)

    expanded = pd.concat(parts, ignore_index=True)
    expanded = expanded.sort_values("quote_datetime").reset_index(drop=True)
    expanded["step"] = np.arange(len(expanded), dtype=np.int32)
    expanded["ttm_bars"] = (len(expanded) - 1 - expanded["step"]).astype(np.int32)
    expanded = expanded[BASE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return expanded


def build_simquote_episodes(source_episodes: list[pd.DataFrame], args: argparse.Namespace) -> list[pd.DataFrame]:
    return [expand_episode(episode, args) for episode in source_episodes]


def save_artifacts(episodes: list[pd.DataFrame], artifact_dir: str | Path) -> None:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    norm_stats = compute_norm_stats(episodes)
    with (artifact_dir / "episodes.pkl").open("wb") as f:
        pickle.dump(episodes, f)
    with (artifact_dir / "norm_stats.pkl").open("wb") as f:
        pickle.dump(norm_stats, f)
    print(f"NormStats={norm_stats}")
    print(f"Saved V3R sim-quote artifacts to {artifact_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V3R simulated-quote episodes.")
    parser.add_argument("--source-artifact-dir", default=str(DEFAULT_SOURCE_ARTIFACT_DIR))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--continuation-bars", type=int, default=60)
    parser.add_argument("--time-value-fraction", type=float, default=1.0)
    parser.add_argument("--moneyness-decay-scale", type=float, default=0.08)
    parser.add_argument("--base-spread-mid-fraction", type=float, default=0.02)
    parser.add_argument("--base-spread-underlying-fraction", type=float, default=0.001)
    parser.add_argument("--stress-spread-fraction", type=float, default=0.01)
    parser.add_argument("--stress-spread-moneyness-scale", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading source episodes from {args.source_artifact_dir}", flush=True)
    source_episodes = load_source_episodes(args.source_artifact_dir)
    print(f"Loaded source episodes: {len(source_episodes)}", flush=True)
    episodes = build_simquote_episodes(source_episodes, args)
    rows = []
    for ep in episodes:
        rows.append(
            {
                "year": int(ep.iloc[0]["year"]),
                "bars": len(ep),
                "synthetic_bars": int(ep["synthetic_bar"].sum()),
                "terminal_synth": bool(ep.iloc[-1]["synthetic_terminal"]),
            }
        )
    stats = pd.DataFrame(rows)
    print("Episodes by year:")
    print(stats["year"].value_counts().sort_index().to_string())
    print(f"Total synthetic bars: {int(stats['synthetic_bars'].sum())}")
    print(f"Mean bars: {stats['bars'].mean():.1f}")
    save_artifacts(episodes, args.artifact_dir)


if __name__ == "__main__":
    main()
