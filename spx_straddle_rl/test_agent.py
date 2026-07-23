from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import ACTION_NAMES, ENTER_LONG, ENTER_SHORT, EXIT, WAIT, StraddleEnv
from rl_agent_common import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MODEL_DIR,
    RandomEpisodeStraddleEnv,
    extract_action,
    load_episode_split,
    save_json,
)

STRADDLE_LEGS = 2


def positive_dollar_cvar(values: pd.Series, confidence: float = 0.05) -> float:
    clean = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    threshold = clean.quantile(confidence)
    tail = clean[clean <= threshold]
    return float(-tail.mean()) if len(tail) else 0.0


def build_portfolio_nav(
    nav_rows: list[dict],
    initial_capital: float,
    max_slots: int,
    start_timestamp: Any,
) -> pd.DataFrame:
    initial_per_slot = initial_capital / max_slots
    initial_rows = [
        {
            "quote_datetime": start_timestamp,
            "slot_id": slot_id,
            "slot_nav": initial_per_slot,
            "slot_active_episode": False,
        }
        for slot_id in range(max_slots)
    ]
    nav_df = pd.DataFrame(initial_rows + nav_rows)
    nav_df["quote_datetime"] = pd.to_datetime(nav_df["quote_datetime"])
    nav_wide = (
        nav_df.pivot_table(
            index="quote_datetime",
            columns="slot_id",
            values="slot_nav",
            aggfunc="last",
        )
        .sort_index()
        .ffill()
        .fillna(initial_per_slot)
    )
    active_wide = (
        nav_df.pivot_table(
            index="quote_datetime",
            columns="slot_id",
            values="slot_active_episode",
            aggfunc="last",
        )
        .sort_index()
        .ffill()
        .fillna(False)
    )
    portfolio = pd.DataFrame(
        {
            "quote_datetime": nav_wide.index,
            "portfolio_nav": nav_wide.sum(axis=1).to_numpy(dtype=float),
            "active_slots": active_wide.sum(axis=1).to_numpy(dtype=float),
        }
    )
    portfolio["date"] = portfolio["quote_datetime"].dt.date.astype(str)
    portfolio["portfolio_return"] = portfolio["portfolio_nav"].pct_change().fillna(0.0)
    portfolio["cummax_nav"] = portfolio["portfolio_nav"].cummax()
    portfolio["drawdown"] = portfolio["portfolio_nav"] / portfolio["cummax_nav"] - 1.0
    return portfolio


def portfolio_metrics(
    portfolio_nav: pd.DataFrame,
    initial_capital: float,
    bars_per_year: int = 252 * 13,
) -> dict[str, float | bool]:
    nav = portfolio_nav["portfolio_nav"].astype(float)
    returns = portfolio_nav["portfolio_return"].astype(float).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    nonzero_returns = returns.iloc[1:] if len(returns) > 1 else returns
    final_nav = float(nav.iloc[-1])
    min_nav = float(nav.min())
    total_return = final_nav / initial_capital - 1.0
    n_bars = max(len(nav) - 1, 1)
    annual_return = (final_nav / initial_capital) ** (bars_per_year / n_bars) - 1.0 if final_nav > 0 else -1.0
    vol = float(nonzero_returns.std(ddof=1)) if len(nonzero_returns) > 1 else 0.0
    annual_vol = vol * np.sqrt(bars_per_year)
    mean_return = float(nonzero_returns.mean()) if len(nonzero_returns) else 0.0
    sharpe = mean_return / vol * np.sqrt(bars_per_year) if vol > 0 else 0.0
    max_drawdown = float(portfolio_nav["drawdown"].min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    return {
        "initial_capital": float(initial_capital),
        "final_nav": final_nav,
        "min_nav": min_nav,
        "nav_went_nonpositive": bool((nav <= 0).any()),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_return_defined": bool(final_nav > 0),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "bar_return_cvar_5": float(positive_dollar_cvar(nonzero_returns)),
        "n_nav_points": float(len(nav)),
    }


def plot_portfolio_curves(
    portfolio_nav: pd.DataFrame,
    output_dir: Path,
    title: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = portfolio_nav.copy()
    plot_df["quote_datetime"] = pd.to_datetime(plot_df["quote_datetime"])
    nav_path = output_dir / "portfolio_nav_curve.png"
    dd_path = output_dir / "portfolio_drawdown_curve.png"
    combined_path = output_dir / "portfolio_nav_drawdown.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_df["quote_datetime"], plot_df["portfolio_nav"], linewidth=1.5)
    ax.set_title(f"{title} Portfolio NAV")
    ax.set_xlabel("Time")
    ax.set_ylabel("Portfolio NAV ($)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(nav_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(plot_df["quote_datetime"], plot_df["drawdown"], linewidth=1.2, color="tab:red")
    ax.set_title(f"{title} Drawdown")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(dd_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(plot_df["quote_datetime"], plot_df["portfolio_nav"], linewidth=1.5)
    axes[0].set_title(f"{title} Portfolio NAV")
    axes[0].set_ylabel("Portfolio NAV ($)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(plot_df["quote_datetime"], plot_df["drawdown"], linewidth=1.2, color="tab:red")
    axes[1].set_title("Drawdown")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(combined_path, dpi=160)
    plt.close(fig)
    return {
        "nav_curve_png": str(nav_path),
        "drawdown_curve_png": str(dd_path),
        "combined_curve_png": str(combined_path),
    }


@dataclass
class SlotState:
    slot_id: int
    capital: float
    available_date: str
    disabled: bool = False


class NumpyMLPPolicy:
    def __init__(self, weights: dict[str, np.ndarray]):
        self.weights = weights

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> "NumpyMLPPolicy":
        checkpoint = Path(checkpoint)
        policy_state_path = checkpoint / "policies" / "default_policy" / "policy_state.pkl"
        if not policy_state_path.exists():
            policy_state_path = checkpoint / "policy_state.pkl"
        if not policy_state_path.exists():
            raise FileNotFoundError(f"Missing policy_state.pkl under {checkpoint}")
        with policy_state_path.open("rb") as f:
            state = pickle.load(f)
        return cls(state["weights"])

    def compute_single_action(self, obs: np.ndarray, explore: bool = False) -> int:
        del explore
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        w = self.weights
        x = np.tanh(
            x @ w["_hidden_layers.0._model.0.weight"].T
            + w["_hidden_layers.0._model.0.bias"]
        )
        x = np.tanh(
            x @ w["_hidden_layers.1._model.0.weight"].T
            + w["_hidden_layers.1._model.0.bias"]
        )
        logits = x @ w["_logits._model.0.weight"].T + w["_logits._model.0.bias"]
        return int(np.argmax(logits[0]))


class RllibRestoredPolicy:
    def __init__(self, checkpoint: str | Path):
        import ray
        from ray.rllib.algorithms.algorithm import Algorithm
        from ray.rllib.models import ModelCatalog
        from ray.tune.registry import register_env

        from action_mask_model import TorchActionMaskModel

        if not ray.is_initialized():
            ray.init(include_dashboard=False, ignore_reinit_error=True)
        ModelCatalog.register_custom_model("v3r_torch_action_mask", TorchActionMaskModel)
        register_env("v3r_eval_env", lambda config: RandomEpisodeStraddleEnv(config))
        register_env(
            "random_v3r_simquote_episode_env",
            lambda config: RandomEpisodeStraddleEnv(config),
        )
        register_env(
            "random_v3r_simquote_dqn_episode_env",
            lambda config: RandomEpisodeStraddleEnv(config),
        )
        self.algo = Algorithm.from_checkpoint(str(checkpoint))

    def compute_single_action(self, obs: np.ndarray, explore: bool = False) -> int:
        return extract_action(self.algo.compute_single_action(obs, explore=explore))

    def close(self) -> None:
        import ray

        self.algo.stop()
        ray.shutdown()


def load_policy(args: argparse.Namespace):
    if args.policy_loader == "numpy":
        return NumpyMLPPolicy.from_checkpoint(args.checkpoint)
    return RllibRestoredPolicy(args.checkpoint)


def fill_price(row: pd.Series, position: int, is_entry: bool) -> float:
    quote_available = bool(row.get("quote_available", True))
    if position == +1:
        if is_entry:
            return float(row["straddle_bid"])
        if quote_available:
            return float(row["straddle_ask"])
        return float(row.get("sim_short_exit_cost", row["straddle_ask"]))
    if position == -1:
        if is_entry:
            return float(row.get("sim_entry_long_cost", row["straddle_ask"]))
        if quote_available:
            return float(row["straddle_bid"])
        return float(row.get("sim_long_exit_value", row["straddle_bid"]))
    raise ValueError("position must be +1 or -1")


def terminal_value(row: pd.Series) -> float:
    return float(abs(float(row["underlying_price"]) - float(row["strike"])))


def required_entry_capital(
    row: pd.Series,
    position: int,
    entry_fill: float,
    contract_multiplier: float,
    entry_capital_check: str,
    short_margin_rate: float,
) -> float:
    premium = float(entry_fill) * contract_multiplier
    if entry_capital_check == "none":
        return 0.0
    if entry_capital_check == "premium":
        return premium
    if entry_capital_check != "short_margin":
        raise ValueError("entry_capital_check must be one of: none, premium, short_margin")
    if position == +1:
        underlying = float(row["underlying_price"])
        return max(premium, underlying * contract_multiplier * short_margin_rate)
    return premium


def choose_contracts(
    row: pd.Series,
    capital: float,
    position: int,
    entry_fill: float,
    args: argparse.Namespace,
) -> int:
    one_way_commission = STRADDLE_LEGS * args.commission_per_option_contract
    if args.contract_sizing == "fixed_one":
        contracts = 1
    elif args.contract_sizing == "stress_budget":
        underlying = float(row["underlying_price"])
        if position == +1:
            stress_loss = max(0.0, args.stress_move * underlying - entry_fill)
        else:
            stress_loss = float(entry_fill)
        stress_loss = stress_loss * args.contract_multiplier + 2.0 * one_way_commission
        contracts = int(np.floor(capital * args.slot_risk_fraction / stress_loss)) if stress_loss > 0 else 1
    else:
        raise ValueError("contract_sizing must be one of: fixed_one, stress_budget")
    if args.max_contracts_per_slot is not None:
        contracts = min(contracts, args.max_contracts_per_slot)
    required = required_entry_capital(
        row=row,
        position=position,
        entry_fill=entry_fill,
        contract_multiplier=args.contract_multiplier,
        entry_capital_check=args.entry_capital_check,
        short_margin_rate=args.short_margin_rate,
    )
    per_contract_cash = required + one_way_commission
    if per_contract_cash > 0:
        contracts = min(contracts, int(np.floor(capital / per_contract_cash)))
    return max(0, int(contracts))


def mark_to_market_nav(
    capital: float,
    position: int,
    entry_fill: float | None,
    row: pd.Series,
    contracts: int,
    contract_multiplier: float,
) -> float:
    if position == 0 or entry_fill is None or contracts <= 0:
        return float(capital)
    exit_fill = fill_price(row, position=position, is_entry=False)
    if position == +1:
        pnl = (entry_fill - exit_fill) * contract_multiplier * contracts
    else:
        pnl = (exit_fill - entry_fill) * contract_multiplier * contracts
    return float(capital + pnl)


def select_daily_candidates(
    episodes: list[pd.DataFrame],
    oos_start_year: int,
    oos_end_year: int,
    target_ttm: int,
) -> list[tuple[str, int, pd.DataFrame]]:
    rows = []
    for episode_id, episode in enumerate(episodes):
        first = episode.iloc[0]
        year = int(first["year"])
        if not (oos_start_year <= year <= oos_end_year):
            continue
        rows.append(
            {
                "episode_id": episode_id,
                "start_date": str(first["date"]),
                "start_datetime": first["quote_datetime"],
                "abs_ttm": abs(float(first["ttm_trading"]) - target_ttm),
                "abs_moneyness": abs(float(first["moneyness"]) - 1.0),
                "spread_rel": float(first["spread_rel"]),
                "episode": episode,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    daily = (
        df.sort_values(
            ["start_date", "abs_ttm", "abs_moneyness", "spread_rel", "start_datetime"],
            ascending=[True, True, True, True, True],
        )
        .groupby("start_date", sort=True, as_index=False)
        .first()
    )
    return [
        (str(row["start_date"]), int(row["episode_id"]), row["episode"])
        for _, row in daily.sort_values("start_date").iterrows()
    ]


def simulate_slot_episode(
    policy: NumpyMLPPolicy,
    episode: pd.DataFrame,
    norm_stats: Any,
    slot: SlotState,
    args: argparse.Namespace,
) -> tuple[SlotState, list[dict], list[dict], dict]:
    env = StraddleEnv(
        {
            "episode_data": episode,
            "norm_stats": norm_stats,
            "model_variant": args.model_variant,
            "alpha": args.alpha,
            "lambda_": args.reward_lambda,
            "mu": args.mu,
            "beta_vrp": args.beta_vrp,
            "use_action_mask_obs": args.use_action_mask,
            "reward_mode": args.reward_mode,
            "obs_features": args.obs_features,
            "min_hold_bars": args.min_hold_bars,
        }
    )
    obs, _info = env.reset()
    capital = float(slot.capital)
    position = 0
    entry_fill: float | None = None
    entry_step: int | None = None
    contracts = 0
    episode_contracts = 0
    entries = 0
    exits = 0
    margin_blocks = 0
    maintenance_liquidation = False
    total_reward = 0.0
    nav_rows: list[dict] = []
    trade_rows: list[dict] = []
    step = 0
    done = False

    while not done:
        row = episode.iloc[min(env.t, len(episode) - 1)]
        action = policy.compute_single_action(obs, explore=False)
        planned_contracts = 0
        if action in {ENTER_SHORT, ENTER_LONG} and position == 0:
            candidate_position = +1 if action == ENTER_SHORT else -1
            if not bool(row.get("can_enter", True)):
                action = WAIT
            else:
                candidate_fill = fill_price(row, position=candidate_position, is_entry=True)
                planned_contracts = choose_contracts(
                    row=row,
                    capital=capital,
                    position=candidate_position,
                    entry_fill=candidate_fill,
                    args=args,
                )
                if planned_contracts < 1:
                    margin_blocks += 1
                    action = WAIT

        obs, reward, terminated, truncated, info = env.step(action)
        actual_action = int(info["action"])
        done = bool(terminated or truncated)
        total_reward += float(reward)

        if actual_action in {ENTER_SHORT, ENTER_LONG} and position == 0:
            position = +1 if actual_action == ENTER_SHORT else -1
            entry_fill = fill_price(row, position=position, is_entry=True)
            contracts = max(1, planned_contracts)
            episode_contracts = contracts
            entry_step = step
            entries += 1
            commission = STRADDLE_LEGS * args.commission_per_option_contract * contracts
            capital -= commission
            trade_rows.append(
                trade_record(
                    slot.slot_id,
                    ACTION_NAMES[actual_action],
                    step,
                    row,
                    position,
                    contracts,
                    entry_fill,
                    commission,
                    capital,
                    -commission,
                )
            )

        if actual_action == EXIT and position != 0 and entry_fill is not None:
            exit_fill = fill_price(row, position=position, is_entry=False)
            gross_pnl = signed_pnl(position, entry_fill, exit_fill, args.contract_multiplier, contracts)
            commission = STRADDLE_LEGS * args.commission_per_option_contract * contracts
            dollar_pnl = gross_pnl - commission
            capital += dollar_pnl
            exits += 1
            trade_rows.append(
                trade_record(
                    slot.slot_id,
                    "EXIT",
                    step,
                    row,
                    position,
                    contracts,
                    exit_fill,
                    commission,
                    capital,
                    dollar_pnl,
                    entry_fill,
                    gross_pnl,
                    entry_step,
                )
            )
            position = 0
            contracts = 0
            entry_fill = None

        if done and position != 0 and entry_fill is not None:
            settle = terminal_value(row)
            gross_pnl = signed_pnl(position, entry_fill, settle, args.contract_multiplier, contracts)
            commission = STRADDLE_LEGS * args.commission_per_option_contract * contracts
            dollar_pnl = gross_pnl - commission
            capital += dollar_pnl
            trade_rows.append(
                trade_record(
                    slot.slot_id,
                    "AUTO_SETTLE",
                    step,
                    row,
                    position,
                    contracts,
                    settle,
                    commission,
                    capital,
                    dollar_pnl,
                    entry_fill,
                    gross_pnl,
                    entry_step,
                )
            )
            position = 0
            contracts = 0
            entry_fill = None

        slot_nav = mark_to_market_nav(
            capital,
            position,
            entry_fill,
            row,
            contracts,
            args.contract_multiplier,
        )
        maintenance_requirement = (
            float(row["underlying_price"])
            * args.contract_multiplier
            * max(contracts, 0)
            * args.maintenance_margin_rate
            if position == +1
            else 0.0
        )
        if position == +1 and maintenance_requirement > 0 and slot_nav < maintenance_requirement:
            exit_fill = fill_price(row, position=position, is_entry=False)
            gross_pnl = signed_pnl(position, entry_fill, exit_fill, args.contract_multiplier, contracts)
            commission = STRADDLE_LEGS * args.commission_per_option_contract * contracts
            dollar_pnl = gross_pnl - commission
            capital = float(slot_nav) - commission
            slot_nav = capital
            maintenance_liquidation = True
            done = True
            trade_rows.append(
                trade_record(
                    slot.slot_id,
                    "MARGIN_LIQUIDATION",
                    step,
                    row,
                    position,
                    contracts,
                    exit_fill,
                    commission,
                    capital,
                    dollar_pnl,
                    entry_fill,
                    gross_pnl,
                    entry_step,
                )
            )
            position = 0
            contracts = 0
            entry_fill = None

        nav_rows.append(
            {
                "quote_datetime": row["quote_datetime"],
                "date": str(row["date"]),
                "slot_id": slot.slot_id,
                "slot_nav": float(slot_nav),
                "slot_capital": float(capital),
                "slot_active_episode": True,
                "position": int(position),
                "contracts": int(contracts),
                "action": ACTION_NAMES.get(actual_action, str(actual_action)),
            }
        )
        step += 1

    final_row = episode.iloc[min(step, len(episode) - 1)] if step < len(episode) else episode.iloc[-1]
    nav_rows.append(
        {
            "quote_datetime": final_row["quote_datetime"],
            "date": str(final_row["date"]),
            "slot_id": slot.slot_id,
            "slot_nav": float(capital),
            "slot_capital": float(capital),
            "slot_active_episode": False,
            "position": 0,
            "contracts": 0,
            "action": "EPISODE_END",
        }
    )
    new_slot = SlotState(
        slot_id=slot.slot_id,
        capital=float(capital),
        available_date=str(final_row["date"]),
        disabled=bool(slot.disabled),
    )
    summary = {
        "slot_id": slot.slot_id,
        "start_date": str(episode.iloc[0]["date"]),
        "end_date": str(final_row["date"]),
        "expiration": str(episode.iloc[0]["expiration"]),
        "strike": float(episode.iloc[0]["strike"]),
        "start_moneyness": float(episode.iloc[0]["moneyness"]),
        "starting_capital": float(slot.capital),
        "ending_capital": float(capital),
        "dollar_pnl": float(capital - slot.capital),
        "contracts": int(episode_contracts),
        "entries": int(entries),
        "exits": int(exits),
        "maintenance_liquidation": bool(maintenance_liquidation),
        "margin_blocks": int(margin_blocks),
        "total_reward": float(total_reward),
    }
    return new_slot, nav_rows, trade_rows, summary


def signed_pnl(
    position: int,
    entry_fill: float | None,
    exit_fill: float,
    contract_multiplier: float,
    contracts: int,
) -> float:
    if entry_fill is None:
        return 0.0
    if position == +1:
        return float((entry_fill - exit_fill) * contract_multiplier * contracts)
    return float((exit_fill - entry_fill) * contract_multiplier * contracts)


def trade_record(
    slot_id: int,
    event: str,
    step: int,
    row: pd.Series,
    position: int,
    contracts: int,
    fill: float,
    commission: float,
    capital: float,
    dollar_pnl: float,
    entry_fill: float | None = None,
    gross_pnl: float | None = None,
    entry_step: int | None = None,
) -> dict:
    record = {
        "slot_id": slot_id,
        "event": event,
        "step": int(step),
        "quote_datetime": str(row["quote_datetime"]),
        "date": str(row["date"]),
        "expiration": str(row["expiration"]),
        "strike": float(row["strike"]),
        "moneyness": float(row["moneyness"]),
        "position": int(position),
        "contracts": int(contracts),
        "fill_price": float(fill),
        "commission": float(commission),
        "capital_after": float(capital),
        "dollar_pnl": float(dollar_pnl),
        "quote_available": bool(row.get("quote_available", True)),
        "synthetic_bar": bool(row.get("synthetic_bar", False)),
    }
    if entry_fill is not None:
        record["entry_fill_price"] = float(entry_fill)
    if gross_pnl is not None:
        record["gross_option_pnl"] = float(gross_pnl)
    if entry_step is not None:
        record["entry_step"] = int(entry_step)
    return record


def run_portfolio(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes, norm_stats = load_episode_split(
        args.artifact_dir,
        args.oos_start_year,
        args.oos_end_year,
    )
    candidates = select_daily_candidates(
        episodes,
        args.oos_start_year,
        args.oos_end_year,
        args.target_ttm,
    )
    if args.max_start_dates is not None:
        candidates = candidates[: args.max_start_dates]
    if not candidates:
        raise ValueError("No OOS candidates found")

    policy = load_policy(args)
    first_date = candidates[0][0]
    initial_per_slot = args.initial_capital / args.max_slots
    slots = [
        SlotState(slot_id=i, capital=initial_per_slot, available_date=first_date)
        for i in range(args.max_slots)
    ]
    all_nav_rows: list[dict] = []
    all_trade_rows: list[dict] = []
    episode_summaries: list[dict] = []
    skipped_rows: list[dict] = []

    for start_i, (start_date, episode_id, episode) in enumerate(candidates):
        if start_i == 0 or (start_i + 1) % args.progress_every == 0 or start_i + 1 == len(candidates):
            print(f"V3R OOS start {start_i + 1}/{len(candidates)} date={start_date}", flush=True)
        free_slots = [
            slot
            for slot in slots
            if not slot.disabled and slot.capital > args.slot_capital_floor and str(slot.available_date) <= start_date
        ]
        if not free_slots:
            skipped_rows.append({"start_date": start_date, "episode_id": episode_id, "reason": "no_free_slot"})
            continue
        slot = sorted(free_slots, key=lambda x: (x.available_date, x.slot_id))[0]
        new_slot, nav_rows, trade_rows, summary = simulate_slot_episode(
            policy, episode, norm_stats, slot, args
        )
        slots[slot.slot_id] = new_slot
        for row in nav_rows:
            row["episode_id"] = episode_id
            row["start_date"] = start_date
        for row in trade_rows:
            row["episode_id"] = episode_id
            row["start_date"] = start_date
        summary["episode_id"] = episode_id
        summary["daily_start_index"] = start_i
        all_nav_rows.extend(nav_rows)
        all_trade_rows.extend(trade_rows)
        episode_summaries.append(summary)

    start_timestamp = candidates[0][2].iloc[0]["quote_datetime"]
    portfolio_nav = build_portfolio_nav(
        all_nav_rows,
        initial_capital=args.initial_capital,
        max_slots=args.max_slots,
        start_timestamp=start_timestamp,
    )
    nav_csv = output_dir / "oos_portfolio_nav.csv"
    slot_nav_csv = output_dir / "oos_slot_nav.csv"
    trades_csv = output_dir / "oos_slot_trades.csv"
    episodes_csv = output_dir / "oos_slot_episodes.csv"
    skipped_csv = output_dir / "oos_skipped_starts.csv"
    portfolio_nav.to_csv(nav_csv, index=False)
    pd.DataFrame(all_nav_rows).to_csv(slot_nav_csv, index=False)
    pd.DataFrame(all_trade_rows).to_csv(trades_csv, index=False)
    pd.DataFrame(episode_summaries).to_csv(episodes_csv, index=False)
    pd.DataFrame(skipped_rows).to_csv(skipped_csv, index=False)

    episode_df = pd.DataFrame(episode_summaries)
    trade_df = pd.DataFrame(all_trade_rows)
    metrics = portfolio_metrics(
        portfolio_nav,
        initial_capital=args.initial_capital,
        bars_per_year=args.bars_per_year,
    )
    plot_paths = plot_portfolio_curves(
        portfolio_nav,
        output_dir=output_dir,
        title=args.plot_title or f"V3R {args.model_variant}",
    )
    summary = {
        "mode": "v3r_portfolio",
        "policy_loader": args.policy_loader,
        "model_variant": args.model_variant,
        "checkpoint": args.checkpoint,
        "artifact_dir": args.artifact_dir,
        "reward_mode": args.reward_mode,
        "obs_features": args.obs_features,
        "min_hold_bars": args.min_hold_bars,
        "oos_years": [args.oos_start_year, args.oos_end_year],
        **metrics,
        "max_slots": args.max_slots,
        "contract_sizing": args.contract_sizing,
        "stress_move": args.stress_move,
        "slot_risk_fraction": args.slot_risk_fraction,
        "commission_per_option_contract": args.commission_per_option_contract,
        "entry_capital_check": args.entry_capital_check,
        "short_margin_rate": args.short_margin_rate,
        "maintenance_margin_rate": args.maintenance_margin_rate,
        "daily_candidates": len(candidates),
        "started_episodes": len(episode_summaries),
        "skipped_starts": len(skipped_rows),
        "entries": int(episode_df["entries"].sum()) if "entries" in episode_df else 0,
        "exits": int(episode_df["exits"].sum()) if "exits" in episode_df else 0,
        "total_contracts": int(episode_df["contracts"].sum()) if "contracts" in episode_df else 0,
        "mean_episode_dollar_pnl": float(episode_df["dollar_pnl"].mean()) if not episode_df.empty else 0.0,
        "episode_pnl_cvar_5": positive_dollar_cvar(episode_df["dollar_pnl"]) if not episode_df.empty else 0.0,
        "maintenance_liquidations": int(episode_df["maintenance_liquidation"].sum()) if "maintenance_liquidation" in episode_df else 0,
        "margin_blocks": int(episode_df["margin_blocks"].sum()) if "margin_blocks" in episode_df else 0,
        "total_commission": float(trade_df["commission"].fillna(0.0).sum()) if "commission" in trade_df else 0.0,
        "mean_active_slots": float(portfolio_nav["active_slots"].mean()),
        "nav_csv": str(nav_csv),
        "slot_nav_csv": str(slot_nav_csv),
        "trades_csv": str(trades_csv),
        "episodes_csv": str(episodes_csv),
        "skipped_csv": str(skipped_csv),
        **plot_paths,
    }
    save_json(output_dir / "oos_portfolio_summary.json", summary)
    print("V3R OOS portfolio summary")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    if hasattr(policy, "close"):
        policy.close()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V3R simulated-quote policy.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-variant", choices=["v3", "exit", "long_short"], required=True)
    parser.add_argument(
        "--policy-loader",
        choices=["numpy", "rllib"],
        default="numpy",
        help="Use numpy for PPO FCNet checkpoints; use rllib for DQN or other RLlib checkpoints.",
    )
    parser.add_argument(
        "--use-action-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return dict observations with action_mask; required for masked DQN checkpoints.",
    )
    parser.add_argument("--oos-start-year", type=int, default=2024)
    parser.add_argument("--oos-end-year", type=int, default=2026)
    parser.add_argument("--max-start-dates", type=int, default=None)
    parser.add_argument("--initial-capital", type=float, default=3_000_000.0)
    parser.add_argument("--max-slots", type=int, default=20)
    parser.add_argument("--contract-multiplier", type=float, default=100.0)
    parser.add_argument("--contract-sizing", choices=["fixed_one", "stress_budget"], default="stress_budget")
    parser.add_argument("--stress-move", type=float, default=0.15)
    parser.add_argument("--slot-risk-fraction", type=float, default=0.50)
    parser.add_argument("--max-contracts-per-slot", type=int, default=None)
    parser.add_argument("--commission-per-option-contract", type=float, default=1.50)
    parser.add_argument("--entry-capital-check", choices=["none", "premium", "short_margin"], default="short_margin")
    parser.add_argument("--short-margin-rate", type=float, default=0.20)
    parser.add_argument("--maintenance-margin-rate", type=float, default=0.15)
    parser.add_argument("--slot-capital-floor", type=float, default=0.0)
    parser.add_argument("--target-ttm", type=int, default=20)
    parser.add_argument("--bars-per-year", type=int, default=252 * 13)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--reward-lambda", type=float, default=0.2)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--beta-vrp", type=float, default=0.5)
    parser.add_argument("--reward-mode", choices=["shaped", "direct", "exit_risk"], default="shaped")
    parser.add_argument("--obs-features", choices=["base", "position"], default="base")
    parser.add_argument("--min-hold-bars", type=int, default=0)
    parser.add_argument("--plot-title", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_portfolio(parse_args())
