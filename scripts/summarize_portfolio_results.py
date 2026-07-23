from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_FIELDS = [
    "label",
    "mode",
    "benchmark",
    "checkpoint",
    "oos_years",
    "initial_capital",
    "final_nav",
    "total_return",
    "annual_return",
    "annual_vol",
    "sharpe",
    "calmar",
    "max_drawdown",
    "min_nav",
    "started_episodes",
    "entries",
    "total_contracts",
    "mean_contracts_per_entry",
    "skipped_starts",
    "margin_blocks",
    "liquidated_slots",
    "maintenance_liquidations",
    "disabled_slots_at_end",
    "mean_episode_dollar_pnl",
    "episode_pnl_cvar_5",
    "mean_active_slots",
    "contract_sizing",
    "stress_move",
    "slot_risk_fraction",
    "max_contracts_per_slot",
    "commission_per_option_contract",
    "total_commission",
    "entry_capital_check",
    "short_margin_rate",
    "maintenance_margin_rate",
    "settlement_mode",
    "fill_mode",
    "summary_path",
]


def load_summary(path: Path, label: str | None = None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["label"] = label or path.parent.name
    payload["summary_path"] = str(path)
    if "entries" not in payload and payload.get("episodes_csv"):
        episodes_csv = Path(payload["episodes_csv"])
        if not episodes_csv.is_absolute():
            episodes_csv = path.parents[3] / episodes_csv if str(episodes_csv).startswith("artifacts_") else path.parent / episodes_csv
        if episodes_csv.exists():
            episode_df = pd.read_csv(episodes_csv)
            if "entries" in episode_df:
                payload["entries"] = int(episode_df["entries"].sum())
    payload["entries"] = payload.get("entries", payload.get("started_episodes"))
    payload["maintenance_liquidations"] = payload.get("maintenance_liquidations", 0)
    payload["margin_blocks"] = payload.get("margin_blocks", 0)
    payload["liquidated_slots"] = payload.get("liquidated_slots", 0)
    return payload


def format_markdown(df: pd.DataFrame) -> str:
    display = df.copy()
    percent_cols = ["total_return", "annual_return", "annual_vol", "max_drawdown"]
    money_cols = ["initial_capital", "final_nav", "min_nav", "mean_episode_dollar_pnl"]
    for col in percent_cols:
        if col in display:
            display[col] = display[col].map(
                lambda x: "" if pd.isna(x) else f"{float(x):.2%}"
            )
    for col in money_cols:
        if col in display:
            display[col] = display[col].map(
                lambda x: "" if pd.isna(x) else f"${float(x):,.0f}"
            )
    for col in ["sharpe", "calmar"]:
        if col in display:
            display[col] = display[col].map(
                lambda x: "" if pd.isna(x) else f"{float(x):.3f}"
            )
    columns = list(display.columns)
    rows = [[str(value) for value in row] for row in display.fillna("").to_numpy()]
    widths = [
        max(len(str(col)), *(len(row[idx]) for row in rows))
        for idx, col in enumerate(columns)
    ]
    header = "| " + " | ".join(
        str(col).ljust(widths[idx]) for idx, col in enumerate(columns)
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize portfolio result JSON files.")
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "Result directories or oos_portfolio_summary.json files. "
            "Directory inputs are searched one level for oos_portfolio_summary.json."
        ),
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: list[dict[str, Any]] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.is_dir():
            candidates = list(path.glob("oos_portfolio_summary.json"))
            if not candidates:
                candidates = list(path.glob("*/oos_portfolio_summary.json"))
            for candidate in sorted(candidates):
                summaries.append(load_summary(candidate))
        else:
            summaries.append(load_summary(path))

    if not summaries:
        raise SystemExit("No summary files found.")

    df = pd.DataFrame(summaries)
    keep = [col for col in SUMMARY_FIELDS if col in df.columns]
    df = df[keep]
    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    markdown = format_markdown(df)
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
