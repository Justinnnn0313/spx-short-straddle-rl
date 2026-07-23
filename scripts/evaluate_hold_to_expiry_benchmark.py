from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "spx_straddle_rl"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from env import ENTER_SHORT, WAIT  # noqa: E402
from test_agent import run_portfolio  # noqa: E402


class HoldToExpiryPolicy:
    def compute_single_action(self, obs, explore: bool = False) -> int:
        del explore
        if isinstance(obs, dict):
            mask = obs["action_mask"]
            features = obs["observations"]
        else:
            mask = None
            features = obs

        pos = float(features[9]) if len(features) > 9 else 0.0
        if abs(pos) < 0.5 and (mask is None or mask[ENTER_SHORT] == 1):
            return ENTER_SHORT
        return WAIT


def patch_policy_loader() -> None:
    import test_agent

    def _load_policy(_args):
        return HoldToExpiryPolicy()

    test_agent.load_policy = _load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed-entry hold-to-expiry benchmark.")
    parser.add_argument("--artifact-dir", default="artifacts_v3r_simquote")
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--plot-title", default="Hold-to-Expiry Benchmark OOS")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    patch_policy_loader()
    portfolio_args = argparse.Namespace(
        artifact_dir=args.artifact_dir,
        checkpoint="hold_to_expiry_rule",
        output_dir=args.output_dir,
        model_variant="exit",
        policy_loader="hold_to_expiry",
        use_action_mask=True,
        oos_start_year=args.oos_start_year,
        oos_end_year=args.oos_end_year,
        max_start_dates=args.max_start_dates,
        initial_capital=args.initial_capital,
        max_slots=args.max_slots,
        contract_multiplier=args.contract_multiplier,
        contract_sizing=args.contract_sizing,
        stress_move=args.stress_move,
        slot_risk_fraction=args.slot_risk_fraction,
        max_contracts_per_slot=args.max_contracts_per_slot,
        commission_per_option_contract=args.commission_per_option_contract,
        entry_capital_check=args.entry_capital_check,
        short_margin_rate=args.short_margin_rate,
        maintenance_margin_rate=args.maintenance_margin_rate,
        slot_capital_floor=args.slot_capital_floor,
        target_ttm=args.target_ttm,
        bars_per_year=args.bars_per_year,
        progress_every=args.progress_every,
        alpha=1.5,
        reward_lambda=0.2,
        mu=0.05,
        beta_vrp=0.5,
        reward_mode="direct",
        obs_features="position",
        min_hold_bars=0,
        plot_title=args.plot_title,
    )
    summary = run_portfolio(portfolio_args)
    summary_path = Path(args.output_dir) / "oos_portfolio_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data["benchmark"] = "hold_to_expiry"
        summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
