from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


WAIT = 0
ENTER_SHORT = 1
EXIT = 2
ENTER_LONG = 3

ACTION_NAMES = {
    WAIT: "WAIT/HOLD",
    ENTER_SHORT: "ENTER_SHORT",
    EXIT: "EXIT",
    ENTER_LONG: "ENTER_LONG",
}


@dataclass
class NormStats:
    spread_mean: float
    score_mean: float
    delta_rv13_std: float


def compute_cvar(returns_window: np.ndarray | pd.Series, confidence: float = 0.05) -> float:
    values = np.asarray(returns_window, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 30:
        return 0.0
    threshold = np.percentile(values, confidence * 100)
    tail = values[values <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


class StraddleEnv(gym.Env):
    """
    Simulated-quote straddle timing environment.

    Variants:
      v3         : WAIT, ENTER_SHORT
      exit       : WAIT/HOLD, ENTER_SHORT, EXIT
      long_short : WAIT/HOLD, ENTER_SHORT, EXIT, ENTER_LONG

    Position convention:
      +1 = short straddle
      -1 = long straddle
       0 = flat
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict):
        super().__init__()
        self.data = config["episode_data"].reset_index(drop=True)
        self.norm: NormStats = config["norm_stats"]
        self.variant = str(config.get("model_variant", "v3"))
        self.alpha = float(config.get("alpha", 1.0))
        self.lambda_ = float(config.get("lambda_", 0.1))
        self.mu = float(config.get("mu", 0.05))
        self.beta_vrp = float(config.get("beta_vrp", 0.0))
        self.exit_bonus = float(config.get("exit_bonus", 0.0))
        self.obs_clip = float(config.get("obs_clip", 20.0))
        self.invalid_action_mode = str(config.get("invalid_action_mode", "remap"))
        self.invalid_action_penalty = float(config.get("invalid_action_penalty", 0.0))
        self.use_action_mask_obs = bool(config.get("use_action_mask_obs", False))
        self.reward_mode = str(config.get("reward_mode", "shaped"))
        self.obs_features = str(config.get("obs_features", "base"))
        self.min_hold_bars = int(config.get("min_hold_bars", 0))

        if self.variant not in {"v3", "exit", "long_short"}:
            raise ValueError("model_variant must be one of: v3, exit, long_short")
        if self.invalid_action_mode not in {"remap", "penalty"}:
            raise ValueError("invalid_action_mode must be one of: remap, penalty")
        if self.reward_mode not in {"shaped", "direct", "exit_risk"}:
            raise ValueError("reward_mode must be one of: shaped, direct, exit_risk")
        if self.obs_features not in {"base", "position"}:
            raise ValueError("obs_features must be one of: base, position")
        if len(self.data) == 0:
            raise ValueError("episode_data must contain at least one row")

        n_actions = {"v3": 2, "exit": 3, "long_short": 4}[self.variant]
        self.obs_dim = 11 if self.obs_features == "base" else 17
        self.base_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.observation_space = self.base_observation_space
        self.action_space = spaces.Discrete(n_actions)
        if self.use_action_mask_obs:
            self.observation_space = spaces.Dict(
                {
                    "observations": self.base_observation_space,
                    "action_mask": spaces.Box(0, 1, shape=(n_actions,), dtype=np.int8),
                }
            )
        self._reset_state()

    def _reset_state(self) -> None:
        self.t = 0
        self.pos = 0
        self.entry_value: float | None = None
        self.entry_step: int | None = None
        self.entry_ttm_bars: int | None = None
        self.entry_underlying: float | None = None
        self.entry_moneyness: float | None = None
        self.pnl = 0.0
        self.realized_pnl = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._format_obs(), {"action_mask": self._get_action_mask()}

    def step(self, action: int):
        action = int(action)
        row = self.data.iloc[self.t]
        ttm_bars = int(row["ttm_bars"])
        is_last = (self.t == len(self.data) - 1) or (ttm_bars == 0)
        pos_before = self.pos
        entry_before = self.entry_value
        value_before = self._position_value(row)

        requested_action = action
        action, invalid_action = self._remap_invalid_action(action, allow_open=not is_last)
        trade_cost = 0.0
        realized = None
        value_after = value_before

        if action in {ENTER_SHORT, ENTER_LONG}:
            self.pos = +1 if action == ENTER_SHORT else -1
            self.entry_value = self._entry_value(row, self.pos)
            self.entry_step = self.t
            self.entry_ttm_bars = ttm_bars
            self.entry_underlying = float(row["underlying_price"])
            self.entry_moneyness = float(row["moneyness"])
            self.pnl = 0.0
            trade_cost = self.alpha * self._spread_rel(row)
        elif action == EXIT and self.pos != 0 and self.entry_value is not None:
            exit_value = self._exit_value(row, self.pos)
            value_after = exit_value
            realized = self._normalised_pnl(exit_value)
            self.realized_pnl += realized
            self.pos = 0
            self.entry_value = None
            self.entry_step = None
            self.entry_ttm_bars = None
            self.entry_underlying = None
            self.entry_moneyness = None
            self.pnl = 0.0
            trade_cost = self.alpha * self._spread_rel(row)

        if self.pos != 0 and self.entry_value and self.entry_value > 0:
            self.pnl = self._normalised_pnl(self._position_value(row))

        if is_last and self.pos != 0 and self.entry_value is not None:
            terminal_value = self._terminal_value(row)
            value_after = terminal_value
            realized = self._normalised_pnl(terminal_value)
            self.realized_pnl += realized
            self.pos = 0
            self.entry_value = None
            self.entry_step = None
            self.entry_ttm_bars = None
            self.entry_underlying = None
            self.entry_moneyness = None
            self.pnl = 0.0

        pnl_for_reward = float(realized if realized is not None else self.pnl)
        if realized is None:
            value_after = self._position_value(row)
        delta_v = 0.0
        if pos_before != 0 and entry_before and entry_before > 0:
            delta_v = (value_before - value_after) * pos_before / entry_before

        reward = self._reward(
            action=action,
            delta_v=delta_v,
            trade_cost=trade_cost,
            pos_before=pos_before,
            pnl=pnl_for_reward,
            row=row,
            ttm_bars=ttm_bars,
            is_last=is_last,
        )
        if invalid_action and self.invalid_action_mode == "penalty":
            reward += self.invalid_action_penalty

        terminated = bool(is_last or action == EXIT)
        truncated = False
        self.t += 1
        obs = self._format_obs()
        info = {
            "pos": self.pos,
            "pnl": pnl_for_reward,
            "episode_pnl": self.realized_pnl + (self.pnl if self.pos != 0 else 0.0),
            "ttm_bars": ttm_bars,
            "requested_action": requested_action,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "invalid_action": bool(invalid_action),
            "action_mask": self._get_action_mask(),
        }
        return obs, float(reward), terminated, truncated, info

    def _reward(
        self,
        action: int,
        delta_v: float,
        trade_cost: float,
        pos_before: int,
        pnl: float,
        row: pd.Series,
        ttm_bars: int,
        is_last: bool,
    ) -> float:
        if self.reward_mode == "direct":
            return float(delta_v - trade_cost)

        if self.reward_mode == "exit_risk":
            downside_penalty = max(0.0, -float(delta_v)) if pos_before != 0 else 0.0
            loss_penalty = max(0.0, -float(pnl)) if pos_before != 0 else 0.0
            late_penalty = float(ttm_bars <= 65 and pos_before != 0)
            drift_penalty = 0.0
            if pos_before != 0 and self.entry_moneyness is not None:
                drift_penalty = abs(float(row["moneyness"]) - float(self.entry_moneyness))
            cvar = float(row.get("straddle_cvar", row.get("cvar", 0.0)))
            tail_penalty = self.lambda_ * (cvar + downside_penalty + 0.25 * loss_penalty)
            timing_penalty = self.mu * (late_penalty + drift_penalty)
            return float(delta_v - trade_cost - tail_penalty - timing_penalty)

        if is_last or action == EXIT:
            return float(pnl - trade_cost)
        cvar = float(row.get("straddle_cvar", row.get("cvar", 0.0)))
        vrp = float(row.get("vrp", 0.0))
        short_risk = self.lambda_ * cvar if pos_before == +1 else 0.0
        late_penalty = self.mu * float(ttm_bars <= 65 and pos_before != 0)
        vrp_term = self.beta_vrp * pos_before * vrp if pos_before != 0 else 0.0
        return float(delta_v + vrp_term - trade_cost - short_risk - late_penalty)

    def _entry_value(self, row: pd.Series, position: int) -> float:
        if position == +1:
            return float(row["straddle_bid"])
        return float(row.get("sim_entry_long_cost", row["straddle_ask"]))

    def _exit_value(self, row: pd.Series, position: int) -> float:
        if bool(row.get("quote_available", True)):
            return float(row["straddle_ask"] if position == +1 else row["straddle_bid"])
        if position == +1:
            return float(row.get("sim_short_exit_cost", row["straddle_ask"]))
        return float(row.get("sim_long_exit_value", row["straddle_bid"]))

    def _terminal_value(self, row: pd.Series) -> float:
        return float(abs(float(row["underlying_price"]) - float(row["strike"])))

    def _position_value(self, row: pd.Series) -> float:
        if self.pos == 0:
            return float(row["straddle_mid"])
        return self._exit_value(row, self.pos)

    def _normalised_pnl(self, current_value: float) -> float:
        if not self.entry_value or self.entry_value <= 0:
            return 0.0
        if self.pos == +1:
            return float((self.entry_value - current_value) / self.entry_value)
        if self.pos == -1:
            return float((current_value - self.entry_value) / self.entry_value)
        return 0.0

    def _spread_rel(self, row: pd.Series) -> float:
        return max(0.0, float(row.get("spread_rel", 0.0)))

    def _holding_bars(self) -> int:
        if self.pos == 0 or self.entry_step is None:
            return 0
        return max(0, int(self.t - self.entry_step))

    def _exit_allowed(self, row: pd.Series, *, is_last: bool = False) -> bool:
        if self.pos == 0:
            return False
        if not bool(row.get("can_exit", False)):
            return False
        if is_last:
            return True
        return self._holding_bars() >= self.min_hold_bars

    def _remap_invalid_action(self, action: int, allow_open: bool = True) -> tuple[int, bool]:
        if action not in ACTION_NAMES or action >= self.action_space.n:
            return WAIT, True
        can_enter = bool(self.data.iloc[self.t].get("can_enter", True))
        can_exit = bool(self.data.iloc[self.t].get("can_exit", False))
        if action in {ENTER_SHORT, ENTER_LONG} and (self.pos != 0 or not allow_open or not can_enter):
            return WAIT, True
        if action == ENTER_LONG and self.variant != "long_short":
            return WAIT, True
        is_last = self.t == len(self.data) - 1 or int(self.data.iloc[self.t]["ttm_bars"]) == 0
        if action == EXIT and not (can_exit and self._exit_allowed(self.data.iloc[self.t], is_last=is_last)):
            return WAIT, True
        return action, False

    def _get_action_mask(self) -> np.ndarray:
        if self.t >= len(self.data):
            return np.zeros(self.action_space.n, dtype=np.int8)
        row = self.data.iloc[self.t]
        ttm_bars = int(row["ttm_bars"])
        can_enter = bool(row.get("can_enter", True)) and ttm_bars > 0
        is_last = self.t == len(self.data) - 1 or ttm_bars == 0
        can_exit = self._exit_allowed(row, is_last=is_last)
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        mask[WAIT] = 1
        if self.pos == 0 and can_enter:
            mask[ENTER_SHORT] = 1
            if self.variant == "long_short":
                mask[ENTER_LONG] = 1
        if self.pos != 0 and can_exit and self.variant in {"exit", "long_short"}:
            mask[EXIT] = 1
        return mask

    def _get_obs(self) -> np.ndarray:
        if self.t >= len(self.data):
            return np.zeros(self.obs_dim, dtype=np.float32)
        row = self.data.iloc[self.t]
        base = [
            float(row["ttm_bars"]) / 260.0,
            float(row["tod"]),
            float(row["vrp"]) / 10.0,
            float(row["spread_rel"]) / (self.norm.spread_mean + 1e-8),
            float(row["score"]) / (self.norm.score_mean + 1e-8),
            float(row["delta_rv13"]) / (self.norm.delta_rv13_std + 1e-8),
            float(row.get("quote_available", True)),
            float(row.get("synthetic_bar", False)),
            float(row.get("sim_time_value", 0.0)) / (float(row["straddle_mid"]) + 1e-8),
            float(self.pos),
            float(self.pnl if self.pos != 0 else 0.0),
        ]
        if self.obs_features == "base":
            obs = np.array(base, dtype=np.float32)
        else:
            current_value = self._position_value(row) if self.pos != 0 else float(row["straddle_mid"])
            entry_value = float(self.entry_value or 0.0)
            entry_underlying = float(self.entry_underlying or float(row["underlying_price"]))
            entry_moneyness = float(self.entry_moneyness or float(row["moneyness"]))
            entry_ttm = float(self.entry_ttm_bars or 0)
            position_features = [
                float(self._holding_bars()) / 260.0,
                entry_ttm / 260.0,
                float(row["ttm_bars"]) / (entry_ttm + 1e-8) if entry_ttm > 0 else 0.0,
                current_value / (entry_value + 1e-8) if entry_value > 0 else 0.0,
                (float(row["underlying_price"]) - entry_underlying) / (entry_underlying + 1e-8),
                float(row["moneyness"]) - entry_moneyness,
            ]
            obs = np.array(base + position_features, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=self.obs_clip, neginf=-self.obs_clip)
        return np.clip(obs, -self.obs_clip, self.obs_clip).astype(np.float32)

    def _format_obs(self):
        obs = self._get_obs()
        if not self.use_action_mask_obs:
            return obs
        return {
            "observations": obs,
            "action_mask": self._get_action_mask().astype(np.int8),
        }
