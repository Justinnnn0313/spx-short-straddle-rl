from __future__ import annotations

import json
import pickle
import re
import socket
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import ray

from env import ENTER_SHORT, StraddleEnv


DEFAULT_ARTIFACT_DIR = Path("artifacts_v3r_simquote")
DEFAULT_MODEL_DIR = DEFAULT_ARTIFACT_DIR / "rl_models"

_ARTIFACT_CACHE: dict[str, tuple[list[pd.DataFrame], Any]] = {}


def load_artifacts(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR):
    artifact_dir = Path(artifact_dir)
    cache_key = str(artifact_dir.resolve())
    if cache_key in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[cache_key]

    with (artifact_dir / "episodes.pkl").open("rb") as f:
        episodes = pickle.load(f)
    with (artifact_dir / "norm_stats.pkl").open("rb") as f:
        norm_stats = pickle.load(f)

    _ARTIFACT_CACHE[cache_key] = (episodes, norm_stats)
    return episodes, norm_stats


def episode_start_year(episode: pd.DataFrame) -> int:
    return int(episode.iloc[0]["year"])


def filter_episodes_by_year(
    episodes: list[pd.DataFrame],
    start_year: int,
    end_year: int,
) -> list[pd.DataFrame]:
    return [
        ep
        for ep in episodes
        if start_year <= episode_start_year(ep) <= end_year
    ]


def load_episode_split(
    artifact_dir: str | Path,
    start_year: int,
    end_year: int,
) -> tuple[list[pd.DataFrame], Any]:
    episodes, norm_stats = load_artifacts(artifact_dir)
    split = filter_episodes_by_year(episodes, start_year, end_year)
    if not split:
        raise ValueError(
            f"No episodes found for start-year range {start_year}-{end_year}."
        )
    return split, norm_stats


def episode_split_stats(episodes: list[pd.DataFrame]) -> dict[str, float]:
    lengths = np.array([len(ep) for ep in episodes], dtype=np.int64)
    return {
        "episodes": float(len(episodes)),
        "total_bars": float(lengths.sum()) if len(lengths) else 0.0,
        "mean_bars": float(lengths.mean()) if len(lengths) else 0.0,
        "median_bars": float(np.median(lengths)) if len(lengths) else 0.0,
        "min_bars": float(lengths.min()) if len(lengths) else 0.0,
        "max_bars": float(lengths.max()) if len(lengths) else 0.0,
    }


def audit_episodes(
    episodes: list[pd.DataFrame],
    *,
    require_complete_expiry: bool = True,
) -> dict[str, float]:
    rows = []
    bad_counts = {
        "duplicate_timestamps": 0,
        "non_monotonic_time": 0,
        "nan_values": 0,
        "nonfinite_core": 0,
        "bad_mid": 0,
        "bad_bid": 0,
        "ask_lt_bid": 0,
        "bad_spread_rel": 0,
        "incomplete_expiry": 0,
    }
    core_cols = [
        "straddle_mid",
        "straddle_bid",
        "straddle_ask",
        "spread_rel",
        "straddle_cvar",
        "short_straddle_pnl_ret",
        "rv",
        "vrp",
        "score",
    ]

    for episode in episodes:
        end = episode.iloc[-1]
        if require_complete_expiry and not (
            str(end["date"]) == str(end["expiration"])
            and float(end["ttm_trading"]) == 0.0
        ):
            bad_counts["incomplete_expiry"] += 1
        bad_counts["duplicate_timestamps"] += int(
            episode["quote_datetime"].duplicated().sum()
        )
        bad_counts["non_monotonic_time"] += int(
            not episode["quote_datetime"].is_monotonic_increasing
        )
        bad_counts["nan_values"] += int(episode.isna().sum().sum())

        core = episode[core_cols].to_numpy(dtype=np.float64)
        bad_counts["nonfinite_core"] += int((~np.isfinite(core)).sum())
        bad_counts["bad_mid"] += int((episode["straddle_mid"] <= 0).sum())
        bad_counts["bad_bid"] += int((episode["straddle_bid"] < 0).sum())
        bad_counts["ask_lt_bid"] += int(
            (episode["straddle_ask"] < episode["straddle_bid"]).sum()
        )
        bad_counts["bad_spread_rel"] += int(
            ((episode["spread_rel"] < 0) | (episode["spread_rel"] > 1)).sum()
        )
        rows.append(
            {
                "year": episode_start_year(episode),
                "bars": len(episode),
            }
        )

    stats = episode_split_stats(episodes)
    if rows:
        year_counts = pd.DataFrame(rows)["year"].value_counts().sort_index()
        for year, count in year_counts.items():
            stats[f"episodes_{int(year)}"] = float(count)
    stats.update({key: float(value) for key, value in bad_counts.items()})
    return stats


def raise_if_episode_audit_fails(audit: dict[str, float]) -> None:
    blocking_keys = [
        "duplicate_timestamps",
        "non_monotonic_time",
        "nan_values",
        "nonfinite_core",
        "bad_mid",
        "bad_bid",
        "ask_lt_bid",
        "bad_spread_rel",
        "incomplete_expiry",
    ]
    failures = {key: audit.get(key, 0.0) for key in blocking_keys if audit.get(key, 0.0)}
    if failures:
        details = ", ".join(f"{key}={value:g}" for key, value in failures.items())
        raise ValueError(f"Episode artifact audit failed: {details}")


class RandomEpisodeStraddleEnv(gym.Env):
    """RLlib wrapper that selects one straddle episode on every reset."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict):
        super().__init__()
        self.artifact_dir = Path(config.get("artifact_dir", DEFAULT_ARTIFACT_DIR))
        self.start_year = int(config.get("start_year", 2016))
        self.end_year = int(config.get("end_year", 2021))
        self.sampling_mode = str(config.get("sampling_mode", "random"))
        self.env_kwargs = {
            "model_variant": str(config.get("model_variant", "v3")),
            "alpha": float(config.get("alpha", 1.0)),
            "lambda_": float(config.get("lambda_", 0.1)),
            "mu": float(config.get("mu", 0.05)),
            "beta_vrp": float(config.get("beta_vrp", 0.0)),
            "invalid_action_mode": str(config.get("invalid_action_mode", "remap")),
            "invalid_action_penalty": float(config.get("invalid_action_penalty", 0.0)),
            "use_action_mask_obs": bool(config.get("use_action_mask_obs", False)),
            "reward_mode": str(config.get("reward_mode", "shaped")),
            "obs_features": str(config.get("obs_features", "base")),
            "min_hold_bars": int(config.get("min_hold_bars", 0)),
        }
        self.episodes, self.norm_stats = load_episode_split(
            self.artifact_dir,
            self.start_year,
            self.end_year,
        )
        self.rng = np.random.default_rng(config.get("seed", None))
        self._epoch_order = np.arange(len(self.episodes), dtype=np.int64)
        self._epoch_cursor = 0
        self._shuffle_epoch_order()
        self.current_env = self._make_env(self.episodes[0])
        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space

        if self.sampling_mode not in {"random", "sequential", "epoch_shuffle"}:
            raise ValueError(
                "sampling_mode must be one of: random, sequential, epoch_shuffle"
            )

    def _shuffle_epoch_order(self) -> None:
        self.rng.shuffle(self._epoch_order)
        self._epoch_cursor = 0

    def _next_episode_index(self) -> int:
        if self.sampling_mode == "random":
            return int(self.rng.integers(0, len(self.episodes)))

        if self._epoch_cursor >= len(self._epoch_order):
            if self.sampling_mode == "epoch_shuffle":
                self._shuffle_epoch_order()
            else:
                self._epoch_cursor = 0

        idx = int(self._epoch_order[self._epoch_cursor])
        self._epoch_cursor += 1
        return idx

    def _make_env(self, episode: pd.DataFrame) -> StraddleEnv:
        return StraddleEnv(
            {
                "episode_data": episode,
                "norm_stats": self.norm_stats,
                **self.env_kwargs,
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._shuffle_epoch_order()
        idx = self._next_episode_index()
        self.current_env = self._make_env(self.episodes[idx])
        return self.current_env.reset(seed=seed, options=options)

    def step(self, action: int):
        return self.current_env.step(action)


def extract_action(action: Any) -> int:
    if isinstance(action, tuple):
        action = action[0]
    if isinstance(action, np.ndarray):
        return int(action.item())
    return int(action)


def compute_policy_action(algo: Any, obs: np.ndarray, explore: bool = False) -> int:
    action = algo.compute_single_action(obs, explore=explore)
    return extract_action(action)


def evaluate_algorithm(
    algo: Any,
    episodes: list[pd.DataFrame],
    norm_stats: Any,
    env_kwargs: dict | None = None,
    max_episodes: int | None = None,
    episode_selection: str = "head",
    explore: bool = False,
    progress_every: int | None = None,
    progress_label: str = "eval",
) -> pd.DataFrame:
    env_kwargs = env_kwargs or {}
    rows: list[dict] = []
    selected = select_eval_episodes(episodes, max_episodes, episode_selection)
    total = len(selected)

    for episode_id, episode in enumerate(selected):
        if progress_every and (
            episode_id == 0
            or (episode_id + 1) % progress_every == 0
            or episode_id + 1 == total
        ):
            print(f"{progress_label}: episode {episode_id + 1}/{total}")

        env = StraddleEnv(
            {
                "episode_data": episode,
                "norm_stats": norm_stats,
                **env_kwargs,
            }
        )
        obs, _info = env.reset()
        done = False
        total_reward = 0.0
        n_trades = 0
        last_info: dict[str, Any] = {}

        while not done:
            action = compute_policy_action(algo, obs, explore=explore)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            if int(info["action"]) == ENTER_SHORT:
                n_trades += 1
            last_info = info

        rows.append(
            {
                "episode_id": episode_id,
                "start_date": str(episode.iloc[0]["date"]),
                "end_date": str(episode.iloc[-1]["date"]),
                "start_year": episode_start_year(episode),
                "expiration": str(episode.iloc[0]["expiration"]),
                "strike": float(episode.iloc[0]["strike"]),
                "bars": int(len(episode)),
                "total_reward": float(total_reward),
                "final_pnl": float(
                    last_info.get("episode_pnl", last_info.get("pnl", 0.0))
                ),
                "n_trades": int(n_trades),
            }
        )

    return pd.DataFrame(rows)


def select_eval_episodes(
    episodes: list[pd.DataFrame],
    max_episodes: int | None,
    mode: str = "head",
) -> list[pd.DataFrame]:
    if max_episodes is None or max_episodes >= len(episodes):
        return episodes
    if max_episodes <= 0:
        return []
    if mode == "head":
        return episodes[:max_episodes]
    if mode == "even":
        indices = np.linspace(0, len(episodes) - 1, max_episodes, dtype=np.int64)
        return [episodes[int(idx)] for idx in indices]
    raise ValueError("episode_selection must be one of: head, even")


def positive_left_tail_cvar(values: pd.Series, confidence: float = 0.05) -> float:
    clean = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    threshold = clean.quantile(confidence)
    tail = clean[clean <= threshold]
    return float(-tail.mean()) if len(tail) else 0.0


def summarize_evaluation(results: pd.DataFrame) -> dict[str, float]:
    if results.empty:
        return {}

    pnl = results["final_pnl"].astype(float)
    reward = results["total_reward"].astype(float)
    pnl_std = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    reward_std = float(reward.std(ddof=1)) if len(reward) > 1 else 0.0

    return {
        "episodes": float(len(results)),
        "mean_final_pnl": float(pnl.mean()),
        "median_final_pnl": float(pnl.median()),
        "std_final_pnl": pnl_std,
        "pnl_cvar_5": positive_left_tail_cvar(pnl, confidence=0.05),
        "mean_total_reward": float(reward.mean()),
        "std_total_reward": reward_std,
        "reward_sharpe_like": (
            float(reward.mean() / reward_std) if reward_std > 0 else 0.0
        ),
        "mean_trades": float(results["n_trades"].mean()),
    }


def validation_score(summary: dict[str, float]) -> float:
    return (
        summary.get("mean_final_pnl", 0.0)
        - 0.5 * summary.get("pnl_cvar_5", 0.0)
        - 0.001 * summary.get("mean_trades", 0.0)
    )


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def checkpoint_path(save_result: Any, fallback_dir: str | Path | None = None) -> str:
    checkpoint = getattr(save_result, "checkpoint", None)
    path = getattr(checkpoint, "path", None)

    if path is None and isinstance(save_result, (str, Path)):
        path = str(save_result)

    if path is None:
        match = re.search(r"path=([^),]+)", str(save_result))
        if match:
            path = match.group(1)

    if path is None and fallback_dir is not None:
        candidate = Path(fallback_dir)
        if (candidate / "rllib_checkpoint.json").exists():
            path = str(candidate)

    if path is None:
        raise ValueError(f"Could not extract checkpoint path from {save_result!r}")
    return str(path)


def init_ray_runtime(
    ray_temp_dir: str | Path | None = None,
    ray_local_mode: bool = False,
    object_store_memory_bytes: int | None = None,
    memory_bytes: int | None = None,
) -> None:
    hostname = socket.gethostname()
    try:
        hostname.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            "Ray/RLlib cannot start on this Windows machine because the computer "
            f"name contains non-ASCII characters: {hostname!r}. Rename the "
            "computer to an ASCII-only name, such as 'Jianning-PC', restart "
            "Windows, then rerun the script."
        ) from exc

    init_kwargs = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "local_mode": bool(ray_local_mode),
        "_node_ip_address": "127.0.0.1",
        "_node_name": "local-node",
    }
    if ray_temp_dir:
        temp_dir = Path(ray_temp_dir).resolve()
        temp_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs["_temp_dir"] = str(temp_dir)
    if object_store_memory_bytes is not None:
        if object_store_memory_bytes <= 0:
            raise ValueError("object_store_memory_bytes must be positive")
        init_kwargs["object_store_memory"] = int(object_store_memory_bytes)
    if memory_bytes is not None:
        if memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")
        init_kwargs["_memory"] = int(memory_bytes)
    ray.init(**init_kwargs)
