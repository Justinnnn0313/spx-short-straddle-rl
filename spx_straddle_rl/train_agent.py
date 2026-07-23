from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from rl_agent_common import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MODEL_DIR,
    checkpoint_path,
    episode_split_stats,
    init_ray_runtime,
    load_episode_split,
    RandomEpisodeStraddleEnv,
    save_json,
)


ENV_NAME = "random_v3r_simquote_episode_env"
DEFAULT_OUTPUT_SENTINEL = "__default__"

FINAL_PRESETS = {
    "main": {
        "lr": 5e-6,
        "gamma": 0.997,
        "gae_lambda": 0.97,
        "clip_param": 0.1,
        "entropy_coeff": 0.001,
        "alpha": 1.5,
        "reward_lambda": 0.2,
        "mu": 0.05,
        "beta_vrp": 0.5,
        "output_dir": DEFAULT_MODEL_DIR / "final_v3r_main",
    },
    "no_tc": {
        "lr": 5e-6,
        "gamma": 0.997,
        "gae_lambda": 0.97,
        "clip_param": 0.1,
        "entropy_coeff": 0.001,
        "alpha": 0.0,
        "reward_lambda": 0.2,
        "mu": 0.05,
        "beta_vrp": 0.5,
        "output_dir": DEFAULT_MODEL_DIR / "final_v3r_no_tc",
    },
}


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def apply_training_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "custom":
        if args.output_dir == DEFAULT_OUTPUT_SENTINEL:
            args.output_dir = str(DEFAULT_MODEL_DIR / "train_run")
        return args

    preset = FINAL_PRESETS[args.preset]
    args.lr = float(preset["lr"])
    args.gamma = float(preset["gamma"])
    args.gae_lambda = float(preset["gae_lambda"])
    args.clip_param = float(preset["clip_param"])
    args.entropy_coeff = float(preset["entropy_coeff"])
    args.alpha = float(preset["alpha"])
    args.reward_lambda = float(preset["reward_lambda"])
    args.mu = float(preset["mu"])
    args.beta_vrp = float(preset["beta_vrp"])
    if args.output_dir == DEFAULT_OUTPUT_SENTINEL:
        args.output_dir = str(DEFAULT_MODEL_DIR / f"final_{args.model_variant}_simquote_lambda020")
    return args


def build_ppo_config(args: argparse.Namespace) -> PPOConfig:
    env_config = {
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
        "start_year": args.train_start_year,
        "end_year": args.train_end_year,
        "model_variant": args.model_variant,
        "alpha": args.alpha,
        "lambda_": args.reward_lambda,
        "mu": args.mu,
        "beta_vrp": args.beta_vrp,
        "seed": args.seed,
        "sampling_mode": args.sampling_mode,
    }

    return (
        PPOConfig()
        .environment(env=ENV_NAME, env_config=env_config)
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
        .rollouts(
            num_rollout_workers=args.num_rollout_workers,
            rollout_fragment_length=args.rollout_fragment_length,
        )
        .training(
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.gae_lambda,
            clip_param=args.clip_param,
            entropy_coeff=args.entropy_coeff,
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
        )
        .debugging(seed=args.seed)
    )


def run_training(args: argparse.Namespace) -> dict:
    args = apply_training_preset(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_episodes, _norm_stats = load_episode_split(
        args.artifact_dir,
        args.train_start_year,
        args.train_end_year,
    )
    split_stats = episode_split_stats(train_episodes)
    iterations_for_one_pass = int(
        np.ceil(split_stats["total_bars"] / args.train_batch_size)
    )
    if args.target_passes is not None:
        args.iterations = int(np.ceil(iterations_for_one_pass * args.target_passes))

    print(
        "Training plan: "
        f"preset={args.preset}, "
        f"model_variant={args.model_variant}, "
        f"train_years={args.train_start_year}-{args.train_end_year}, "
        f"iterations={args.iterations}, "
        f"target_passes={args.target_passes if args.target_passes is not None else 'manual'}, "
        f"sampling_mode={args.sampling_mode}, "
        f"output_dir={output_dir}"
    )
    print(
        "Training data: "
        f"episodes={int(split_stats['episodes'])}, "
        f"total_bars={int(split_stats['total_bars'])}, "
        f"mean_bars={split_stats['mean_bars']:.1f}, "
        f"median_bars={split_stats['median_bars']:.1f}, "
        f"approx_iterations_per_full_pass={iterations_for_one_pass}"
    )
    print(
        "Config: "
        f"lr={args.lr}, gamma={args.gamma}, gae_lambda={args.gae_lambda}, "
        f"clip_param={args.clip_param}, entropy_coeff={args.entropy_coeff}, "
        f"alpha={args.alpha}, reward_lambda={args.reward_lambda}, "
        f"mu={args.mu}, beta_vrp={args.beta_vrp}"
    )

    register_env(ENV_NAME, lambda config: RandomEpisodeStraddleEnv(config))

    init_ray_runtime(
        ray_temp_dir=args.ray_temp_dir,
        ray_local_mode=args.ray_local_mode,
        object_store_memory_bytes=int(args.object_store_memory_gb * 1024**3),
        memory_bytes=int(args.ray_memory_gb * 1024**3) if args.ray_memory_gb else None,
    )
    algo = build_ppo_config(args).build()

    checkpoints: list[str] = []
    last_result: dict = {}
    started_at = time.perf_counter()
    try:
        for iteration in range(1, args.iterations + 1):
            result = algo.train()
            last_result = result
            reward_mean = result.get("episode_reward_mean", float("nan"))
            len_mean = result.get("episode_len_mean", float("nan"))
            elapsed = time.perf_counter() - started_at
            avg_per_iter = elapsed / max(iteration, 1)
            eta = avg_per_iter * (args.iterations - iteration)
            print(
                f"iter={iteration:04d}/{args.iterations:04d} "
                f"progress={iteration / args.iterations:.1%} "
                f"episode_reward_mean={reward_mean:.6f} "
                f"episode_len_mean={len_mean:.2f} "
                f"elapsed={format_duration(elapsed)} "
                f"eta={format_duration(eta)}"
            )

            if iteration % args.checkpoint_every == 0:
                checkpoint = checkpoint_path(algo.save(str(output_dir)), output_dir)
                checkpoints.append(str(checkpoint))
                print(f"saved checkpoint: {checkpoint}")

        final_checkpoint = checkpoint_path(algo.save(str(output_dir)), output_dir)
        checkpoints.append(str(final_checkpoint))
        print(f"saved final checkpoint: {final_checkpoint}")

    finally:
        algo.stop()
        ray.shutdown()

    summary = {
        "preset": args.preset,
        "model_variant": args.model_variant,
        "train_years": [args.train_start_year, args.train_end_year],
        "iterations": args.iterations,
        "target_passes": args.target_passes,
        "split_stats": split_stats,
        "approx_iterations_per_full_pass": iterations_for_one_pass,
        "approx_sampled_bars": args.iterations * args.train_batch_size,
        "approx_passes": (
            (args.iterations * args.train_batch_size) / split_stats["total_bars"]
            if split_stats["total_bars"] > 0
            else 0.0
        ),
        "final_episode_reward_mean": last_result.get("episode_reward_mean"),
        "final_episode_len_mean": last_result.get("episode_len_mean"),
        "checkpoints": checkpoints,
        "final_checkpoint": checkpoints[-1] if checkpoints else None,
        "config": vars(args),
    }
    save_json(output_dir / "training_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RLlib PPO on V3R simulated-quote SPX straddle episodes."
    )
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_SENTINEL)
    parser.add_argument(
        "--model-variant",
        choices=["v3", "exit", "long_short"],
        default="v3",
        help=(
            "v3: WAIT/ENTER_SHORT; exit: add EXIT; "
            "long_short: add EXIT and ENTER_LONG."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=["custom", "main", "no_tc"],
        default="custom",
        help=(
            "Use 'main' for the selected TC-aware final model, "
            "'no_tc' for the same PPO settings with alpha=0, or 'custom'."
        ),
    )
    parser.add_argument("--train-start-year", type=int, default=2016)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument(
        "--sampling-mode",
        choices=["random", "sequential", "epoch_shuffle"],
        default="epoch_shuffle",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--target-passes",
        type=float,
        default=None,
        help=(
            "If set, override --iterations so sampled train_batch_size steps "
            "approximately cover the train split this many times."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--num-rollout-workers", type=int, default=0)
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--rollout-fragment-length", type=int, default=256)
    parser.add_argument("--ray-local-mode", action="store_true")
    parser.add_argument(
        "--ray-temp-dir",
        default=str(Path(tempfile.gettempdir()) / "ray_tmp_v3"),
    )
    parser.add_argument(
        "--object-store-memory-gb",
        type=float,
        default=1.0,
        help="Ray object store size in GB. A smaller value avoids Windows error 1450.",
    )
    parser.add_argument(
        "--ray-memory-gb",
        type=float,
        default=None,
        help=(
            "Optional Ray task/actor memory override in GB. Useful on Windows when "
            "Ray underestimates available memory after reserving object store memory."
        ),
    )

    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.001)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--sgd-minibatch-size", type=int, default=512)
    parser.add_argument("--num-sgd-iter", type=int, default=10)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--reward-lambda", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--beta-vrp", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
