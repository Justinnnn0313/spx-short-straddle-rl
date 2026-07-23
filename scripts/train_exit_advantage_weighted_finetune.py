from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "spx_straddle_rl"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from env import ENTER_SHORT, EXIT, WAIT, StraddleEnv  # noqa: E402
from rl_agent_common import load_episode_split, save_json  # noqa: E402


class ExitActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, 1)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.actor(h).squeeze(-1), self.critic(h).squeeze(-1)


def load_bc_initialized_model(path: str | Path) -> tuple[ExitActorCritic, np.ndarray, np.ndarray, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = ExitActorCritic(int(state["input_dim"]), int(state["hidden_dim"]))
    bc_state = state["model_state"]
    model.shared[0].weight.data.copy_(bc_state["net.0.weight"])
    model.shared[0].bias.data.copy_(bc_state["net.0.bias"])
    model.shared[2].weight.data.copy_(bc_state["net.2.weight"])
    model.shared[2].bias.data.copy_(bc_state["net.2.bias"])
    model.actor.weight.data.copy_(bc_state["net.4.weight"])
    model.actor.bias.data.copy_(bc_state["net.4.bias"])
    return model, np.asarray(state["mean"], dtype=np.float32), np.asarray(state["std"], dtype=np.float32), state


def bc_teacher_logit(bc_state: dict, x: torch.Tensor) -> torch.Tensor:
    h = F.linear(x, bc_state["model_state"]["net.0.weight"], bc_state["model_state"]["net.0.bias"])
    h = F.relu(h)
    h = F.linear(h, bc_state["model_state"]["net.2.weight"], bc_state["model_state"]["net.2.bias"])
    h = F.relu(h)
    return F.linear(h, bc_state["model_state"]["net.4.weight"], bc_state["model_state"]["net.4.bias"]).squeeze(-1)


def make_env(episode, norm_stats, args: argparse.Namespace) -> StraddleEnv:
    return StraddleEnv(
        {
            "episode_data": episode,
            "norm_stats": norm_stats,
            "model_variant": "exit",
            "alpha": args.alpha,
            "lambda_": args.reward_lambda,
            "mu": args.mu,
            "beta_vrp": args.beta_vrp,
            "reward_mode": "shaped",
            "obs_features": "position",
            "use_action_mask_obs": True,
            "min_hold_bars": args.min_hold_bars,
            "invalid_action_mode": "remap",
        }
    )


def obs_parts(obs) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(obs["observations"], dtype=np.float32), np.asarray(obs["action_mask"], dtype=np.int8)


def scale_obs(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    x_np = (features.reshape(1, -1) - mean) / std
    return torch.from_numpy(x_np.astype(np.float32)).squeeze(0)


def dist_value(model: ExitActorCritic, x: torch.Tensor) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
    logit, value = model(x.unsqueeze(0) if x.ndim == 1 else x)
    logits = torch.stack([torch.zeros_like(logit), logit], dim=-1)
    return Categorical(logits=logits), value, logit


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    out = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        out.append(running)
    out.reverse()
    return torch.tensor(list(reversed(out)), dtype=torch.float32)


def choose_action(
    model: ExitActorCritic,
    features: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    deterministic: bool,
) -> tuple[int, int | None, torch.Tensor | None]:
    pos = float(features[9]) if len(features) > 9 else 0.0
    if abs(pos) < 0.5:
        return (ENTER_SHORT if mask[ENTER_SHORT] == 1 else WAIT), None, None
    if len(mask) <= EXIT or mask[EXIT] == 0:
        return WAIT, None, None
    x = scale_obs(features, mean, std)
    dist, _value, _logit = dist_value(model, x)
    mapped = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
    mapped_int = int(mapped.item())
    return EXIT if mapped_int == 1 else WAIT, mapped_int, x


def run_episode(
    model: ExitActorCritic,
    episode,
    norm_stats,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
    deterministic: bool = False,
) -> dict:
    env = make_env(episode, norm_stats, args)
    obs, _ = env.reset()
    rewards: list[float] = []
    decision_indices: list[int] = []
    xs: list[torch.Tensor] = []
    actions_01: list[int] = []
    actions: list[int] = []
    infos: list[dict] = []
    done = False
    while not done:
        features, mask = obs_parts(obs)
        action, mapped, x = choose_action(model, features, mask, mean, std, deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        reward_idx = len(rewards)
        rewards.append(float(reward))
        actions.append(int(action))
        infos.append(info)
        if mapped is not None and x is not None:
            decision_indices.append(reward_idx)
            xs.append(x)
            actions_01.append(mapped)
    returns = discounted_returns(rewards, args.gamma)
    return {
        "xs": xs,
        "actions_01": actions_01,
        "returns": returns[decision_indices] if decision_indices else torch.empty(0, dtype=torch.float32),
        "actions": actions,
        "episode_reward": float(sum(rewards)),
        "episode_pnl": float(infos[-1].get("episode_pnl", 0.0)) if infos else 0.0,
        "exited": int(EXIT in actions),
        "steps": len(actions),
    }


def flatten(results: list[dict]) -> dict[str, torch.Tensor]:
    xs = [x for r in results for x in r["xs"]]
    if not xs:
        return {}
    return {
        "x": torch.stack(xs),
        "action": torch.tensor([a for r in results for a in r["actions_01"]], dtype=torch.long),
        "return": torch.cat([r["returns"] for r in results]).detach(),
    }


def compute_loss(
    model: ExitActorCritic,
    bc_state: dict,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    dist, value, current_logits = dist_value(model, batch["x"])
    with torch.no_grad():
        advantage = batch["return"] - value.detach()
        if len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        if args.method == "awr":
            weights = torch.exp(advantage / args.aw_temperature)
        else:
            weights = torch.exp(torch.clamp(advantage, min=0.0) / args.aw_temperature)
        weights = torch.clamp(weights, max=args.max_weight)
    log_prob = dist.log_prob(batch["action"])
    actor_loss = -(weights * log_prob).mean()
    value_loss = ((value - batch["return"]) ** 2).mean()
    entropy = dist.entropy().mean()
    with torch.no_grad():
        teacher_probs = torch.sigmoid(bc_teacher_logit(bc_state, batch["x"]))
    bc_anchor = F.binary_cross_entropy_with_logits(current_logits, teacher_probs)
    loss = actor_loss + args.value_coeff * value_loss - args.entropy_coeff * entropy + args.bc_coeff * bc_anchor
    return loss, {
        "actor_loss": float(actor_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "bc_anchor_loss": float(bc_anchor.item()),
        "mean_weight": float(weights.mean().item()),
        "max_weight": float(weights.max().item()),
    }


def evaluate(model, episodes, norm_stats, mean, std, args, max_episodes: int) -> dict:
    model.eval()
    rows = []
    with torch.no_grad():
        for ep in episodes[:max_episodes]:
            rows.append(run_episode(model, ep, norm_stats, mean, std, args, deterministic=True))
    rewards = np.asarray([r["episode_reward"] for r in rows], dtype=np.float64)
    pnls = np.asarray([r["episode_pnl"] for r in rows], dtype=np.float64)
    return {
        "episodes": int(len(rows)),
        "mean_reward": float(rewards.mean()) if len(rows) else 0.0,
        "mean_episode_pnl": float(pnls.mean()) if len(rows) else 0.0,
        "exit_rate": float(np.mean([r["exited"] for r in rows])) if rows else 0.0,
        "mean_steps": float(np.mean([r["steps"] for r in rows])) if rows else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BC exit policy with AWR/AWAC-style advantage weighting.")
    parser.add_argument("--method", choices=["awr", "awac"], default="awr")
    parser.add_argument("--artifact-dir", default="artifacts_v3r_simquote")
    parser.add_argument("--bc-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-start-year", type=int, default=2016)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--episodes-per-update", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--aw-temperature", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--value-coeff", type=float, default=0.5)
    parser.add_argument("--entropy-coeff", type=float, default=0.0005)
    parser.add_argument("--bc-coeff", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-hold-bars", type=int, default=26)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--reward-lambda", type=float, default=0.2)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--beta-vrp", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes, norm_stats = load_episode_split(args.artifact_dir, args.train_start_year, args.train_end_year)
    model, mean, std, bc_state = load_bc_initialized_model(args.bc_model)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    started = time.perf_counter()

    for iteration in range(1, args.iterations + 1):
        model.train()
        selected = [episodes[int(i)] for i in rng.integers(0, len(episodes), size=args.episodes_per_update)]
        results = [run_episode(model, ep, norm_stats, mean, std, args, deterministic=False) for ep in selected]
        batch = flatten(results)
        metrics = {"loss": 0.0, "actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "bc_anchor_loss": 0.0, "mean_weight": 0.0, "max_weight": 0.0}
        if batch:
            n = len(batch["action"])
            losses = []
            for _epoch in range(args.update_epochs):
                order = rng.permutation(n)
                for start in range(0, n, args.minibatch_size):
                    idx = torch.from_numpy(order[start : start + args.minibatch_size].astype(np.int64))
                    mini = {key: value[idx] for key, value in batch.items()}
                    loss, step_metrics = compute_loss(model, bc_state, mini, args)
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optim.step()
                    losses.append(float(loss.item()))
                    metrics.update(step_metrics)
            metrics["loss"] = float(np.mean(losses)) if losses else 0.0

        row = {
            "iteration": iteration,
            "method": args.method,
            "decisions": int(len(batch.get("action", []))) if batch else 0,
            "mean_train_reward": float(np.mean([r["episode_reward"] for r in results])),
            "mean_train_pnl": float(np.mean([r["episode_pnl"] for r in results])),
            "train_exit_rate": float(np.mean([r["exited"] for r in results])),
            **metrics,
        }
        if iteration == 1 or iteration % args.eval_every == 0 or iteration == args.iterations:
            eval_row = evaluate(model, episodes, norm_stats, mean, std, args, args.eval_episodes)
            row.update({f"eval_{k}": v for k, v in eval_row.items()})
            elapsed = int(time.perf_counter() - started)
            print(
                f"iter={iteration:04d}/{args.iterations:04d} method={args.method} "
                f"loss={row['loss']:.6f} train_pnl={row['mean_train_pnl']:.2f} "
                f"exit_rate={row['train_exit_rate']:.3f} eval_pnl={eval_row['mean_episode_pnl']:.2f} "
                f"eval_exit_rate={eval_row['exit_rate']:.3f} decisions={row['decisions']} elapsed={elapsed}s",
                flush=True,
            )
        history.append(row)

    model_path = output_dir / "exit_actor_critic_finetuned.pt"
    payload = {
        "model_state": model.state_dict(),
        "input_dim": int(bc_state["input_dim"]),
        "hidden_dim": int(bc_state["hidden_dim"]),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "bc_model": str(args.bc_model),
        "config": vars(args) | {"reward_mode": "shaped"},
        "history": history,
    }
    torch.save(payload, model_path)
    save_json(output_dir / "training_summary.json", {"model_path": str(model_path), "history": history, "config": payload["config"]})
    print(f"model: {model_path}")
    print(f"summary: {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
