from __future__ import annotations

import argparse
import json
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


def load_initialized_model(
    path: str | Path,
    init_mode: str = "bc",
    freeze_actor: bool = False,
) -> tuple[ExitActorCritic, np.ndarray, np.ndarray, dict | None]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    input_dim = int(state["input_dim"])
    hidden_dim = int(state["hidden_dim"])
    model = ExitActorCritic(input_dim, hidden_dim)
    bc_state = state["model_state"]
    if init_mode == "bc":
        model.shared[0].weight.data.copy_(bc_state["net.0.weight"])
        model.shared[0].bias.data.copy_(bc_state["net.0.bias"])
        model.shared[2].weight.data.copy_(bc_state["net.2.weight"])
        model.shared[2].bias.data.copy_(bc_state["net.2.bias"])
        model.actor.weight.data.copy_(bc_state["net.4.weight"])
        model.actor.bias.data.copy_(bc_state["net.4.bias"])
        if freeze_actor:
            for p in list(model.shared.parameters()) + list(model.actor.parameters()):
                p.requires_grad = False
        init_state: dict | None = state
    elif init_mode == "random":
        init_state = None
    else:
        raise ValueError("init_mode must be one of: bc, random")
    mean = np.asarray(state["mean"], dtype=np.float32)
    std = np.asarray(state["std"], dtype=np.float32)
    return model, mean, std, init_state


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
            "reward_mode": args.reward_mode,
            "obs_features": "position",
            "use_action_mask_obs": True,
            "min_hold_bars": args.min_hold_bars,
            "invalid_action_mode": "remap",
        }
    )


def obs_parts(obs) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(obs["observations"], dtype=np.float32), np.asarray(obs["action_mask"], dtype=np.int8)


def choose_action(
    model: ExitActorCritic,
    features: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    deterministic: bool,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    pos = float(features[9]) if len(features) > 9 else 0.0
    if abs(pos) < 0.5:
        if mask[ENTER_SHORT] == 1:
            return ENTER_SHORT, None, None, None
        return WAIT, None, None, None
    if len(mask) <= EXIT or mask[EXIT] == 0:
        return WAIT, None, None, None

    x_np = (features.reshape(1, -1) - mean) / std
    x = torch.from_numpy(x_np.astype(np.float32))
    exit_logit, value = model(x)
    logits = torch.stack([torch.zeros_like(exit_logit), exit_logit], dim=-1)
    dist = Categorical(logits=logits)
    mapped = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
    action = EXIT if int(mapped.item()) == 1 else WAIT
    log_prob = dist.log_prob(mapped).squeeze(0)
    entropy = dist.entropy().squeeze(0)
    return action, log_prob, value.squeeze(0), entropy


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    out = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        out.append(running)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)


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
    done = False
    rewards: list[float] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    decision_features: list[torch.Tensor] = []
    actions: list[int] = []
    infos: list[dict] = []
    while not done:
        features, mask = obs_parts(obs)
        action, log_prob, value, entropy = choose_action(model, features, mask, mean, std, deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        rewards.append(float(reward))
        actions.append(int(action))
        infos.append(info)
        if log_prob is not None and value is not None and entropy is not None:
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            x_np = (features.reshape(1, -1) - mean) / std
            decision_features.append(torch.from_numpy(x_np.astype(np.float32)).squeeze(0))
    return {
        "rewards": rewards,
        "log_probs": log_probs,
        "values": values,
        "entropies": entropies,
        "decision_features": decision_features,
        "actions": actions,
        "episode_reward": float(sum(rewards)),
        "episode_pnl": float(infos[-1].get("episode_pnl", 0.0)) if infos else 0.0,
        "entered": int(ENTER_SHORT in actions),
        "exited": int(EXIT in actions),
        "steps": len(actions),
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
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "mean_episode_pnl": float(pnls.mean()) if len(pnls) else 0.0,
        "exit_rate": float(np.mean([r["exited"] for r in rows])) if rows else 0.0,
        "mean_steps": float(np.mean([r["steps"] for r in rows])) if rows else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune exit policy with lightweight actor-critic.")
    parser.add_argument("--artifact-dir", default="artifacts_v3r_simquote")
    parser.add_argument("--bc-model", required=True)
    parser.add_argument("--init-mode", choices=["bc", "random"], default="bc")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-start-year", type=int, default=2016)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--value-coeff", type=float, default=0.5)
    parser.add_argument("--entropy-coeff", type=float, default=0.001)
    parser.add_argument("--bc-coeff", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-hold-bars", type=int, default=26)
    parser.add_argument("--reward-mode", choices=["direct", "shaped", "exit_risk"], default="direct")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--reward-lambda", type=float, default=0.2)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--beta-vrp", type=float, default=0.5)
    parser.add_argument("--freeze-actor", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes, norm_stats = load_episode_split(args.artifact_dir, args.train_start_year, args.train_end_year)
    model, mean, std, bc_state = load_initialized_model(
        args.bc_model,
        init_mode=args.init_mode,
        freeze_actor=args.freeze_actor,
    )
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    history = []
    started = time.perf_counter()

    for iteration in range(1, args.iterations + 1):
        model.train()
        batch = [episodes[int(i)] for i in rng.integers(0, len(episodes), size=args.episodes_per_update)]
        policy_losses = []
        value_losses = []
        entropies = []
        bc_anchor_losses = []
        episode_rewards = []
        episode_pnls = []
        exits = []
        for ep in batch:
            result = run_episode(model, ep, norm_stats, mean, std, args, deterministic=False)
            episode_rewards.append(result["episode_reward"])
            episode_pnls.append(result["episode_pnl"])
            exits.append(result["exited"])
            if not result["log_probs"]:
                continue
            returns = discounted_returns(result["rewards"][-len(result["log_probs"]):], args.gamma)
            values = torch.stack(result["values"])
            log_probs = torch.stack(result["log_probs"])
            entropy = torch.stack(result["entropies"]).mean()
            if bc_state is not None and args.bc_coeff > 0 and result["decision_features"]:
                decision_x = torch.stack(result["decision_features"])
                current_logits, _ = model(decision_x)
                with torch.no_grad():
                    teacher_probs = torch.sigmoid(bc_teacher_logit(bc_state, decision_x))
                bc_anchor_losses.append(F.binary_cross_entropy_with_logits(current_logits, teacher_probs))
            advantages = returns - values.detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            policy_losses.append(-(log_probs * advantages).mean())
            value_losses.append(((values - returns) ** 2).mean())
            entropies.append(entropy)

        if policy_losses:
            loss = torch.stack(policy_losses).mean()
            loss = loss + args.value_coeff * torch.stack(value_losses).mean()
            loss = loss - args.entropy_coeff * torch.stack(entropies).mean()
            if bc_state is not None and args.bc_coeff > 0 and bc_anchor_losses:
                loss = loss + args.bc_coeff * torch.stack(bc_anchor_losses).mean()
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            loss_value = float(loss.item())
        else:
            loss_value = 0.0

        row = {
            "iteration": iteration,
            "loss": loss_value,
            "mean_train_reward": float(np.mean(episode_rewards)),
            "mean_train_pnl": float(np.mean(episode_pnls)),
            "train_exit_rate": float(np.mean(exits)),
            "bc_anchor_loss": float(torch.stack(bc_anchor_losses).mean().item()) if bc_anchor_losses else 0.0,
        }
        if iteration == 1 or iteration % args.eval_every == 0 or iteration == args.iterations:
            eval_row = evaluate(model, episodes, norm_stats, mean, std, args, args.eval_episodes)
            row.update({f"eval_{k}": v for k, v in eval_row.items()})
            elapsed = int(time.perf_counter() - started)
            print(
                f"iter={iteration:04d}/{args.iterations:04d} loss={loss_value:.6f} "
                f"train_reward={row['mean_train_reward']:.4f} train_exit_rate={row['train_exit_rate']:.3f} "
                f"eval_reward={eval_row['mean_reward']:.4f} eval_exit_rate={eval_row['exit_rate']:.3f} "
                f"elapsed={elapsed}s",
                flush=True,
            )
        history.append(row)

    model_path = output_dir / "exit_actor_critic_finetuned.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(model.shared[0].in_features),
            "hidden_dim": int(model.shared[0].out_features),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "bc_model": str(args.bc_model),
            "init_mode": args.init_mode,
            "config": vars(args),
            "history": history,
        },
        model_path,
    )
    save_json(output_dir / "training_summary.json", {"model_path": str(model_path), "history": history, "config": vars(args)})
    print(f"model: {model_path}")
    print(f"summary: {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
