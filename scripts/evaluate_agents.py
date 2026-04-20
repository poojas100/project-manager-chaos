from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from agents.greedy_agent import GreedyAgent
from agents.heuristic_agent import HeuristicAgent
from agents.model_agent import SB3PolicyAgent
from agents.random_agent import RandomAgent
from env.project_env import ProjectEnv


def run_episode(env: ProjectEnv, agent) -> tuple[float, dict]:
    _, info = env.reset()
    state = info["state"]
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.act(state)
        _, reward, terminated, truncated, info = env.step(action)
        state = info["state"]
        total_reward += reward

    return total_reward, info["metrics"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline agents over many episodes.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ppo-model-path", type=Path, default=None)
    parser.add_argument("--dqn-model-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: dict[str, list[tuple[float, dict]]] = {
        "random": [],
        "greedy": [],
        "heuristic": [],
    }
    if args.ppo_model_path is not None:
        results["ppo"] = []
    if args.dqn_model_path is not None:
        results["dqn"] = []

    for episode in range(args.episodes):
        random_env = ProjectEnv(seed=args.seed + episode)
        greedy_env = ProjectEnv(seed=args.seed + episode)
        heuristic_env = ProjectEnv(seed=args.seed + episode)

        results["random"].append(
            run_episode(random_env, RandomAgent(random_env, seed=args.seed + episode))
        )
        results["greedy"].append(run_episode(greedy_env, GreedyAgent(greedy_env)))
        results["heuristic"].append(run_episode(heuristic_env, HeuristicAgent()))
        if args.ppo_model_path is not None:
            ppo_env = ProjectEnv(seed=args.seed + episode)
            ppo_agent = SB3PolicyAgent.from_path(ppo_env, "ppo", args.ppo_model_path)
            results["ppo"].append(run_episode(ppo_env, ppo_agent))
        if args.dqn_model_path is not None:
            dqn_env = ProjectEnv(seed=args.seed + episode)
            dqn_agent = SB3PolicyAgent.from_path(dqn_env, "dqn", args.dqn_model_path)
            results["dqn"].append(run_episode(dqn_env, dqn_agent))

    for agent_name, episodes in results.items():
        rewards = [reward for reward, _ in episodes]
        completions = [metrics["completed_tasks"] for _, metrics in episodes]
        missed = [metrics["missed_deadlines"] for _, metrics in episodes]
        burnout = [metrics["burnout_incidents"] for _, metrics in episodes]
        print(
            f"{agent_name}: avg_reward={mean(rewards):.2f}, "
            f"avg_completed={mean(completions):.2f}, "
            f"avg_missed={mean(missed):.2f}, "
            f"avg_burnout={mean(burnout):.2f}"
        )


if __name__ == "__main__":
    main()
