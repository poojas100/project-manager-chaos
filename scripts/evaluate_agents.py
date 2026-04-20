from __future__ import annotations

import argparse
from statistics import mean

from agents.heuristic_agent import HeuristicAgent
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: dict[str, list[tuple[float, dict]]] = {"random": [], "heuristic": []}

    for episode in range(args.episodes):
        random_env = ProjectEnv(seed=args.seed + episode)
        heuristic_env = ProjectEnv(seed=args.seed + episode)

        results["random"].append(
            run_episode(random_env, RandomAgent(random_env, seed=args.seed + episode))
        )
        results["heuristic"].append(run_episode(heuristic_env, HeuristicAgent()))

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
