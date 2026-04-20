from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.greedy_agent import GreedyAgent
from agents.heuristic_agent import HeuristicAgent
from agents.model_agent import SB3PolicyAgent
from agents.random_agent import RandomAgent
from env.project_env import ProjectEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a narrated Project Manager Chaos episode.")
    parser.add_argument(
        "--agent",
        choices=("heuristic", "random", "greedy", "ppo", "dqn"),
        default="heuristic",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-path", type=Path, default=None)
    return parser.parse_args()


def build_agent(args: argparse.Namespace, env: ProjectEnv):
    if args.agent == "heuristic":
        return HeuristicAgent()
    if args.agent == "random":
        return RandomAgent(env, seed=args.seed)
    if args.agent == "greedy":
        return GreedyAgent(env)
    if args.agent in {"ppo", "dqn"}:
        if args.model_path is None:
            raise ValueError(f"--model-path is required when agent={args.agent}")
        return SB3PolicyAgent.from_path(env=env, algorithm=args.agent, model_path=args.model_path)
    raise ValueError(f"Unsupported agent: {args.agent}")


def main() -> None:
    args = parse_args()
    env = ProjectEnv(seed=args.seed)
    _, info = env.reset(seed=args.seed)
    state = info["state"]

    agent = build_agent(args, env)

    print(f"Running demo with agent={args.agent} seed={args.seed}")
    print("=" * 72)

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action = agent.act(state)
        _, reward, terminated, truncated, info = env.step(action)
        state = info["state"]
        total_reward += reward

        print(f"Step {state['step']}")
        print(f"Action: {json.dumps(action)}")
        print(f"Event: {info['event']}")
        print(f"Reward: {reward:.2f}")
        print(env.render())
        print("-" * 72)

    print(f"Episode finished. total_reward={total_reward:.2f}")
    print(f"Metrics: {json.dumps(info['metrics'], indent=2)}")


if __name__ == "__main__":
    main()
