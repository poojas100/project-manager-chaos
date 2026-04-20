from __future__ import annotations

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from env.events import apply_event
from env.project_env import ProjectEnv


def test_reset_produces_valid_initial_state():
    env = ProjectEnv(seed=3)
    observation, info = env.reset(seed=3)

    assert observation.shape == env.observation_space.shape
    assert info["state"]["time_left"] == env.max_steps
    assert len(info["state"]["tasks"]) == 3
    assert set(info["state"]["team"].keys()) == {"engineer", "designer"}


def test_work_action_updates_progress():
    env = ProjectEnv(seed=1, event_probability=0.0)
    _, info = env.reset(seed=1)
    state = info["state"]
    task_before = next(task for task in state["tasks"] if task["id"] == 1)["progress"]

    _, _, _, _, info = env.step({"action": "work_on_task", "task_id": 1, "effort_level": 1})
    task_after = next(task for task in info["state"]["tasks"] if task["id"] == 1)["progress"]

    assert task_after > task_before


def test_deadlines_decrement_once_per_step():
    env = ProjectEnv(seed=1, event_probability=0.0)
    _, info = env.reset(seed=1)
    deadline_before = next(task for task in info["state"]["tasks"] if task["id"] == 2)["deadline"]

    _, _, _, _, info = env.step({"action": "do_nothing"})
    deadline_after = next(task for task in info["state"]["tasks"] if task["id"] == 2)["deadline"]

    assert deadline_after == deadline_before - 1


def test_completed_task_reward_only_once():
    env = ProjectEnv(seed=1, event_probability=0.0)
    env.reset(seed=1)
    task = next(task for task in env.state["tasks"] if task["id"] == 1)
    task["progress"] = 0.95

    _, reward_1, _, _, info_1 = env.step({"action": "work_on_task", "task_id": 1, "effort_level": 2})
    _, reward_2, _, _, info_2 = env.step({"action": "do_nothing"})

    assert info_1["metrics"]["completed_tasks"] == 1
    assert reward_1 > reward_2


def test_missed_deadlines_penalize():
    env = ProjectEnv(seed=1, event_probability=0.0)
    env.reset(seed=1)
    task = next(task for task in env.state["tasks"] if task["id"] == 2)
    task["deadline"] = 0
    task["progress"] = 0.2

    _, reward, _, _, info = env.step({"action": "do_nothing"})

    assert reward < 0
    assert info["metrics"]["missed_deadlines"] >= 1


def test_event_application_keeps_state_legal():
    env = ProjectEnv(seed=1, event_probability=0.0)
    env.reset(seed=1)
    apply_event(env.state, "scope_change")

    assert env.state["events"]["scope_change"] is True
    assert len(env.state["tasks"]) == 4
    assert env.state["tasks"][-1]["owner"] in env.state["team"]


def test_invalid_actions_do_not_crash_environment():
    env = ProjectEnv(seed=1, event_probability=0.0)
    env.reset(seed=1)
    _, reward, terminated, truncated, info = env.step({"action": "work_on_task", "task_id": 99, "effort_level": 1})

    assert isinstance(reward, float)
    assert info["invalid_action"] is True
    assert terminated is False
    assert truncated is False


def test_episode_ends_by_completion_or_max_steps():
    env = ProjectEnv(seed=1, event_probability=0.0)
    env.reset(seed=1)
    for task in env.state["tasks"]:
        task["progress"] = 1.0

    _, _, terminated, truncated, _ = env.step({"action": "do_nothing"})

    assert terminated is True
    assert truncated is False


def test_baseline_agents_can_run_rollouts():
    env_random = ProjectEnv(seed=5)
    env_heuristic = ProjectEnv(seed=5)
    random_agent = RandomAgent(env_random, seed=5)
    heuristic_agent = HeuristicAgent()

    for env, agent in ((env_random, random_agent), (env_heuristic, heuristic_agent)):
        _, info = env.reset(seed=5)
        state = info["state"]
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            action = agent.act(state)
            _, _, terminated, truncated, info = env.step(action)
            state = info["state"]
            steps += 1
        assert steps > 0


def test_seeded_episode_regression():
    env = ProjectEnv(seed=11, event_probability=0.5)
    _, info = env.reset(seed=11)
    actions = [
        {"action": "prioritize_task", "task_id": 2},
        {"action": "work_on_task", "task_id": 2, "effort_level": 2},
        {"action": "allocate_budget", "task_id": 3, "amount": 100},
    ]

    rewards = []
    events = []
    for action in actions:
        _, reward, _, _, info = env.step(action)
        rewards.append(round(reward, 4))
        events.append(info["event"])

    assert rewards == [-0.32, 0.01, -0.48]
    assert events == ["scope_change", "none", "engineer_sick"]
