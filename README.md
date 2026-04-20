# Project Manager Chaos

`Project Manager Chaos` is a small reinforcement learning playground where a project manager agent learns to make structured decisions under deadlines, fatigue, budget pressure, and random project chaos.

The environment is the center of the project. Agents only choose actions. This keeps the repo ready for:

- baseline rule agents
- greedy one-step action selection
- value-based training with DQN
- PPO training with `stable-baselines3`
- future LLM-driven action selection using the same action schema

## Repo Layout

```text
project-manager-chaos/
|-- agents/
|-- env/
|-- scripts/
|-- tests/
|-- training/
|-- utils/
|-- artifacts/
|-- requirements.txt
```

## Quick Start

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run a narrated demo:

```bash
python -m scripts.run_demo --agent heuristic --seed 7
```

4. Run a quick multi-agent evaluation:

```bash
python -m scripts.evaluate_agents --episodes 10 --seed 7
```

5. Train a PPO baseline:

```bash
python -m training.train_ppo --timesteps 5000 --output-dir artifacts/ppo_baseline
```

6. Train a DQN baseline:

```bash
python -m training.train_dqn --timesteps 5000 --output-dir artifacts/dqn_baseline
```

7. Evaluate trained models alongside baselines:

```bash
python -m scripts.evaluate_agents --episodes 10 --seed 7 --ppo-model-path artifacts/ppo_baseline/ppo_project_manager.zip --dqn-model-path artifacts/dqn_baseline/dqn_project_manager.zip
```

## Environment Design

Each timestep contains:

- tasks with owners, deadlines, progress, priority, and blocked state
- team members with capacity, fatigue, and specialty
- project-level budget, time left, and client satisfaction
- chaos flags for sickness, urgent bugs, and scope changes

The canonical action schema is:

- `work_on_task(task_id, effort_level)`
- `reassign_task(task_id, new_owner)`
- `prioritize_task(task_id)`
- `allocate_budget(task_id, amount)`
- `do_nothing()`

The Gymnasium environment supports both:

- discrete integer actions for PPO
- discrete integer actions for DQN
- structured action dictionaries for heuristic and LLM agents

## Testing

Run the test suite with:

```bash
pytest
```

The tests cover reset, action effects, reward behavior, seeded regression, invalid actions, and rollout stability.
