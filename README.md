# SAC Ball Maze

Soft Actor-Critic (SAC) with a curiosity bonus, trained on Gymnasium-Robotics
`PointMaze`. A force-controlled ball learns to roll to a goal cell in a maze.

Written as a lesson project: the code is commented to explain *why* each step
exists, not just what it does.

## What's in here

| File | Role |
|------|------|
| [`main.py`](main.py) | Entry point. Hyperparameters, env setup, kicks off training. |
| [`agent.py`](agent.py) | The `Agent` class: SAC update rule, training loop, checkpoint I/O. |
| [`model.py`](model.py) | The three networks: `Actor`, `Critic`, `PredictiveModel`. |
| [`buffer.py`](buffer.py) | `ReplayBuffer` — stores past transitions to sample training batches from. |
| [`gym_robotics_custom.py`](gym_robotics_custom.py) | Flattens PointMaze's dict observation into one 1-D array. |
| [`evaluate.py`](evaluate.py) | Loads checkpoints and plays greedy episodes with the viewer open. |
| [`run_cpu.sh`](run_cpu.sh) | SLURM batch script for running training on an HPC CPU node. |

## How it works

SAC is an off-policy actor-critic algorithm. Four networks, each with its own
Adam optimizer:

```
       state
         |
         v
    +---------+   action    +----------+
    |  Actor  |------------>|  Critic  |---> Q1, Q2   "how good was that action?"
    | (policy)|             | (twin Q) |
    +---------+             +----------+
         ^                        ^
         |  "pick actions the     |  trained against
         |   Critic scores high"  |  targets from
         |                        |
         |                  +-----------------+
         |                  | Critic target   |  slow-moving copy,
         |                  | (soft-updated)  |  keeps targets stable
         |                  +-----------------+
         |
    +------------------+
    | PredictiveModel  |  predicts next state; its error becomes
    | (curiosity)      |  an intrinsic reward for novel states
    +------------------+
```

Three pieces worth naming:

- **Twin critics.** Two Q-networks, and the update uses `min(Q1, Q2)`. Taking
  the smaller of two estimates counteracts the tendency of a single Q-network
  to drift optimistic.
- **Entropy term.** The policy loss is `alpha * log_pi - Q`, so the agent is
  rewarded for keeping its action distribution random. That randomness *is*
  the exploration. `alpha` is fixed at 0.12 here — the SAC paper learns it.
- **Curiosity bonus.** `PredictiveModel` predicts the next state from
  (state, action). Where it predicts badly, the agent hasn't been much, so its
  error is scaled by `exploration_scaling_factor` and added to the reward.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` pulls PyTorch from the CUDA 11.8 index. The same wheels run
fine on CPU-only machines.

## Training

```bash
python main.py
```

Picks GPU automatically if one is visible, otherwise CPU. Defaults live at the
top of [`main.py`](main.py):

| Setting | Value | Meaning |
|---------|-------|---------|
| `episodes` | 100 | training episodes (passed to `agent.train`) |
| `max_episode_steps` | 100 | steps before an episode is cut off |
| `warmup` | 20 | first N episodes use random actions, to fill the buffer |
| `batch_size` | 64 | transitions per gradient step |
| `updates_per_step` | 4 | gradient steps per env step |
| `hidden_size` | 512 | width of every hidden layer |
| `learning_rate` | 1e-4 | Adam LR, shared by all four optimizers |
| `gamma` | 0.99 | discount — how much future reward counts now |
| `tau` | 0.005 | target-network soft update rate |
| `alpha` | 0.12 | entropy weight |
| `exploration_scaling_factor` | 1.5 | multiplier on the curiosity bonus |

The default maze is a 3-cell straight corridor (`STRAIGHT_MAZE`), the easiest
warm-up layout. Swap `maze_map` for a real U-maze once that solves.

Checkpoints are written to `checkpoints/` every 10 episodes; TensorBoard logs
go to `runs/`.

### On an HPC cluster (SLURM)

```bash
sbatch run_cpu.sh
squeue --me
tail -f sac-<jobid>.out
```

Edit the `PY=` line in [`run_cpu.sh`](run_cpu.sh) to point at your environment's
interpreter — conda is not on `PATH` in a batch shell, so the script calls the
interpreter directly instead of activating.

## Watching the results

```bash
tensorboard --logdir runs/
```

`reward/episode` is the curve that matters. The loss curves are diagnostics:
critic loss rising while reward rises is normal (Q values are growing), and
policy loss falls for the same reason.

To watch the trained agent:

```bash
python evaluate.py
```

Runs 5 deterministic episodes with `render_mode="human"` and prints per-episode
reward plus a success rate. Needs a display — run it locally, not on a compute
node, after copying `checkpoints/` back.

## Requirements

- Python 3.8+
- `gymnasium==0.29.1`, `gymnasium-robotics==1.2.4` (MuJoCo comes with it)
- PyTorch, TensorBoard
