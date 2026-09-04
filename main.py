import gymnasium as gym
from gym_robotics_custom import RoboGymObservationWrapper
from model import *
from agent import Agent
from buffer import ReplayBuffer


if __name__ == "__main__":

    replay_buffer_size = 1000000
    warmup = 20
    batch_size = 64
    updates_per_step = 4
    gamma = 0.99
    tau = 0.005
    alpha = 0.12
    target_update_interval = 1
    hidden_size = 512
    learning_rate = 0.0001
    exploration_scaling_factor = 1.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        # Allow TF32 matmuls. Faster on Ampere and newer, and the reduced
        # precision is irrelevant for reinforcement-learning-sized networks.
        torch.set_float32_matmul_precision("high")
    else:
        print("No CUDA device found, training on CPU")

    env_name = "PointMaze_UMaze-v3"
    max_episode_steps = 100

    STRAIGHT_MAZE = [[1,1,1,1,1],
                     [1,0,0,0,1],
                     [1,1,1,1,1]]

    # No render_mode: training is headless. A compute node has no display,
    # and skipping the renderer is faster everywhere.
    env = gym.make(env_name, maze_map=STRAIGHT_MAZE,
                   max_episode_steps=max_episode_steps)
    env = RoboGymObservationWrapper(env)

    observation_size = env.observation_space.shape[0]

    agent = Agent(num_inputs=observation_size,
                  action_space=env.action_space,
                  policy="Gaussian",
                  hidden_dim=hidden_size,
                  gamma=gamma,
                  tau=tau,
                  alpha=alpha,
                  learning_rate=learning_rate,
                  target_update_interval=target_update_interval,
                  exploration_scaling_factor=exploration_scaling_factor,
                  device=device)

    memory = ReplayBuffer(replay_buffer_size, observation_size, env.action_space.shape[0])

    agent.train(env=env,
                memory=memory,
                episodes=100,
                warmup=warmup,
                batch_size=batch_size,
                updates_per_step=updates_per_step,
                max_episode_steps=max_episode_steps,
                env_name=env_name,
                summary_writer_name=f"straight_maze_{env_name}_gamma{gamma}_lr{learning_rate}")

    env.close()
