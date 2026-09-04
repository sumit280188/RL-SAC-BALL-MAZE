"""Watch a trained policy run the maze.

Loads checkpoints/ and plays greedy (deterministic) episodes with the
MuJoCo viewer open. Run this on a machine with a display, not on the
HPC compute node - copy checkpoints/ back first.
"""

import gymnasium as gym
from gym_robotics_custom import RoboGymObservationWrapper
from model import *
from agent import Agent

if __name__ == "__main__":

    episodes = 5
    hidden_size = 512
    max_episode_steps = 100

    device = torch.device("cpu")

    env_name = "PointMaze_UMaze-v3"

    STRAIGHT_MAZE = [[1,1,1,1,1],
                     [1,0,0,0,1],
                     [1,1,1,1,1]]

    env = gym.make(env_name, maze_map=STRAIGHT_MAZE,
                   max_episode_steps=max_episode_steps,
                   render_mode="human")
    env = RoboGymObservationWrapper(env)

    observation_size = env.observation_space.shape[0]

    # Only the policy is used below, but Agent builds all four networks,
    # so the unused hyperparameters still have to be passed something.
    agent = Agent(num_inputs=observation_size,
                  action_space=env.action_space,
                  policy="Gaussian",
                  hidden_dim=hidden_size,
                  gamma=0.99,
                  tau=0.005,
                  alpha=0.12,
                  learning_rate=0.0001,
                  target_update_interval=1,
                  exploration_scaling_factor=1.5,
                  device=device)

    agent.load_checkpoint(evaluate=True)

    rewards = []
    successes = 0

    for i_episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done and episode_steps < max_episode_steps:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1

        success = bool(info.get("success", terminated))
        successes += success
        rewards.append(episode_reward)
        print(f"Episode: {i_episode}, steps: {episode_steps}, "
              f"reward: {round(episode_reward, 2)}, success: {success}")

    print(f"\nAverage reward over {episodes} episodes: {sum(rewards) / episodes:.2f}")
    print(f"Success rate: {successes}/{episodes}")

    env.close()
