import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

KEYS = ("observation", "achieved_goal", "desired_goal")


class RoboGymObservationWrapper(ObservationWrapper):
    """Flatten PointMaze's Dict observation into one 1-D array."""

    def __init__(self, env):
        super().__init__(env)

        # Declare the flattened space so observation_space.shape[0] is
        # correct without having to call reset() first.
        low = np.concatenate([env.observation_space[k].low for k in KEYS])
        high = np.concatenate([env.observation_space[k].high for k in KEYS])
        self.observation_space = Box(low, high, dtype=np.float32)

    def observation(self, observation):
        return np.concatenate([observation[k] for k in KEYS])
