import numpy as np


class ReplayBuffer:
    """
    Fixed-size circular store of past transitions.

    A transition is one step of experience:

        (state, action, reward, next_state, mask)

    SAC trains on random past transitions rather than on the most recent
    ones, which breaks the correlation between consecutive samples.

    mask is 1 when the episode continued and 0 when it truly ended,
    so it is the opposite of a 'done' flag.

    The buffer never grows. Once it is full, the oldest entry is
    overwritten by the newest one.
    """

    def __init__(self, max_size, input_size, n_actions):
        self.mem_size = max_size

        # Total transitions ever stored. Keeps counting past mem_size,
        # so it doubles as the write position (see store_transition).
        self.mem_cntr = 0

        # One array per field, all indexed by the same position.
        # Transition i lives at row i of every array.
        self.state_memory = np.zeros((max_size, input_size), dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, input_size), dtype=np.float32)
        self.action_memory = np.zeros((max_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.mask_memory = np.zeros(max_size, dtype=np.float32)

    def can_sample(self, batch_size):
        """
        Is there enough experience to train on yet?

        Requires five times the batch size so that early batches are not
        drawn from a handful of near-identical transitions.
        """
        return self.mem_cntr > batch_size * 5

    def store_transition(self, state, action, reward, next_state, mask):
        """Write one transition, overwriting the oldest once full."""

        # Wraps back to 0 after mem_size writes, so the buffer keeps the
        # most recent mem_size transitions and nothing older.
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.mask_memory[index] = mask

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """Draw a random batch of transitions for one training update."""

        # Before the buffer fills, rows past mem_cntr are still zeros.
        # Cap the draw so those empty rows are never sampled.
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        masks = self.mask_memory[batch]

        return states, actions, rewards, next_states, masks
