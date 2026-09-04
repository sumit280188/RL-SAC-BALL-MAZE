import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# Limits for the log standard deviation of the Gaussian policy
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# Small value used to avoid log(0)
EPSILON = 1e-6


# ---------------------------------------------------------
# Weight Initialization
# ---------------------------------------------------------

def weights_init_(m):
    """
    Initialize all Linear layers.

    Xavier initialization is used for weights.
    Biases are initialized to zero.
    """

    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# =========================================================
# Critic Network
# =========================================================

class Critic(nn.Module):

    def __init__(
        self,
        num_inputs,
        hidden_dim,
        num_actions,
        checkpoint_dir='checkpoints',
        name='critic_network'
    ):
        super(Critic, self).__init__()

        # -------------------------------------------------
        # Critic 1
        #
        # Input:
        #   state + action
        #
        # Architecture:
        #   (state + action) -> hidden_dim
        #                     -> hidden_dim
        #                     -> 1
        # -------------------------------------------------

        self.linear1 = nn.Linear(
            num_inputs + num_actions,
            hidden_dim
        )

        self.linear2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.linear3 = nn.Linear(
            hidden_dim,
            1
        )

        # -------------------------------------------------
        # Critic 2
        #
        # This is a completely separate network.
        #
        # Architecture:
        #   (state + action) -> hidden_dim
        #                     -> hidden_dim
        #                     -> 1
        # -------------------------------------------------

        self.linear4 = nn.Linear(
            num_inputs + num_actions,
            hidden_dim
        )

        self.linear5 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.linear6 = nn.Linear(
            hidden_dim,
            1
        )

        # -------------------------------------------------
        # Checkpoint information
        # -------------------------------------------------

        self.name = name
        self.checkpoint_dir = checkpoint_dir

        self.checkpoint_file = os.path.join(
            self.checkpoint_dir,
            name + '_sac'
        )

        # Initialize the weights of all Linear layers
        self.apply(weights_init_)

    def forward(self, state, action):
        """
        Calculate Q1(state, action) and Q2(state, action).
        """

        # -------------------------------------------------
        # Combine state and action
        #
        # Example:
        # state  -> [batch_size, num_inputs]
        # action -> [batch_size, num_actions]
        #
        # xu     -> [batch_size, num_inputs + num_actions]
        # -------------------------------------------------

        xu = torch.cat([state, action], 1)

        # -------------------------------------------------
        # Critic 1 forward pass
        # -------------------------------------------------

        x1 = self.linear1(xu)
        x1 = F.relu(x1)

        x1 = self.linear2(x1)
        x1 = F.relu(x1)

        x1 = self.linear3(x1)

        # -------------------------------------------------
        # Critic 2 forward pass
        # -------------------------------------------------

        x2 = self.linear4(xu)
        x2 = F.relu(x2)

        x2 = self.linear5(x2)
        x2 = F.relu(x2)

        x2 = self.linear6(x2)

        # Return both Q-value estimates
        return x1, x2

    def save_checkpoint(self):
        """Save the critic network weights."""

        print('... saving checkpoint ...')

        torch.save(
            self.state_dict(),
            self.checkpoint_file
        )

    def load_checkpoint(self):
        """Load the critic network weights."""

        print('... loading checkpoint ...')

        self.load_state_dict(
            torch.load(self.checkpoint_file,
                       map_location=next(self.parameters()).device)
        )


# =========================================================
# Actor Network
# =========================================================

class Actor(nn.Module):

    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        action_space=None,
        checkpoint_dir='checkpoints',
        name='actor_network'
    ):
        super(Actor, self).__init__()

        # -------------------------------------------------
        # Shared neural network
        #
        # state -> hidden_dim -> hidden_dim
        # -------------------------------------------------

        self.linear1 = nn.Linear(
            num_inputs,
            hidden_dim
        )

        self.linear2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        # -------------------------------------------------
        # Two output layers
        #
        # mean_linear:
        #   Produces the mean of the Gaussian distribution
        #
        # log_std_linear:
        #   Produces the log standard deviation
        # -------------------------------------------------

        self.mean_linear = nn.Linear(
            hidden_dim,
            num_actions
        )

        self.log_std_linear = nn.Linear(
            hidden_dim,
            num_actions
        )

        # -------------------------------------------------
        # Checkpoint information
        # -------------------------------------------------

        self.name = name
        self.checkpoint_dir = checkpoint_dir

        self.checkpoint_file = os.path.join(
            self.checkpoint_dir,
            name + '_sac'
        )

        # Initialize network weights
        self.apply(weights_init_)

        # -------------------------------------------------
        # Action scaling
        # -------------------------------------------------
        #
        # SAC internally produces actions in:
        #
        #       [-1, 1]
        #
        # These values are then converted to the
        # environment's actual action range.
        # -------------------------------------------------

        # register_buffer, not a plain attribute: only parameters and
        # registered buffers are moved by .to(device), so a plain tensor
        # would stay on the CPU and break every forward pass on the GPU.
        if action_space is None:

            self.register_buffer("action_scale", torch.tensor(1.))
            self.register_buffer("action_bias", torch.tensor(0.))

        else:

            self.register_buffer("action_scale", torch.FloatTensor(
                (action_space.high - action_space.low) / 2.
            ))

            self.register_buffer("action_bias", torch.FloatTensor(
                (action_space.high + action_space.low) / 2.
            ))

    def forward(self, state):
        """
        Given a state, calculate:

        mean
        log_std

        of the Gaussian policy.
        """

        # First hidden layer
        x = self.linear1(state)
        x = F.relu(x)

        # Second hidden layer
        x = self.linear2(x)
        x = F.relu(x)

        # Gaussian distribution parameters
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        # Keep log_std within a stable range
        log_std = torch.clamp(
            log_std,
            min=LOG_SIG_MIN,
            max=LOG_SIG_MAX
        )

        return mean, log_std

    def sample(self, state):
        """
        Sample an action from the policy.

        Returns:
            action
            log_prob
            mean
        """

        # -------------------------------------------------
        # Step 1: Get mean and log standard deviation
        # -------------------------------------------------

        mean, log_std = self.forward(state)

        # Convert:
        #
        # log_std -> std
        #
        # because:
        #
        # std = exp(log_std)
        # -------------------------------------------------

        std = log_std.exp()

        # -------------------------------------------------
        # Step 2: Create Gaussian distribution
        # -------------------------------------------------

        normal = Normal(mean, std)

        # -------------------------------------------------
        # Step 3: Sample from Gaussian distribution
        #
        # rsample() uses the reparameterization trick.
        #
        # This allows gradients to flow through the
        # sampled action during actor training.
        # -------------------------------------------------

        x_t = normal.rsample()

        # -------------------------------------------------
        # Step 4: Squash the action using tanh
        #
        # tanh converts the value approximately to:
        #
        #       [-1, 1]
        # -------------------------------------------------

        y_t = torch.tanh(x_t)

        # -------------------------------------------------
        # Step 5: Convert [-1, 1] to the environment's
        # actual action range.
        # -------------------------------------------------

        action = (
            y_t * self.action_scale
            + self.action_bias
        )

        # -------------------------------------------------
        # Step 6: Calculate log probability
        #
        # First calculate the Gaussian log probability.
        # -------------------------------------------------

        log_prob = normal.log_prob(x_t)

        # -------------------------------------------------
        # Correct the log probability because we applied
        # tanh to the sampled action.
        # -------------------------------------------------

        log_prob -= torch.log(
            self.action_scale *
            (1 - y_t.pow(2)) +
            EPSILON
        )

# same as network.py, but with the log_prob correction for tanh squashing(0)
#         1 - F.tanh²(u) = 4 / (eᵘ + e⁻ᵘ)²
#         log(1 - tanh²u) = log 4 - 2·log(eᵘ + e⁻ᵘ)
#                 = 2·log 2 - 2·log(eᵘ(1 + e⁻²ᵘ))
#                 = 2·log 2 - 2·[u + log(1 + e⁻²ᵘ)]
#                 = 2·(log 2 - u - softplus(-2u))

        # -------------------------------------------------
        # Sum the log probabilities of all action
        # dimensions.
        # -------------------------------------------------

        log_prob = log_prob.sum(
            1,
            keepdim=True
        )

        # -------------------------------------------------
        # Deterministic action based on the mean.
        #
        # This is useful during evaluation.
        # -------------------------------------------------

        mean = (
            torch.tanh(mean) *
            self.action_scale +
            self.action_bias
        )

        return action, log_prob, mean

    def to(self, device):
        """
        Move the Actor network and action-scaling
        tensors to the selected device.
        """

        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)

        return super(Actor, self).to(device)

    def save_checkpoint(self):
        """Save the actor network weights."""

        print('... saving checkpoint ...')

        torch.save(
            self.state_dict(),
            self.checkpoint_file
        )

    def load_checkpoint(self):
        """Load the actor network weights."""

        print('... loading checkpoint ...')

        self.load_state_dict(
            torch.load(self.checkpoint_file,
                       map_location=next(self.parameters()).device)
        )

class PredictiveModel(torch.nn.Module):
    """
    Predictive model for the environment dynamics.

    This model predicts the next state given the current state and action.
    """

    def __init__(self, num_inputs,
            num_actions,
            hidden_dim,
            checkpoint_dir='checkpoints',
            name='predictive_network'):
        super(PredictiveModel, self).__init__()

        self.fc1 = torch.nn.Linear(num_inputs + num_actions, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_inputs)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

    def forward(self, state, action):
        """
        Given a state and action, predict the next state.
        """

        # Combine state and action
        x = torch.cat([state, action], dim=1)

        # Forward pass through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        predicted_state = self.fc3(x)

        return predicted_state

    def save_checkpoint(self):
        """Save the predictive model weights."""

        print('... saving checkpoint ...')

        torch.save(
            self.state_dict(),
            self.checkpoint_file
        )

    def load_checkpoint(self):
        """Load the predictive model weights."""

        print('... loading checkpoint ...')

        self.load_state_dict(
            torch.load(self.checkpoint_file,
                       map_location=next(self.parameters()).device)
        )
