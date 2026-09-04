import datetime
import os

import torch
from torch.optim import Adam

from model import Actor, Critic, PredictiveModel
from buffer import ReplayBuffer
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def hard_update(target, source):
    """Copy every weight from source into target, exactly."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    """
    Nudge target a small step toward source.

        target = (1 - tau) * target + tau * source

    With tau = 0.005 the target moves 0.5% of the way per call, so it
    lags behind the critic and gives the training targets something
    slow-moving to aim at.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class Agent:
    """
    Soft Actor-Critic agent.

    Holds four networks' worth of state:

        policy            picks actions
        critic            scores (state, action) pairs
        critic_target     slow-moving copy of critic, used for stable targets
        predictive_model  predicts the next state; its error is the curiosity bonus

    plus one optimizer each.
    """

    def __init__(self, num_inputs, action_space, policy, hidden_dim,
                 gamma, tau, alpha, learning_rate, target_update_interval,
                 exploration_scaling_factor, device):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.device = device
        num_actions = action_space.shape[0]

        self.critic = Critic(num_inputs, hidden_dim, num_actions).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)

        # Starts as an exact copy of the critic, then only ever drifts
        # toward it via soft_update.
        self.critic_target = Critic(num_inputs, hidden_dim, num_actions,
                                    name='critic_target_network').to(self.device)
        hard_update(self.critic_target, self.critic)

        self.policy = Actor(num_inputs, num_actions, hidden_dim, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

        #Initialize the predictive model
        self.predictive_model = PredictiveModel(num_inputs, action_space.shape[0], hidden_dim).to(self.device)
        self.predictive_model_optim = Adam(self.predictive_model.parameters(), lr=learning_rate)

        self.exploration_scaling_factor = exploration_scaling_factor


    def select_action(self, state, evaluate=False):
        """
        Choose one action for one state.

        evaluate=False  sample from the policy      (explores, for training)
        evaluate=True   use the distribution mean   (deterministic, for scoring)
        """
        # unsqueeze(0) turns shape (8,) into (1, 8): the networks always
        # expect a batch dimension, even for a single state.
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)

        # Back to a plain numpy array of shape (num_actions,) for env.step().
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory: ReplayBuffer, batch_size, updates):
        """
        One SAC training step on a batch from the replay buffer.

        Not implemented yet. This is where gamma, alpha and soft_update
        get used.
        """

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        #predictive model update
        predicted_next_state = self.predictive_model(state_batch, action_batch)
        prediction_error = F.mse_loss(predicted_next_state, next_state_batch)
        prediction_error_no_reduction = F.mse_loss(predicted_next_state, next_state_batch, reduction='none')

        scaled_intrinsic_reward = prediction_error_no_reduction.mean(dim=1).detach()
        scaled_intrinsic_reward = self.exploration_scaling_factor * torch.reshape(scaled_intrinsic_reward, (batch_size, 1))

        reward_batch += scaled_intrinsic_reward

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            #"If I were in the next state S', what action would I take?"
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            #"How good would that future action be?"
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            #Target Q = reward + γ × future value : "I believe the correct answer should be ...."

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        #action_batch from the replay buffer is the action that was actually taken in that state, not the one the Actor would have chosen. The Critic is being asked: "How good was that action I actually took?"
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        #update critic network : This is simply the neural-network learning mechanism.
        self.critic_optim.zero_grad() #Remove gradients from the previous update.
        qf_loss.backward() #Calculate how every Critic weight contributed to this error.
        self.critic_optim.step() #Use those gradients to change the Critic's weights.

        #update predictive model network
        self.predictive_model_optim.zero_grad()
        prediction_error.backward()
        self.predictive_model_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch) #pi: the action currently produced by the Actor.
        #"What action should the Actor choose?"
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        #pi comes from the Actor, so the Critic is being asked: "How good is the action the Actor is currently producing?"
        #"Critics, tell me how good the Actor's chosen action is."
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        #Actor asks: "Which action should I produce that the Critic thinks is good, while still maintaining exploration?"

        #update the policy network.
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = torch.tensor(0.).to(self.device) #so alpha is fixed. SAC can also automatically learn alpha, but that part isn't implemented in this version.
        alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), prediction_error.item()

    def train(self, env, env_name, memory: ReplayBuffer, episodes=1000, batch_size=64,
              updates_per_step=1, summary_writer_name="", max_episode_steps=100,
              warmup=20):

        #TensorboardX logging
        summary_writer_name = f"runs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_" + summary_writer_name
        writer = SummaryWriter(summary_writer_name)

        #training loop
        total_numsteps = 0
        updates = 0

        for i_episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            done = False
            state, info = env.reset()

            while not done and episode_steps < max_episode_steps:
                if warmup > i_episode:
                    action = env.action_space.sample()  # Sample random action for warmup
                else:
                    action = self.select_action(state)  # Sample action from policy

                if memory.can_sample(batch_size=batch_size):
                    for i in range(updates_per_step):
                        qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha_tlogs, prediction_loss = self.update_parameters(memory, batch_size=batch_size, updates=updates)
                        writer.add_scalar('loss/critic_1', qf1_loss, updates)
                        writer.add_scalar('loss/critic_2', qf2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/alpha', alpha_loss, updates)
                        writer.add_scalar('alpha/tlogs', alpha_tlogs, updates)
                        writer.add_scalar('loss/prediction', prediction_loss, updates)
                        updates += 1

                next_state, reward, terminated, truncated, info = env.step(action) #step the environment
                done = terminated or truncated

                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                # Ignore the done signal if it comes from hitting the time horizon.
                # (that is, when it's an artificial terminal signal that isn't based on the agent's state)
                mask = 1 if episode_steps == max_episode_steps else float(not done)

                memory.store_transition(state, action, reward, next_state, mask)
                state = next_state

            writer.add_scalar('reward/episode', episode_reward, i_episode)
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

            if i_episode % 10 == 0:
                self.save_model()
                print("Models saved")

    def save_model(self):
        """Write all four networks to the checkpoints directory."""
        os.makedirs("checkpoints/", exist_ok=True)

        print("Saving models")
        self.policy.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()
        self.predictive_model.save_checkpoint()

    def load_checkpoint(self, evaluate=False):
        """
        Load saved networks if they exist.

        Missing checkpoints are fatal when evaluating, but fine when
        training, which just starts from scratch.
        """
        try:
            print("Loading models")
            self.policy.load_checkpoint()
            self.critic.load_checkpoint()
            self.critic_target.load_checkpoint()
            self.predictive_model.load_checkpoint()
            print("Models loaded successfully")
        except FileNotFoundError:
            if evaluate:
                raise Exception(
                    "No checkpoints found. Cannot evaluate without a trained model."
                )
            print("No checkpoints found. Starting from scratch.")

        # eval() and train() switch layer behaviour (dropout, batchnorm).
        # This network has neither, but the calls keep the habit correct.
        networks = [self.policy, self.critic, self.critic_target, self.predictive_model]
        for net in networks:
            net.eval() if evaluate else net.train()
