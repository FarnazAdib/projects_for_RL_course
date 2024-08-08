import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from utilities import OUNoise, OUActionNoise

import gymnasium as gym
from plotting import plot_losses, plot_score


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Q_network(nn.Module):
    def __init__(self, state_size, action_size, layer_size):
        super(Q_network, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # Q1
        self.fcq1 = nn.Linear(self.state_size + self.action_size, layer_size)
        self.fcq2 = nn.Linear(layer_size, layer_size)
        self.q1_value = nn.Linear(layer_size, 1)

        # Q2
        """Start your code here. 3 lines of code"""


        self.q2_value = None
        """End your code here."""

        self.apply(weights_init_)

    def forward(self, state, action):
        z = torch.cat([state, action], 1)

        # build Q1
        z1 = torch.relu(self.fcq1(z))
        z1 = torch.relu(self.fcq2(z1))
        Q1 = self.q1_value(z1)

        # build Q2
        z2 = torch.relu(self.fcq3(z))
        z2 = torch.relu(self.fcq4(z2))
        Q2 = self.q2_value(z2)
        return Q1, Q2


class Gaussian_policy_network(nn.Module):
    def __init__(self, state_size, action_size, layer_size, epsilon=1e-6):
        super(Gaussian_policy_network, self).__init__()

        self.log_sig_max = 2.0
        self.log_sig_min = -20.0
        self.max_action = 2.0
        self.min_action = -2.0
        self.epsilon = epsilon

        self.state_size = state_size
        self.action_size = action_size

        self.fcp1 = nn.Linear(self.state_size, layer_size)
        self.fcp2 = nn.Linear(layer_size, layer_size)
        self.policy_mean = nn.Linear(layer_size, action_size)
        self.policy_log_std = nn.Linear(layer_size, action_size)

        self.apply(weights_init_)

        self.action_scale = torch.FloatTensor([(self.max_action - self.min_action) / 2.])
        self.action_bias = torch.FloatTensor([(self.max_action + self.min_action) / 2.])

    def forward(self, state):
        x = torch.relu(self.fcp1(state))
        x = torch.relu(self.fcp2(x))
        """Start your code here. 2-3 lines of code"""
        mean = None
        log_std = None
        """End your code here."""
        return mean, log_std

    def sample_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        """Start your code here"""
        x_t = None
        y_t = None
        action = None # Bounded action
        """End your code here"""

        # Enforcing Action Bound
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        # Enforcing Action Bound
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Gaussian_policy_network, self).to(device)


class Deterministic_policy_network(nn.Module):
    """
    For evaluaton only.
    """
    def __init__(self, state_size, action_size, layer_size):
        super(Deterministic_policy_network, self).__init__()

        self.log_sig_max = 2.0
        self.log_sig_min = -20.0
        self.max_action = 2.0
        self.min_action = -2.0

        self.state_size = state_size
        self.action_size = action_size

        self.fcp1 = nn.Linear(self.state_size, layer_size)
        self.fcp2 = nn.Linear(layer_size, layer_size)

        self.policy_mean = nn.Linear(layer_size, action_size)
        self.noise = torch.Tensor(action_size)

        self.apply(weights_init_)

        self.action_scale = torch.FloatTensor([(self.max_action - self.min_action) / 2.])
        self.action_bias = torch.FloatTensor([(self.max_action + self.min_action) / 2.])

    def forward(self, state):
        x = torch.relu(self.fcp1(state))
        x = torch.relu(self.fcp2(x))
        mean = torch.tanh(self.policy_mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample_action(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(Deterministic_policy_network, self).to(device)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, gamma):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # self.seed = random.seed(seed)
        self.gamma = gamma

        self.n_step_buffer = deque(maxlen=1)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == 1:
            state, action, reward, next_state, done = self.calc_multistep_return()

            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def calc_multistep_return(self):
        Return = 0
        for idx in range(1):
            Return += self.gamma ** idx * self.n_step_buffer[idx][2]

        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], Return, self.n_step_buffer[-1][3], \
            self.n_step_buffer[-1][4]

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class SAC_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 layer_size,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 ALPHA,
                 TAU,
                 GAMMA,
                 policy_type,
                 target_update_interval,
                 auto_tuning,
                 device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.alpha = ALPHA
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.target_update_interval = target_update_interval
        self.auto_tuning = auto_tuning
        self.n_steps = 0
        self.n_updates = 0
        self.start_steps = 1000

        # Q-Network-local
        self.qnetwork_local = Q_network(state_size, action_size, layer_size).to(device)
        self.optimizer_critic = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Q-network-target
        self.qnetwork_target = Q_network(state_size, action_size, layer_size).to(device)


        # Actor-Networks
        if policy_type == 'Gaussian':

            if self.auto_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(self.action_size).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = optim.Adam([self.log_alpha], lr=LR)

            self.policy = Gaussian_policy_network(self.state_size, self.action_size, layer_size).to(self.device)
            self.policy_optim = optim.Adam(self.policy.parameters(), lr=LR)
            print("Gaussian policy")

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = Deterministic_policy_network(self.state_size, self.action_size, layer_size).to(self.device)
            self.policy_optim = optim.Adam(self.policy.parameters(), lr=LR)
            print("Deterministic policy")

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, self.GAMMA)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample_action(state)
        else:
            _, _, action = self.policy.sample_action(state)
        return action.detach().cpu().numpy()[0]

    def learn(self, experiences, updates):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """

        # Unpack the sampled data
        states, actions, rewards, next_states, dones = experiences

        ## Critic network
        self.optimizer_critic.zero_grad()

        with torch.no_grad():
            """Start you code here. 4 lines of code"""
            next_state_action, next_state_log_pi, _ = None
            qf1_next_target, qf2_next_target = None
            v_target = None
            q_target = None
            """End your code here"""

        qf1, qf2 = self.qnetwork_local(states, actions)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, q_target)
        qf2_loss = F.mse_loss(qf2, q_target)
        qf_loss = qf1_loss + qf2_loss

        self.optimizer_critic.zero_grad()
        qf_loss.backward()
        self.optimizer_critic.step()

        pi, log_pi, _ = self.policy.sample_action(states)

        qf1_pi, qf2_pi = self.qnetwork_local(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.auto_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
                target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def rollout(self, env):
        state, info = env.reset()
        done = False
        truncated = False
        score = 0.0
        loss = 0.0
        Q1_losses = []
        Q2_losses = []
        Q_losses =[]
        entropy_losses = []
        policy_losses = []

        while not (done or truncated):

            # Get the action
            if self.start_steps > self.n_steps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = self.select_action(state)  # Sample action from policy

            # Apply the action
            next_state, reward, done, truncated, _ = env.step(action)

            # Add the sample to the memory
            self.memory.add(state, action, reward, next_state, done)

            # Sample from buffer and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()

                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.learn(experiences, self.n_updates)

                Q1_losses.append(critic_1_loss)
                Q2_losses.append(critic_2_loss)
                Q_losses.append((critic_1_loss+critic_2_loss)/2)
                policy_losses.append(policy_loss)
                entropy_losses.append(ent_loss)

                loss = np.mean(Q_losses)
                self.n_updates += 1

            # Update the new state and score
            state = next_state
            score += reward
            self.n_steps += 1
        return score, loss


def run_sac(env):
    seed = 0
    LAYER_SIZE = 256
    BUFFER_SIZE = 50000
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 0.005
    LR = 0.001
    ALPHA = 0.1
    NUMBER_OF_EPISODES = 200
    policy_type = 'Gaussian'
    target_update_interval = 1
    auto_tuning = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    agent = SAC_Agent(state_size=state_size,
                      action_size=action_size,
                      layer_size=LAYER_SIZE,
                      BATCH_SIZE=BATCH_SIZE,
                      BUFFER_SIZE=BUFFER_SIZE,
                      LR=LR,
                      ALPHA=ALPHA,
                      TAU=TAU,
                      GAMMA=GAMMA,
                      policy_type=policy_type,
                      target_update_interval=target_update_interval,
                      auto_tuning=auto_tuning,
                      device=device)

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(NUMBER_OF_EPISODES):
        # Run a rollout
        score, _ = agent.rollout(env)
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        if i_episode % 5 == 0:
            print('\rEpisode {} \tTotal reward: {:.2f} \tAverage Score: {:.2f}'.format(i_episode, score,
                                                                                       np.mean(scores_window)))
    plot_score(scores)


if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    run_sac(env)
