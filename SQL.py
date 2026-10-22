import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple

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

        self.fcq1 = nn.Linear(self.state_size + self.action_size, layer_size)
        self.fcq2 = nn.Linear(layer_size, layer_size)
        self.q_value = nn.Linear(layer_size, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        """
        state:  (..., state_size)
        action: (..., action_size)
        returns Q: (..., 1)
        """
        z = torch.cat([state, action], -1)
        z = torch.relu(self.fcq1(z))
        z = torch.relu(self.fcq2(z))
        Q = self.q_value(z)
        return Q


class Sampling_network(nn.Module):
    """
    Amortized sampler a = f(s, xi), xi ~ N(0, I). It is trained with SVGD so that
    the distribution of its outputs approaches pi(a|s) ∝ exp(Q(s, a) / alpha).
    """
    def __init__(self, state_size, action_size, layer_size):
        super(Sampling_network, self).__init__()

        self.max_action = 2.0
        self.state_size = state_size
        self.action_size = action_size

        self.fcp1 = nn.Linear(self.state_size + self.action_size, layer_size)
        self.fcp2 = nn.Linear(layer_size, layer_size)
        self.action_out = nn.Linear(layer_size, action_size)

        self.apply(weights_init_)

    def forward(self, state, n_particles=1):
        """
        state: (N, state_size)
        returns actions: (N, n_particles, action_size) in [-max_action, max_action]
        """
        state = state.unsqueeze(1).expand(-1, n_particles, -1)
        xi = torch.randn(state.shape[0], n_particles, self.action_size, device=state.device)
        x = torch.cat([state, xi], -1)
        x = torch.relu(self.fcp1(x))
        x = torch.relu(self.fcp2(x))
        """Start your code here. 1 line of code"""
        # Hint: the action is between -2 and 2.
        actions = None
        """End your code here."""
        return actions


def adaptive_isotropic_gaussian_kernel(xs, ys):
    """
    RBF kernel k(x, y) = exp(-||x - y||^2 / h) with the bandwidth h chosen by the median heuristic.
    xs: (N, Kx, D), ys: (N, Ky, D)
    returns
        kappa:      (N, Kx, Ky)     kernel values
        kappa_grad: (N, Kx, Ky, D)  gradient of the kernel with respect to xs
    """
    Kx = xs.shape[1]
    diff = xs.unsqueeze(2) - ys.unsqueeze(1)  # (N, Kx, Ky, D)
    dist_sq = (diff ** 2).sum(-1)  # (N, Kx, Ky)

    # Median heuristic for the bandwidth (one bandwidth per state)
    h = dist_sq.flatten(1).median(dim=1).values / np.log(Kx + 1)
    h = (h + 1e-6).detach().view(-1, 1, 1)

    """Start your code here. 2 lines of code"""
    kappa = None
    kappa_grad = None
    """End your code here."""
    return kappa, kappa_grad


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


class SQL_Agent():
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
                 value_n_particles,
                 kernel_n_particles,
                 kernel_update_ratio,
                 device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.alpha = ALPHA
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.value_n_particles = value_n_particles
        self.kernel_n_particles = kernel_n_particles
        self.kernel_update_ratio = kernel_update_ratio
        self.max_action = 2.0
        self.n_steps = 0
        self.start_steps = 1000

        # Q-Network-local
        self.qnetwork_local = Q_network(state_size, action_size, layer_size).to(device)
        self.optimizer_critic = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Q-network-target
        self.qnetwork_target = Q_network(state_size, action_size, layer_size).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # Sampling network (the policy)
        self.policy = Sampling_network(state_size, action_size, layer_size).to(device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, self.GAMMA)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(state, n_particles=1)
        return action.cpu().numpy()[0, 0]

    def soft_value(self, network, states):
        """
        Soft value V(s) = alpha * log int exp(Q(s, a) / alpha) da, estimated by importance
        sampling with a uniform proposal q(a) over [-max_action, max_action]^D:
            V(s) ≈ alpha * log( 1/M * sum_j exp(Q(s, a_j) / alpha) / q(a_j) ),   a_j ~ q
        """
        M = self.value_n_particles
        actions = (2 * torch.rand(states.shape[0], M, self.action_size, device=self.device) - 1) * self.max_action
        states = states.unsqueeze(1).expand(-1, M, -1)
        Q = network(states, actions).squeeze(-1)  # (N, M)

        log_q = -self.action_size * np.log(2 * self.max_action)  # log density of the uniform proposal
        """Start your code here. 1 line of code"""
        V = None
        """End your code here."""
        return V

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """

        # Unpack the sampled data
        states, actions, rewards, next_states, dones = experiences

        ## Soft Q-function
        with torch.no_grad():
            """Start your code here. 2 lines of code"""
            V_next = None
            q_target = None
            """End your code here"""

        Q = self.qnetwork_local(states, actions)
        q_loss = F.mse_loss(Q, q_target)

        self.optimizer_critic.zero_grad()
        q_loss.backward()
        self.optimizer_critic.step()

        ## Sampling network with amortized Stein variational gradient descent (SVGD)
        K = self.kernel_n_particles
        n_fixed = int(K * self.kernel_update_ratio)
        sampled_actions = self.policy(states, n_particles=K)  # (N, K, D)

        # The first n_fixed particles define the empirical distribution, the rest are moved by SVGD
        fixed_actions = sampled_actions[:, :n_fixed].detach().requires_grad_(True)
        updated_actions = sampled_actions[:, n_fixed:]

        # Target log-density log p(a|s) = Q(s, a) / alpha, plus the correction for the tanh squashing
        states_fixed = states.unsqueeze(1).expand(-1, n_fixed, -1)
        log_p = self.qnetwork_local(states_fixed, fixed_actions).squeeze(-1) / self.alpha
        squash_correction = torch.log(1 - (fixed_actions / self.max_action) ** 2 + 1e-6).sum(-1)
        log_p = log_p + squash_correction
        grad_log_p = torch.autograd.grad(log_p.sum(), fixed_actions)[0].unsqueeze(2)  # (N, n_fixed, 1, D)

        kappa, kappa_grad = adaptive_isotropic_gaussian_kernel(fixed_actions.detach(), updated_actions.detach())
        kappa = kappa.unsqueeze(-1)  # (N, n_fixed, n_updated, 1)

        """Start your code here. 2 lines of code"""
        # SVGD direction: phi(a_i) = E_{a_j}[ k(a_j, a_i) grad log p(a_j) + grad_{a_j} k(a_j, a_i) ]
        action_gradients = None  # (N, n_updated, D)
        # Surrogate loss whose gradient wrt the policy parameters is -phi * d(action)/d(params)
        policy_loss = None
        """End your code here."""

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # ------------------- update target network ------------------- #
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

        return q_loss.item(), policy_loss.item()

    def rollout(self, env):
        state, info = env.reset()
        done = False
        truncated = False
        score = 0.0
        loss = 0.0
        Q_losses = []
        policy_losses = []

        while not (done or truncated):

            # Get the action
            if self.start_steps > self.n_steps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = self.select_action(state)  # Sample action from the sampling network

            # Apply the action
            next_state, reward, done, truncated, _ = env.step(action)

            # Add the sample to the memory
            self.memory.add(state, action, reward, next_state, done)

            # Sample from buffer and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()

                q_loss, policy_loss = self.learn(experiences)

                Q_losses.append(q_loss)
                policy_losses.append(policy_loss)

                loss = np.mean(Q_losses)

            # Update the new state and score
            state = next_state
            score += reward
            self.n_steps += 1
        return score, loss


def run_sql(env):
    seed = 0
    LAYER_SIZE = 256
    BUFFER_SIZE = 50000
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 0.005
    LR = 0.001
    ALPHA = 0.1
    VALUE_N_PARTICLES = 16
    KERNEL_N_PARTICLES = 32
    KERNEL_UPDATE_RATIO = 0.5
    NUMBER_OF_EPISODES = 200
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    agent = SQL_Agent(state_size=state_size,
                      action_size=action_size,
                      layer_size=LAYER_SIZE,
                      BATCH_SIZE=BATCH_SIZE,
                      BUFFER_SIZE=BUFFER_SIZE,
                      LR=LR,
                      ALPHA=ALPHA,
                      TAU=TAU,
                      GAMMA=GAMMA,
                      value_n_particles=VALUE_N_PARTICLES,
                      kernel_n_particles=KERNEL_N_PARTICLES,
                      kernel_update_ratio=KERNEL_UPDATE_RATIO,
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
    run_sql(env)
