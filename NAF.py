import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
from collections import deque, namedtuple

import gymnasium as gym
import copy
from plotting import plot_losses, plot_score
from torch.distributions import MultivariateNormal, Normal


class NAF(nn.Module):
    def __init__(self, state_size, action_size, layer_size, device):
        super(NAF, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.fc1 = nn.Linear(self.state_size, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)

        """Start your code here. Approximately 3 lines of code"""
        self.action_values = None
        self.value = None
        self.matrix_entries = None
        """End your code here."""


    def forward(self, state, action=None):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        # Value function
        V = self.value(x)

        # Policy
        """Start you code here. 1 line of code."""
        # Hint: the action is between -2 and 2.
        action_value = None * torch.tanh(None)
        """End you code here."""
        action_value = action_value.unsqueeze(-1)

        # Building the square matrix
        # Create matrix L
        L = torch.zeros((state.shape[0], self.action_size, self.action_size)).to(self.device)

        # Get lower triagular indices
        tril_indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)

        # Get the matrix entries from the neural network and filling the L matrix
        """Start you code here. 1 line of code."""
        entries = None
        """End you code here."""
        L[:, tril_indices[0], tril_indices[1]] = entries
        L.diagonal(dim1=1, dim2=2).exp_()

        # calculate state-dependent, positive-definite square matrix
        P = L * L.transpose(2, 1)

        Q = None
        if action is not None:
            # calculate Advantage:
            A = (-0.5 * torch.matmul(torch.matmul((action.unsqueeze(-1) - action_value).transpose(2, 1), P),
                                     (action.unsqueeze(-1) - action_value))).squeeze(-1)
            """Start your code here. 1 line of code."""
            Q = None
            """End your code here. """
        # add noise to action mu:
        dist = MultivariateNormal(action_value.squeeze(-1), torch.inverse(P))
        # dist = Normal(action_value.squeeze(-1), 1)
        action = dist.sample()
        action = torch.clamp(action, min=-2.0, max=2.0)
        # action = action_value.squeeze(-1)

        return action, Q, V


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
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 layer_size,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE

        # Q-Network
        self.qnetwork_local = NAF(state_size, action_size, layer_size, device).to(device)
        self.qnetwork_target = NAF(state_size, action_size, layer_size, device).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # print(self.qnetwork_local)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, self.GAMMA)

    def rollout(self, env):
        state, info = env.reset()
        done = False
        truncated = False
        score = 0.0
        loss = 0.0
        Q_losses = []
        while not (done or truncated):

            # Get the action
            action = self.get_action(state)

            # Apply the action
            next_state, reward, done, truncated, _ = env.step(np.array([action]))

            # Add the sample to the memory
            self.memory.add(state, action, reward, next_state, done)

            # Sample from buffer and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                Q_losses.append(loss)
                #writer.add_scalar("Q_loss", np.mean(Q_losses))
                loss = np.mean(Q_losses)

            # Update the new state and score
            state = next_state
            score += reward
        return score, loss

    def get_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action, _, _ = self.qnetwork_local(state.unsqueeze(0))
        self.qnetwork_local.train()
        return action.cpu().squeeze().numpy()

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences

        # The Value for the next state from target model
        with torch.no_grad():
            _, _, V_ = self.qnetwork_target(next_states)

        """Start your code here. Approximately 3 lines of code"""
        # The target Q for the current state using the target V
        Q_targets = None

        # Get expected Q values from local model
        _, Q, _ = None

        # Compute loss
        loss = None
        """End your code here"""

        # Minimize the loss
        loss.backward()
        # clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

        return loss.detach().cpu().numpy()


def run_naf(env):
    seed = 0
    LAYER_SIZE = 64
    BUFFER_SIZE = 10000
    BATCH_SIZE = 128
    GAMMA = 0.975
    TAU = 0.0025
    LR = 1e-3
    NUMBER_OF_EPISODES = 200
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    agent = DQN_Agent(state_size=state_size,
                        action_size=action_size,
                        layer_size=LAYER_SIZE,
                        BATCH_SIZE=BATCH_SIZE,
                        BUFFER_SIZE=BUFFER_SIZE,
                        LR=LR,
                        TAU=TAU,
                        GAMMA=GAMMA,
                        device=device)

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(NUMBER_OF_EPISODES):
        # Run a rollout
        score, _ = agent.rollout(env)
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        if i_episode % 5 == 0:
            print('\rEpisode {} \tTotal reward: {:.2f} \tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_window)))
    plot_score(scores)


if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    run_naf(env)





