import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
from collections import deque, namedtuple
from utilities import OUNoise, OUActionNoise

import gymnasium as gym
import copy
from plotting import plot_losses, plot_score
from torch.distributions import MultivariateNormal, Normal


class DDPG_actor(nn.Module):
    def __init__(self, state_size, action_size, layer_size):
        super(DDPG_actor, self).__init__()

        self.state_size = state_size
        self.fca1 = nn.Linear(self.state_size, layer_size)
        self.fca2 = nn.Linear(layer_size, layer_size)
        """Start your code here. 1 line of code"""
        self.action_values = None
        """End your code here"""


    def forward(self, state):
        x = torch.relu(self.fca1(state))
        x = torch.relu(self.fca2(x))

        # Policy
        """Start your code here. 1 line of code."""
        action_value = None
        """End your code here."""
        return action_value


class DDPG_critic(nn.Module):
    def __init__(self, state_size, action_size, layer_size):
        super(DDPG_critic, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.fcs1 = nn.Linear(self.state_size, 16)
        self.fcs2 = nn.Linear(16, 32)

        self.fcu2 = nn.Linear(self.action_size, 32)

        self.fcc3 = nn.Linear(64, layer_size)
        self.fcc4 = nn.Linear(layer_size, layer_size)

        """Start your code here. 1 line of code."""
        self.qvalue = None
        """End your code here"""

    def forward(self, state, action):
        x = torch.relu(self.fcs1(state))
        x = torch.relu(self.fcs2(x))

        u = torch.relu(self.fcu2(action))

        z = torch.cat([x, u], 1)

        z = torch.relu(self.fcc3(z))
        z = torch.relu(self.fcc4(z))
        """Start your code here. 1 line of code."""
        Q = None
        """End your code here."""
        return Q

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


class DDPG_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 layer_size,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR_critic,
                 LR_actor,
                 TAU,
                 GAMMA,
                 device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))

        # Q-Networks
        self.qnetwork_local = DDPG_critic(state_size, action_size, layer_size).to(device)
        self.qnetwork_target = DDPG_critic(state_size, action_size, layer_size).to(device)
        self.optimizer_critic = optim.Adam(self.qnetwork_local.parameters(), lr=LR_critic)

        # Actor-Networks
        self.actor_network_local = DDPG_actor(state_size, action_size, layer_size).to(device)
        self.actor_network_target = DDPG_actor(state_size, action_size, layer_size).to(device)
        self.optimizer_actor = optim.Adam(self.actor_network_local.parameters(), lr=LR_actor)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, self.GAMMA)



    def get_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_network_local.eval()
        with torch.no_grad():
            action = self.actor_network_local(state.unsqueeze(0))
            noise = self.noise()
            action = action.cpu().squeeze().numpy() + noise
            action = np.clip(action, -2, 2)[0]
        self.actor_network_local.train()
        return action

    def learn(self, experiences):
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
            # The Q from the target network
            next_inputs = self.actor_network_target(next_states)
            Q_network_ = self.qnetwork_target(next_states, next_inputs)

        """Start your code here. 3 lines of code."""
        # The target Q for the current state using the target V
        Q_targets = None

        # The Q from the local network. We should find the gradient so we do not put it in no_grad
        Q = None

        # Compute the critic loss
        loss_critic = None
        """End your code here."""

        loss_critic.backward()
        # clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer_critic.step()

        ## Actor network
        self.optimizer_actor.zero_grad()

        """Start your code here. 3 lines of code."""
        actions_local = None
        q_local = None
        loss_actor = None
        """End your code"""

        # Minimize the critic and actor losses
        loss_actor.backward()
        self.optimizer_actor.step()


        # ------------------- update target network ------------------- #
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

        # ------------------- update target network ------------------- #
        for target_param, local_param in zip(self.actor_network_target.parameters(), self.actor_network_local.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

        # self.noise.reset()

        return loss_critic.detach().cpu().numpy()

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


def run_ddpg(env):
    #writer = SummaryWriter("runs/" + "DDPG_test")
    seed = 0
    LAYER_SIZE = 256
    BUFFER_SIZE = 50000
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 0.005
    LR_critic = 0.002
    LR_actor = 0.001
    NUMBER_OF_EPISODES = 200
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    agent = DDPG_Agent(state_size=state_size,
                        action_size=action_size,
                        layer_size=LAYER_SIZE,
                        BATCH_SIZE=BATCH_SIZE,
                        BUFFER_SIZE=BUFFER_SIZE,
                        LR_critic=LR_critic,
                        LR_actor=LR_actor,
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
    run_ddpg(env)





