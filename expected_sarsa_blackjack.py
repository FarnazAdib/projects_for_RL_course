from __future__ import annotations
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from plotting import plot_smooth_performance


class ESARSA_agent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def rollout(self, env):
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            # get the action
            action = self.get_action(obs)

            # step the environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            self.learn(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        self.decay_epsilon()

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Greedy action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def learn(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):

        """Learns the Q-value of an action."""
        num_actions = env.action_space.n
        expected_term = 0

        q_max = np.max(self.q_values[next_obs])
        n_greedy_actions = 0
        for i in range(num_actions):
            if self.q_values[next_obs][i] == q_max:
                n_greedy_actions += 1

        """start your code here. Appr. 6 lines of code"""
        non_greedy_action_probability = None
        greedy_action_probability = None

        for i in range(num_actions):
            if self.q_values[next_obs][i] == q_max:
                expected_term += None
            else:
                expected_term += None

        temporal_difference = None
        self.q_values[obs][action] = None
        """end your code here"""

        self.training_error.append(temporal_difference)
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def run_qlearning(env):
# hyperparameters
    learning_rate = 0.01
    n_episodes = 500_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.01

    agent = ESARSA_agent(
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    for episode in tqdm(range(n_episodes)):
        agent.rollout(env)

    # compute and assign a rolling average of the data to provide a smoother graph
    rolling_length = 500
    plot_smooth_performance(env, agent,
                            rolling_length)


if __name__ == "__main__":
    env = gym.make("Blackjack-v1", sab=True)
    run_qlearning(env)
