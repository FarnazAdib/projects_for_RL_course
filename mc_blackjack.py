import gymnasium as gym
import numpy as np
from collections import defaultdict
from plotting import plot_smooth_performance
from tqdm import tqdm


class MC_agent:
    def __init__(
            self,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ):
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        self.n_actions = env.action_space.n

        self.q_values = defaultdict(lambda: np.zeros(self.n_actions))

    def rollout(self, env):
        obs, info = env.reset()
        done = False
        episode = []

        # play one episode
        while not done:
            # get the action
            action = np.random.choice(np.arange(self.n_actions), p=self.get_prob_actions(obs)) \
                if obs in self.q_values else env.action_space.sample()

            # step the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode.append((obs, action, reward))

            done = terminated or truncated
            obs = next_obs
        self.decay_epsilon()
        return episode

    def get_prob_actions(self, obs):
        policy_s = np.ones(self.n_actions) * self.epsilon / self.n_actions

        """start your code here"""
        best_a = None
        policy_s[best_a] = None
        """end your code here"""
        return policy_s

    def learn(self, episode):
        # observe that you can have multiple observations, actions and rewards from an episode
        obss, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([self.discount_factor ** i for i in range(len(rewards) + 1)])
        for i, obs in enumerate(obss):
            old_Q = self.q_values[obs][actions[i]]
            """start your code here. Appr. 1 line of code"""
            # Hint, in nonstationary environments, the q function is update as
            # q_values[obs][actions[i]] = old_Q + self.lr * ().
            self.q_values[obs][actions[i]] = old_Q + self.lr * (None)
            """end your code here"""
        self.training_error.append(0.0)


    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def run_mc(env):
    learning_rate = 0.01
    n_episodes = 500_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.05
    discount_factor = 1.0

    agent = MC_agent(learning_rate= learning_rate, initial_epsilon=start_epsilon, epsilon_decay=epsilon_decay,
                     final_epsilon=final_epsilon, discount_factor=discount_factor)

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    for i_episode in tqdm(range(n_episodes)):

        # rollout the env
        episode = agent.rollout(env)

        # learn
        agent.learn(episode)

    # compute and assign a rolling average of the data to provide a smoother graph
    rolling_length = 500
    plot_smooth_performance(env, agent, rolling_length)


if __name__ == "__main__":
    env = gym.make("Blackjack-v1", sab=True)
    run_mc(env)
