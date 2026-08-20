"""
Proximal Policy Optimization (PPO) -- draft code for the project.

Complete the code by replacing `None` (and the marked TODO lines) in the
blocks denoted by "Start your code here" and "End your code here".

PPO trains a Gaussian actor and a value critic on Pendulum-v1 using the
clipped-surrogate objective and Generalized Advantage Estimation (GAE).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym

from utilities import RolloutBuffer, RunningNormalizer
from plotting import plot_score


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(m.bias, 0)


class ActorCritic(nn.Module):
    """Gaussian actor with a state-independent log-std, plus a value critic."""

    def __init__(self, state_size, action_size, layer_size):
        super(ActorCritic, self).__init__()
        self.action_size = action_size

        # Actor trunk: two hidden layers.
        self.fca1 = nn.Linear(state_size, layer_size)
        self.fca2 = nn.Linear(layer_size, layer_size)
        # State-independent log standard deviation (a learnable parameter).
        self.log_std = nn.Parameter(torch.zeros(action_size))

        # Critic trunk: two hidden layers.
        self.fcv1 = nn.Linear(state_size, layer_size)
        self.fcv2 = nn.Linear(layer_size, layer_size)

        """Start your code here. 2 lines of code.
        Build the two output layers:
          - self.mean : maps the last actor hidden layer (size layer_size)
                        to the action mean (size action_size).
          - self.value: maps the last critic hidden layer (size layer_size)
                        to a single scalar value.
        """
        self.mean = None
        self.value = None
        """End your code here."""

        self.apply(weights_init_)
        # Smaller gain on the policy mean head for stable early training.
        torch.nn.init.orthogonal_(self.mean.weight, gain=0.01)

        self.log_sig_min = -20.0
        self.log_sig_max = 2.0

    def get_distribution(self, state):
        """Return the Gaussian action distribution Normal(mean, std)."""
        x = torch.tanh(self.fca1(state))
        x = torch.tanh(self.fca2(x))
        """Start your code here. 3-4 lines of code.
        1) Compute mean from x using self.mean.
        2) Clamp self.log_std between self.log_sig_min and self.log_sig_max.
        3) Take the exponential to get the std and expand it to the shape of mean.
        4) Return a Normal distribution with this mean and std.
        """
        dist = None
        """End your code here."""
        return dist

    def get_value(self, state):
        """Return the critic estimate V(state)."""
        v = torch.tanh(self.fcv1(state))
        v = torch.tanh(self.fcv2(v))
        return self.value(v).squeeze(-1)

    def evaluate_actions(self, state, action):
        """Return log-prob, entropy and value for given state-action pairs."""
        dist = self.get_distribution(state)
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.get_value(state)
        return log_prob, entropy, value


class PPO_Agent:
    """Interacts with and learns from the environment using clipped PPO."""

    def __init__(self, state_size, action_size, layer_size, LR, GAMMA,
                 GAE_LAMBDA, CLIP, ENT_COEF, VF_COEF, EPOCHS, MINIBATCH_SIZE,
                 ROLLOUT_SIZE, MAX_GRAD_NORM, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.GAMMA = GAMMA
        self.CLIP = CLIP
        self.ENT_COEF = ENT_COEF
        self.VF_COEF = VF_COEF
        self.EPOCHS = EPOCHS
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.MAX_GRAD_NORM = MAX_GRAD_NORM
        self.max_action = 2.0

        self.net = ActorCritic(state_size, action_size, layer_size).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)

        self.buffer = RolloutBuffer(ROLLOUT_SIZE, state_size, action_size,
                                    GAMMA, GAE_LAMBDA, device)
        self.obs_rms = RunningNormalizer(state_size)

    def act(self, state):
        """Sample an action and return (env_action, stored_action, value, logp)."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.net.get_distribution(state_t)
            raw_action = dist.sample()
            log_prob = dist.log_prob(raw_action).sum(axis=-1)
            value = self.net.get_value(state_t)
        raw_action = raw_action.squeeze(0).cpu().numpy()
        env_action = np.clip(raw_action, -self.max_action, self.max_action)
        return env_action, raw_action, value.item(), log_prob.item()

    def learn(self):
        """One PPO update: several epochs of minibatch SGD on the clipped loss."""
        states, actions, old_log_probs, advantages, returns, old_values = self.buffer.get()
        n = states.shape[0]
        idx = np.arange(n)
        last_loss = 0.0

        for _ in range(self.EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, n, self.MINIBATCH_SIZE):
                mb = idx[start:start + self.MINIBATCH_SIZE]
                mb_states = states[mb]
                mb_actions = actions[mb]
                mb_old_log_probs = old_log_probs[mb]
                mb_adv = advantages[mb]
                mb_returns = returns[mb]

                log_prob, entropy, value = self.net.evaluate_actions(mb_states, mb_actions)

                """Start your code here. 6 lines of code.
                1) ratio: exp of the difference between the new and old log-probs.
                2) surr1: ratio * mb_adv.
                3) surr2: clamp the ratio to [1 - CLIP, 1 + CLIP] and multiply by mb_adv.
                4) policy_loss: negative mean of the elementwise minimum of surr1 and surr2.
                5) value_loss: 0.5 * mean squared error between mb_returns and value.
                6) entropy_loss: negative mean of entropy (an exploration bonus).
                """
                ratio = None
                surr1 = None
                surr2 = None
                policy_loss = None
                value_loss = None
                entropy_loss = None
                """End your code here."""

                loss = policy_loss + self.VF_COEF * value_loss + self.ENT_COEF * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.MAX_GRAD_NORM)
                self.optimizer.step()
                last_loss = loss.item()
        return last_loss

    def collect_rollout(self, env, state):
        """Fill the rollout buffer, returning list of completed episode returns."""
        episode_returns = []
        ep_ret = 0.0
        observed_states = []
        while not self.buffer.is_full():
            observed_states.append(state)
            norm_state = self.obs_rms.normalize(state).astype(np.float32)
            env_action, raw_action, value, log_prob = self.act(norm_state)
            next_state, reward, done, truncated, _ = env.step(env_action)
            self.buffer.add(norm_state, raw_action, reward, value, log_prob, float(done))
            ep_ret += reward
            state = next_state
            if done or truncated:
                episode_returns.append(ep_ret)
                ep_ret = 0.0
                state, _ = env.reset()
        if observed_states:
            self.obs_rms.update(np.asarray(observed_states, dtype=np.float32))
        # Bootstrap value for the last state.
        norm_state = self.obs_rms.normalize(state).astype(np.float32)
        state_t = torch.as_tensor(norm_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = self.net.get_value(state_t).item()
        self.buffer.compute_returns_and_advantages(last_value)
        return state, episode_returns


def run_ppo(env):
    seed = 0
    LAYER_SIZE = 64
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP = 0.2
    LR = 3e-4
    ENT_COEF = 0.0
    VF_COEF = 0.5
    EPOCHS = 10
    MINIBATCH_SIZE = 64
    ROLLOUT_SIZE = 2048
    MAX_GRAD_NORM = 0.5
    NUMBER_OF_UPDATES = 120

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using", device)
    np.random.seed(seed)
    torch.manual_seed(seed)

    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    agent = PPO_Agent(state_size, action_size, LAYER_SIZE, LR, GAMMA,
                      GAE_LAMBDA, CLIP, ENT_COEF, VF_COEF, EPOCHS,
                      MINIBATCH_SIZE, ROLLOUT_SIZE, MAX_GRAD_NORM, device)

    scores = []
    state, _ = env.reset(seed=seed)
    for update in range(NUMBER_OF_UPDATES):
        state, episode_returns = agent.collect_rollout(env, state)
        agent.learn()
        if episode_returns:
            avg = float(np.mean(episode_returns))
            scores.append(avg)
            if update % 5 == 0:
                print(f"Update {update} \tAvg episode return: {avg:.2f}")
    plot_score(scores)
    return scores


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    run_ppo(env)
