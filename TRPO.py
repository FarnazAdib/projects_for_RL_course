"""
Trust Region Policy Optimization (TRPO) -- draft code for the project.

Complete the code by replacing `None` in the blocks denoted by
"Start your code here" and "End your code here".

TRPO takes the largest policy improvement step that stays inside a trust
region defined by a KL-divergence constraint. The natural-gradient step is
found with conjugate gradient (CG) on the Fisher-vector product (FVP) and is
then shrunk by a backtracking line search that enforces the KL constraint.
A separate value network provides the GAE advantages.
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


class PolicyNetwork(nn.Module):
    """Gaussian policy with state-independent log-std."""

    def __init__(self, state_size, action_size, layer_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))

        """Start your code here. 1 line of code.
        Build the output layer self.mean mapping the last hidden layer
        (size layer_size) to the action mean (size action_size).
        """
        self.mean = None
        """End your code here."""

        self.apply(weights_init_)
        torch.nn.init.orthogonal_(self.mean.weight, gain=0.01)
        self.log_sig_min = -20.0
        self.log_sig_max = 2.0

    def get_distribution(self, state):
        """Return the Gaussian action distribution Normal(mean, std)."""
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        """Start your code here. 3-4 lines of code.
        1) Compute mean from x using self.mean.
        2) Clamp self.log_std between self.log_sig_min and self.log_sig_max.
        3) Take the exponential to get the std and expand it to the shape of mean.
        4) Return a Normal distribution with this mean and std.
        """
        dist = None
        """End your code here."""
        return dist


class ValueNetwork(nn.Module):
    def __init__(self, state_size, layer_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.value = nn.Linear(layer_size, 1)
        self.apply(weights_init_)

    def forward(self, state):
        v = torch.tanh(self.fc1(state))
        v = torch.tanh(self.fc2(v))
        return self.value(v).squeeze(-1)


class TRPO_Agent:
    def __init__(self, state_size, action_size, layer_size, VF_LR, GAMMA,
                 GAE_LAMBDA, MAX_KL, CG_ITERS, CG_DAMPING, BACKTRACK_ITERS,
                 BACKTRACK_COEF, VF_ITERS, ROLLOUT_SIZE, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.GAMMA = GAMMA
        self.MAX_KL = MAX_KL
        self.CG_ITERS = CG_ITERS
        self.CG_DAMPING = CG_DAMPING
        self.BACKTRACK_ITERS = BACKTRACK_ITERS
        self.BACKTRACK_COEF = BACKTRACK_COEF
        self.VF_ITERS = VF_ITERS
        self.max_action = 2.0

        self.policy = PolicyNetwork(state_size, action_size, layer_size).to(device)
        self.value_net = ValueNetwork(state_size, layer_size).to(device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=VF_LR)

        self.buffer = RolloutBuffer(ROLLOUT_SIZE, state_size, action_size,
                                    GAMMA, GAE_LAMBDA, device)
        self.obs_rms = RunningNormalizer(state_size)

    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.policy.get_distribution(state_t)
            raw_action = dist.sample()
            log_prob = dist.log_prob(raw_action).sum(axis=-1)
            value = self.value_net(state_t)
        raw_action = raw_action.squeeze(0).cpu().numpy()
        env_action = np.clip(raw_action, -self.max_action, self.max_action)
        return env_action, raw_action, value.item(), log_prob.item()

    # ---- TRPO core -------------------------------------------------------
    def surrogate_loss(self, states, actions, advantages, old_log_probs):
        """Importance-sampled policy advantage (to be maximized).

        L(theta) = E[ pi_theta(a|s) / pi_old(a|s) * A ].
        """
        dist = self.policy.get_distribution(states)
        log_prob = dist.log_prob(actions).sum(axis=-1)
        """Start your code here. 2 lines of code.
        1) ratio: exp of the difference between the new and old log-probs.
        2) return the mean of ratio * advantages.
        """
        ratio = None
        return None
        """End your code here."""

    def mean_kl(self, states, old_dist):
        """Mean KL divergence D_KL(old || new) over the batch."""
        new_dist = self.policy.get_distribution(states)
        kl = torch.distributions.kl_divergence(old_dist, new_dist).sum(axis=-1)
        return kl.mean()

    def flat_grad(self, y, params, retain_graph=None, create_graph=False):
        grads = torch.autograd.grad(y, params, retain_graph=retain_graph,
                                    create_graph=create_graph, allow_unused=True)
        flat = []
        for g, p in zip(grads, params):
            if g is None:
                flat.append(torch.zeros_like(p).view(-1))
            else:
                flat.append(g.contiguous().view(-1))
        return torch.cat(flat)

    def fisher_vector_product(self, states, old_dist, vector):
        """Compute F @ vector, where F is the Fisher information matrix.

        F is the Hessian of the mean KL divergence. We never form F
        explicitly; instead we use two automatic-differentiation passes:
          1) grad_kl = d(KL)/d(theta)
          2) F @ v   = d( grad_kl . v )/d(theta)
        A small damping term (CG_DAMPING) is added for numerical stability.
        """
        params = list(self.policy.parameters())
        kl = self.mean_kl(states, old_dist)
        grad_kl = self.flat_grad(kl, params, create_graph=True)
        grad_vector_product = (grad_kl * vector).sum()
        fvp = self.flat_grad(grad_vector_product, params, retain_graph=True)
        return fvp + self.CG_DAMPING * vector

    def conjugate_gradient(self, states, old_dist, b):
        """Solve F x = b for x using conjugate gradient (F given via FVP)."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for _ in range(self.CG_ITERS):
            fvp = self.fisher_vector_product(states, old_dist, p)
            alpha = rdotr / (torch.dot(p, fvp) + 1e-8)
            x += alpha * p
            r -= alpha * fvp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            p = r + (new_rdotr / rdotr) * p
            rdotr = new_rdotr
        return x

    def get_flat_params(self):
        return torch.cat([p.data.view(-1) for p in self.policy.parameters()])

    def set_flat_params(self, flat_params):
        idx = 0
        for p in self.policy.parameters():
            n = p.numel()
            p.data.copy_(flat_params[idx:idx + n].view(p.shape))
            idx += n

    def learn(self):
        states, actions, old_log_probs, advantages, returns, _ = self.buffer.get()

        # ----- policy update via natural gradient + line search -----
        with torch.no_grad():
            old_dist = self.policy.get_distribution(states)
            old_dist = Normal(old_dist.mean.detach(), old_dist.stddev.detach())

        loss = self.surrogate_loss(states, actions, advantages, old_log_probs)
        params = list(self.policy.parameters())
        g = self.flat_grad(loss, params, retain_graph=True)

        # Natural-gradient direction step_dir = F^{-1} g via conjugate gradient.
        step_dir = self.conjugate_gradient(states, old_dist, g)

        """Start your code here. 3 lines of code.
        Rescale step_dir so that the induced KL is (at most) MAX_KL.
        Using shs = 0.5 * step_dir^T F step_dir:
          1) shs: 0.5 * dot(step_dir, F @ step_dir). Use fisher_vector_product.
          2) lagrange_multiplier: sqrt( shs / MAX_KL ).
          3) full_step: step_dir / lagrange_multiplier.
        """
        shs = None
        lagrange_multiplier = None
        full_step = None
        """End your code here."""

        old_params = self.get_flat_params()

        # Backtracking line search enforcing the KL constraint.
        success = False
        for i in range(self.BACKTRACK_ITERS):
            step_frac = self.BACKTRACK_COEF ** i
            new_params = old_params + step_frac * full_step
            self.set_flat_params(new_params)
            with torch.no_grad():
                new_loss = self.surrogate_loss(states, actions, advantages, old_log_probs)
                kl = self.mean_kl(states, old_dist)
            improvement = new_loss - loss
            if kl <= self.MAX_KL and improvement > 0:
                success = True
                break
        if not success:
            self.set_flat_params(old_params)

        # ----- value function update (simple regression on the returns) -----
        for _ in range(self.VF_ITERS):
            value_pred = self.value_net(states)
            value_loss = (returns - value_pred).pow(2).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        return value_loss.item()

    def collect_rollout(self, env, state):
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
        norm_state = self.obs_rms.normalize(state).astype(np.float32)
        state_t = torch.as_tensor(norm_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            last_value = self.value_net(state_t).item()
        self.buffer.compute_returns_and_advantages(last_value)
        return state, episode_returns


def run_trpo(env):
    seed = 0
    LAYER_SIZE = 64
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    MAX_KL = 0.02
    CG_ITERS = 10
    CG_DAMPING = 0.1
    BACKTRACK_ITERS = 10
    BACKTRACK_COEF = 0.8
    VF_LR = 3e-4
    VF_ITERS = 80
    ROLLOUT_SIZE = 2048
    NUMBER_OF_UPDATES = 150

    device = torch.device("cpu")
    print("Using", device)
    np.random.seed(seed)
    torch.manual_seed(seed)

    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    agent = TRPO_Agent(state_size, action_size, LAYER_SIZE, VF_LR, GAMMA,
                       GAE_LAMBDA, MAX_KL, CG_ITERS, CG_DAMPING,
                       BACKTRACK_ITERS, BACKTRACK_COEF, VF_ITERS,
                       ROLLOUT_SIZE, device)

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
    run_trpo(env)
